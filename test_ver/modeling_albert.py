# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ALBERT model. """

import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_albert import AlbertConfig
from transformers.modeling_bert import ACT2FN, BertEmbeddings, BertSelfAttention, prune_linear_layer
from transformers.modeling_utils import PreTrainedModel

from .file_utils import add_start_docstrings, add_start_docstrings_to_callable

# ****
import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.nn.functional import softmax
from .data.metrics.__init__ import glue_compute_metrics as compute_metrics

logger = logging.getLogger(__name__)


ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "albert-base-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-pytorch_model.bin",
    "albert-large-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-pytorch_model.bin",
    "albert-xlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-pytorch_model.bin",
    "albert-xxlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-pytorch_model.bin",
    "albert-base-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-pytorch_model.bin",
    "albert-large-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-pytorch_model.bin",
    "albert-xlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-pytorch_model.bin",
    "albert-xxlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-pytorch_model.bin",
}


def softmax_T(x, dim=1, T=1):
    "蒸馏版softmax"
    y = torch.exp(x / T) / torch.sum(torch.exp(x / T), dim=dim).reshape(-1, 1)
    return y


class KL_loss(nn.Module):
    def __init__(self):
        super().__init__()

    # input:original logit
    def forward(self, p_logit, q_logit):
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                             - F.log_softmax(q_logit, dim=-1)), 1)
        return (_kl).mean()


class ExitConfig(object):
    def __init__(self, config_path):
        f = open(config_path)
        d = json.load(f)
        self.shared = d['shared']
        self.thres_name = d['thres_name']
        self.exit_layer_name = d['exit_layer_name']
        self.cnt_thres = d['cnt_thres']
        self.cls_hidden_size = d['cls_hidden_size']
        self.cls_num_attention_heads = d['cls_num_attention_heads']
        self.kd_enable = d['kd_enable']
        self.kd_T = d['kd_T']
        self.kd_range = d['kd_range']
        self.kd_T_s = d['kd_T_s']
        self.kd_T_l = d['kd_T_l']
        self.kd_T_idx_s = d['kd_T_idx_s']
        self.margin = d['margin']
        self.exit_thres = d['exit_thres']
        self.use_fc2 = d['use_fc2']
        self.fc_size1 = d['fc_size1']
        self.fc_size2 = d['fc_size2']
        self.w_lr = d['w_lr']
        self.w_init = d['w_init']
        self.use_dml = d['use_dml']
        self.num_teachers = d['num_teachers']
        self.use_label_smooth = d['use_label_smooth']
        self.e_label_smoothing = d['e_label_smoothing']
        self.model_name = d['model_name']
        self.my_out_dir = d['my_out_dir']
        self.num_exit = d['num_exit']
        self.idx_lst = d['idx_lst']
        # print(self.model_name)
        # print(self.num_teachers)
        # print(self.idx_lst)
        # input()


class Entropy(nn.Module):
    "自实现二分类交叉熵，支持浮点标签, 以及蒸馏处理"

    def __init__(self):
        super().__init__()

    def softmax_T(self, x, dim=1, T=1):
        "蒸馏版softmax"
        y = torch.exp(x / T) / torch.sum(torch.exp(x / T),
                                         dim=dim).reshape(-1, 1)
        return y

    def forward(self, x, y, T=1):
        # x = softmax(x, dim=-1)
        x = self.softmax_T(x, dim=-1, T=T)
        loss = -(1 - y) * torch.log(x[:, 0]) - y * torch.log(x[:, 1])
        return loss.mean()


# class KL_loss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     # input:original logit
#     def forward(self, p_logit, q_logit):
#         p = F.softmax(p_logit, dim=-1)
#         _kl = torch.sum(
#             p * (F.log_softmax(p_logit, dim=-1)
#                  - F.log_softmax(q_logit, dim=-1)), 1
#         )
#         return (_kl).mean()


def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        print(name)

    for name, array in zip(names, arrays):
        original_name = name

        # If saved from the TF HUB module
        name = name.replace("module/", "")

        # Renaming and simplifying
        name = name.replace("ffn_1", "ffn")
        name = name.replace("bert/", "albert/")
        name = name.replace("attention_1", "attention")
        name = name.replace("transform/", "")
        name = name.replace("LayerNorm_1", "full_layer_layer_norm")
        name = name.replace("LayerNorm", "attention/LayerNorm")
        name = name.replace("transformer/", "")

        # The feed forward layer had an 'intermediate' step which has been abstracted away
        name = name.replace("intermediate/dense/", "")
        name = name.replace("ffn/intermediate/output/dense/", "ffn_output/")

        # ALBERT attention was split between self and output which have been abstracted away
        name = name.replace("/output/", "/")
        name = name.replace("/self/", "/")

        # The pooler is a linear layer
        name = name.replace("pooler/dense", "pooler")

        # The classifier was simplified to predictions from cls/predictions
        name = name.replace("cls/predictions", "predictions")
        name = name.replace("predictions/attention", "predictions")

        # Naming was changed to be more explicit
        name = name.replace("embeddings/attention", "embeddings")
        name = name.replace("inner_group_", "albert_layers/")
        name = name.replace("group_", "albert_layer_groups/")

        # Classifier
        if len(name.split("/")) == 1 and ("output_bias" in name or "output_weights" in name):
            name = "classifier/" + name

        # No ALBERT model currently handles the next sentence prediction task
        if "seq_relationship" in name:
            continue

        name = name.split("/")

        # Ignore the gradients applied by the LAMB/ADAM optimizers.
        if (
            "adam_m" in name
            or "adam_v" in name
            or "AdamWeightDecayOptimizer" in name
            or "AdamWeightDecayOptimizer_1" in name
            or "global_step" in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue

        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {} from {}".format(name, original_name))
        pointer.data = torch.from_numpy(array)

    return model


class AlbertEmbeddings(BertEmbeddings):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__(config)

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size)
        self.LayerNorm = torch.nn.LayerNorm(
            config.embedding_size, eps=config.layer_norm_eps)


class AlbertAttention(BertSelfAttention):
    def __init__(self, config, hidden_size=None, num_attention_heads=None):
        super().__init__(config)
        if hidden_size is None:
            hidden_size = config.hidden_size
        if num_attention_heads is None:
            num_attention_heads = config.num_attention_heads

        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        # Convert to set and emove already pruned heads
        heads = set(heads) - self.pruned_heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum(
            "bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(
            input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)


class BERT_SelfAttention(nn.Module):
    def __init__(self, config, hidden_size=None, num_attention_heads=None):
        super().__init__()
        if hidden_size == None:
            hidden_size = config.hidden_size
        if num_attention_heads == None:
            num_attention_heads = config.num_attention_heads

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, use_attention_mask=True):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if use_attention_mask:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.full_layer_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(
            config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(
            hidden_states, attention_mask, head_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(
            ffn_output + attention_output[0])

        # add attentions if we output them
        return (hidden_states,) + attention_output[1:]


class AlbertLayerGroup(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 根据需要，是否附加输出中间信息
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        # 实际上就1层
        self.albert_layers = nn.ModuleList(
            [AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(
                hidden_states, attention_mask, head_mask[layer_index])
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        # last-layer hidden state, (layer hidden states), (layer attentions)
        return outputs


class AlbertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "albert"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ExitLayer(nn.Module):
    """
    output:(logits,)
    退出层类,目前是插入到trm内部

    """

    def __init__(self, config, exit_config):
        super().__init__()
        self.config = config
        self.exit_config = exit_config
        self.exit_thres = config.exit_thres
        self.num_labels = config.num_labels

        if not config.use_out_pooler:
            self.pooler = nn.Linear(config.hidden_size, config.fc_size1)
        self.pooler_activation = nn.Tanh()
        # 分类器
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        if config.use_fc2:
            self.classifier1 = nn.Linear(config.fc_size1, config.fc_size2)
            self.classifier2 = nn.Linear(config.fc_size2, config.num_labels)
        else:
            # 原始输出结构 单FC
            self.classifier = nn.Linear(config.fc_size1, config.num_labels)

        # 统计数据
        self.exit_loss = 0
        self.entropy = 0
        self.softm_logits = 0

        # 退出相关
        self.exit_cnt_dict = {"entropy": 0, "ori": 0}  # 每次的样本退出情况
        self.is_right = False

    def forward(self, encoder_outputs, T=1, pooler=None):
        sequence_output = encoder_outputs[0]    # 从tuple里取第一个
        if self.config.pooler_input == "cls":
            pool_input = sequence_output[:, 0]
        elif self.config.pooler_input == "cls-mean":
            t1 = sequence_output[:, 0]
            t2 = sequence_output.mean(dim=1)
            pool_input = t1 + t2
        elif self.config.pooler_input == "cls-max":
            t1 = sequence_output[:, 0]
            t2 = torch.max(sequence_output, dim=1)[0]
            pool_input = t1 + t2
        elif self.config.pooler_input == "mean":
            pool_input = sequence_output.mean(dim=1)
        elif self.config.pooler_input == "max":
            pool_input = torch.max(sequence_output, dim=1)[0]
        elif self.config.pooler_input == "mean-max":
            t1 = sequence_output.mean(dim=1)
            t2 = torch.max(sequence_output, dim=1)[0]
            pool_input = t1 + t2
        else:
            raise Exception("Wrong pooler input!")

        if self.config.use_out_pooler:
            pooled_output = self.pooler_activation(pooler(pool_input))
        else:
            pooled_output = self.pooler_activation(self.pooler(pool_input))

        # outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        # 分类器 cls
        # pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        if self.config.use_fc2:
            tmp = self.classifier1(pooled_output)
            tmp = self.pooler_activation(tmp)
            logits = self.classifier2(tmp)
        else:
            logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,)     # + outputs[2:]

        # *****计算相关信息/指标
        # 获取归一化熵
        if self.config.kd_enable:
            tmp = softmax_T(logits, dim=1, T=T)
        else:
            tmp = softmax(logits, dim=1)
        self.softm_logits = tmp
        self.entropy = (tmp * torch.log(tmp)).sum(dim=1) / \
            np.log(1 / self.num_labels)

        # 计算loss并加到输出
        # if labels is not None:
        #     if self.num_labels == 1:
        #         #  We are doing regression
        #         loss_fct = MSELoss()
        #         self.exit_loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         if self.config.kd_enable and T != 1:
        #             loss_fct = Entropy()
        #             self.exit_loss = loss_fct(
        #                 logits.view(-1, self.num_labels), labels.view(-1), T=T)
        #         else:  # 无KD
        #             loss_fct = CrossEntropyLoss()
        #             self.exit_loss = loss_fct(
        #                 logits.view(-1, self.num_labels), labels.view(-1))

        #     outputs = (self.exit_loss,) + outputs

        # 计算acc等
        # preds = None
        # out_label_ids = None
        # if preds is None:
        #     preds = logits.detach().cpu().numpy()
        #     out_label_ids = labels.detach().cpu().numpy()
        # else:
        #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        #     out_label_ids = np.append(
        #         out_label_ids, labels.detach().cpu().numpy(), axis=0
        #     )
        # preds = np.argmax(preds, axis=1)
        # try:
        #     result = compute_metrics(self.config.finetuning_task, preds, out_label_ids)[
        #         "acc"
        #     ]  # 返回acc字典
        # except:
        #     result = compute_metrics(self.config.finetuning_task, preds, out_label_ids)[
        #         "mcc"
        #     ]  # 返回mcc字典
        # if result == 1:
        #     self.is_right = True
        # else:
        #     self.is_right = False
        # (loss), logits, (hidden_states), (attentions)

        return outputs  # (logits,)

    def label_smoothing(self, x):
        e = self.config.e_label_smoothing
        return ((1 - e) * x) + (e / x.shape[-1])

    def lower_than_thres(self, x=None):
        if self.config.thres_name == "entropy":
            # 归一熵
            # return self.entropy < self.exit_thres
            if self.entropy.mean() < self.exit_thres:
                self.exit_cnt_dict["entropy"] = self.exit_cnt_dict["entropy"] + 1
                return True
            else:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False
        elif self.config.thres_name == "logits":
            # 两边的logit
            return self.softm_logits.min() < self.exit_thres
        elif self.config.thres_name == "bias_1":
            if self.entropy.mean() < self.exit_thres:
                self.exit_cnt_dict["entropy"] = self.exit_cnt_dict["entropy"] + 1
                return True
            # elif self.entropy.mean() > self.config.exit_thres_margin + self.config.exit_thres:
            #     return False
            cnt_bias = int(self.config.cnt_thres)
            margin = self.config.margin
            if len(x) < cnt_bias:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False
            cnt_large = 0
            cnt_less = 0
            # input(x[-cnt_bias:])
            for i in range(cnt_bias, 0, -1):
                if x[-i] < 0.5 - margin:
                    cnt_less += 1
                elif x[-i] > 0.5 + margin:
                    cnt_large += 1
            if cnt_large == cnt_bias or cnt_less == cnt_bias:
                self.exit_cnt_dict["bias_1"] = self.exit_cnt_dict.get(
                    "bias_1", 0) + 1
                return True
            else:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False
        elif self.config.thres_name == "bias_2":
            if self.entropy.mean() < self.exit_thres:
                self.exit_cnt_dict["entropy"] = self.exit_cnt_dict["entropy"] + 1
                return True
            cnt_bias = int(self.config.cnt_thres)
            margin = self.config.margin
            if len(x) < cnt_bias:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False
            if x[-cnt_bias] < 0.5 - margin:
                is_larger = False
            elif x[-cnt_bias] > 0.5 + margin:
                is_larger = True
            else:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False

            logit = x[-cnt_bias]
            for i in range(cnt_bias - 1, 0, -1):
                if is_larger and x[-i] < logit:
                    self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                    return False
                if not is_larger and x[-i] > logit:
                    self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                    return False
                logit = x[-i]
            self.exit_cnt_dict["bias_2"] = self.exit_cnt_dict.get(
                "bias_2", 0) + 1
            return True
        elif self.config.thres_name == "bias_3":
            if self.entropy.mean() < self.exit_thres:
                self.exit_cnt_dict["entropy"] = self.exit_cnt_dict["entropy"] + 1
                return True
            cnt_bias = int(self.config.cnt_thres)
            margin = self.config.margin
            if len(x) < cnt_bias:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False
            max_tmp = max(x[-cnt_bias:])
            min_tmp = min(x[-cnt_bias:])

            if min_tmp > 0.5 + margin or max_tmp < 0.5 - margin:
                pass
            else:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False
            if (max_tmp - min_tmp) <= self.config.range_tmp:
                self.exit_cnt_dict["bias_3"] = self.exit_cnt_dict.get(
                    "bias_3", 0) + 1
                return True
            else:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False


class AlbertTransformer(nn.Module):

    def __init__(self, config, exit_config):
        super().__init__()

        self.config = config
        self.exit_config = exit_config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(
            config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList(
            [AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

        # 共享退出层 or not
        # if config.shared:
        #     self.exit_out_layer = ExitLayer(config, exit_config)
        # else:
        #     self.exit_out_layers = nn.ModuleList([ExitLayer(
        #         config, exit_config) for _ in range(config.num_hidden_layers)])

        # 不同的共享程度
        if config.num_exit_layers == 1:
            self.exit_out_layer = ExitLayer(config, exit_config)
        else:
            self.exit_out_layers = nn.ModuleList([ExitLayer(
                config, exit_config) for _ in range(config.num_exit_layers)])

        self.exit_logits_lst = []
        self.cnt_sp = 0

    def forward(self, hidden_states, attention_mask=None, head_mask=None, is_eval=False, pooler=None):
        # 每次inference清零
        self.exit_logits_lst = []
        exit_sm_logits_lst = []

        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        cnt = 0
        while cnt < self.config.num_hidden_layers:
            # Number of layers in a hidden group
            layers_per_group = int(
                self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(
                cnt / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx
                          * layers_per_group: (group_idx + 1) * layers_per_group],
            )
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # choose the exit_layer
            # if self.config.shared:
            #     exit_out_layer = self.exit_out_layer
            # else:
            #     exit_out_layer = self.exit_out_layers[cnt]

            if self.config.num_exit_layers == 1:
                exit_out_layer = self.exit_out_layer
            else:
                layer_idx = cnt//(self.config.num_hidden_layers //
                                  self.config.num_exit_layers)
                exit_out_layer = self.exit_out_layers[layer_idx]

            # 尚未归一化的logits
            exit_logits = exit_out_layer((hidden_states,), pooler=pooler)
            self.exit_logits_lst.append(exit_logits)
            cnt += 1
            if is_eval:  # 只在inference时提前退出
                # print("eval")
                sm_logits = softmax(exit_logits[0], dim=1)
                # input(sm_logits.shape)
                exit_sm_logits_lst.append(sm_logits[0][0])
                if exit_out_layer.lower_than_thres(exit_sm_logits_lst):
                    # print("sp", self.cnt_sp, "exit at:", cnt-1)
                    # input()
                    break

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        # 加入最后一层（eval时可能中途exit）的idx & exit_logits
        # 特殊处理为退出layer的 idx
        outputs = outputs + (cnt - 1, exit_logits)

        self.cnt_sp += 1
        # last-layer hidden state,exit_idx,exit_out
        return outputs


class AlbertExitModel(AlbertPreTrainedModel):

    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config, exit_config):
        super().__init__(config)

        self.config = config
        self.exit_config = exit_config
        self.embeddings = AlbertEmbeddings(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.encoder = AlbertTransformer(config, exit_config)
        # self.pooler_activation = nn.Tanh()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx
                                  * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(
                heads)

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        is_eval=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask,
                                       head_mask=head_mask, is_eval=is_eval, pooler=self.pooler)

        # 更改逻辑
        sequence_output = encoder_outputs[0]    # hidden_state
        exit_idx_output = encoder_outputs[-2]
        logits_output = encoder_outputs[-1]
        outputs = (sequence_output, exit_idx_output,
                   logits_output) + encoder_outputs[1:-2]
        # (最后encoder输出的隐藏态，最后的分类输出exit_logits)+可选的中间量

        # check the attention's size
        # print(type(outputs[-1]), len(outputs[-1]))
        # print(outputs[-1][0].shape)
        # input()

        # attn_score 累积版本得分
        # if self.config.output_attentions:
        #     attn_score = outputs[-1]   # tuple of each layer's attention
        #     seq_len = attn_score[0].shape[-1]
        #     num_heads = attn_score[0].shape[-3]
        #     init_score = torch.eye(seq_len).repeat(
        #         num_heads, 1, 1).unsqueeze(0).cuda()  # each token's w is one-hot
        #     new_attn = (init_score,)    # [1,head,len,len]
        #     for i, attention in enumerate(attn_score):  # each layer
        #         # attenton:[bs,head,len,len]
        #         res_tmp = torch.zeros_like(attention)
        #         for j, head in enumerate(attention[0]):  # each head matrix
        #             # head:[len,len]
        #             last_arr = new_attn[i][0][j]    # last res
        #             res_tmp[0][j] = torch.matmul(head, last_arr)

        #         new_attn = new_attn + (res_tmp,)
        #     outputs = outputs[:-1] + (new_attn[1:],)    # update the attention

        return outputs


class DataWriter(object):
    """
    记录过程中的统计数据,包括
    - 总体视角，记录各层的数据
    - 样本视角，记录不同种样本之间的差异
    """

    def __init__(self, config, exit_config):
        super().__init__()
        self.config = config
        self.exit_config = exit_config
        self.model_name = config.model_name
        self.data_num = config.num_exit + 1  # 包含原退出层在最后

        # 总体结果
        self.info = {}

        # 总体视角，关注各层
        self.final_dict = {
            "loss_lst": [[] for _ in range(self.data_num)],  # 层数*样本数
            "entropy_lst": [[] for _ in range(self.data_num)],
            "logits_lst": [[] for _ in range(self.data_num)],
            # 和上式形状一致，样本通过各层时预测正确与否
            "is_right_lst": [[] for _ in range(self.data_num)],
            # 后期计算
            "cnt_correct": [0] * self.data_num,
            "cnt_total": 0,  # 可由上式len出
            "acc_lst": [0] * self.data_num,
        }
        self.exit_dict = {
            # loss等4项最后从final里面按sample_idx索引取子集
            "loss_lst": [[] for _ in range(self.data_num)],
            "entropy_lst": [[] for _ in range(self.data_num)],
            "logits_lst": [[] for _ in range(self.data_num)],
            # 和上式形状一致，各层exit的样本预测正确与否
            "is_right_lst": [[] for _ in range(self.data_num)],
            # 各层exit的样本序号
            "exit_sample_idx_lst": [[] for _ in range(self.data_num)],
            # 后期计算
            "cnt_correct": [0] * self.data_num,
            "cnt_exit": [0] * self.data_num,  # 可由上式len出
            "acc_lst": [0] * self.data_num,
        }
        self.final_dict_of_arr = {}
        self.exit_dict_of_arr = {
            "loss_lst": [0] * self.data_num,
            "entropy_lst": [0] * self.data_num,
            "logits_lst": [0] * self.data_num,
            # 和上式形状一致，各层exit的样本预测正确与否
            "is_right_lst": [0] * self.data_num,
            # 各层exit的样本序号
            "exit_sample_idx_lst": [0] * self.data_num,
            # 后期计算
            "cnt_correct": [0] * self.data_num,
            "cnt_exit": [0] * self.data_num,  # 可由上式len出
            "acc_lst": [0] * self.data_num,
        }
        # 样本视角
        # self.

    def update(self, d):
        """
        每次inference根据样本计算结果，更新数据字典，默认是batch_size=1
        """
        exit_idx = d['exit_idx']
        for k in ['loss_lst', 'entropy_lst', 'logits_lst', 'is_right_lst']:
            # print(k)
            for i in range(self.data_num):
                self.final_dict[k][i].append(d[k][i])
                # print(d[k][i])
                # input(self.final_dict[k])
                if i == exit_idx or (i == self.data_num - 1 and exit_idx == -1):
                    self.exit_dict[k][i].append(d[k][exit_idx])
                else:
                    self.exit_dict[k][i].append(np.nan)
        self.exit_dict['exit_sample_idx_lst'][exit_idx].append(
            self.final_dict['cnt_total'])
        self.exit_dict['cnt_exit'][exit_idx] += 1
        self.final_dict['cnt_total'] += 1    # 样本计数

        # for k, v in self.exit_dict.items():
        #     print(k, v)
        # input('look')

    def after_run(self):
        "运行完后对数据进行后处理，方便保存与查看"
        for k in ['loss_lst', 'entropy_lst', 'logits_lst', 'is_right_lst']:
            self.final_dict_of_arr[k] = np.array(self.final_dict[k])
            self.exit_dict_of_arr[k] = np.array(self.exit_dict[k])

        # final all
        self.final_dict_of_arr['acc_lst'] = self.final_dict_of_arr['is_right_lst'].mean(
            axis=1)
        self.final_dict_of_arr['cnt_correct'] = np.sum(
            self.final_dict_of_arr['is_right_lst'] == 1, axis=1)

        # exit
        self.exit_dict_of_arr['acc_lst'] = np.nanmean(
            self.exit_dict_of_arr['is_right_lst'], axis=1)
        self.exit_dict_of_arr['cnt_correct'] = np.nansum(
            self.exit_dict_of_arr['is_right_lst'] == 1, axis=1)
        self.exit_dict_of_arr['cnt_exit'] = np.array(
            self.exit_dict['cnt_exit'])
        print("cnt_exit", self.exit_dict_of_arr['cnt_exit'])

        # avg compute layers
        tmp = self.config.idx_lst.copy()
        tmp.append(self.config.num_hidden_layers - 1)   # append对应原来模型中的最后一层
        layer_lst = np.array(tmp)
        layer_lst = layer_lst + 1
        self.info["avg_layer"] = np.sum(
            layer_lst * self.exit_dict_of_arr['cnt_exit']) / self.final_dict['cnt_total']
        self.info["compute_ratio"] = (
            self.info["avg_layer"] / self.config.num_hidden_layers
        )
        self.info["exit_sample_ratio"] = 1 - \
            self.exit_dict_of_arr['cnt_exit'][-1] / \
            self.final_dict['cnt_total']

    def print_info(self, is_eval=False):
        print("*****data print BEGIN*****")
        self.after_run()
        print("all_info:")
        for i in range(self.data_num):
            print("layer {} {:.4f} {}".format(i + 1, self.final_dict_of_arr['acc_lst'][i],
                                              self.final_dict['cnt_total']))
        print("exit_info:")
        for i in range(self.data_num):
            print("layer {} {:.4f} {}".format(i + 1, self.exit_dict_of_arr['acc_lst'][i],
                                              self.exit_dict_of_arr['cnt_exit'][i]))

        # loss_all 统计信息
        arr = self.final_dict_of_arr['loss_lst'].T
        column = ["loss_" + str(i) for i in self.config.idx_lst]
        column.append("loss_ori")
        x = pd.DataFrame(arr, columns=column)
        # x.describe().to_csv()
        print(x.describe())

        # loss_exit
        column = ["loss_exit" + str(i) for i in self.config.idx_lst]
        column.append("loss_ori")
        arr = self.exit_dict_of_arr['loss_lst'].T
        x = pd.DataFrame(arr, columns=column)
        # x.describe().to_csv()
        print(x.describe())
        print("\nGeneral Info:")
        for k, v in self.info.items():
            print(k, ": {:.4f}".format(v))
        print("*****data print END*****")

    def to_file(self, is_eval=False):
        # 输出目录设置
        print("*****to_file BEGIN*****")
        # my_out_dir = "/home/xiekeli/examples/stat_out/"
        my_out_dir = self.config.my_out_dir
        my_out_dir = os.path.join(my_out_dir, self.model_name, self.task)
        if not os.path.exists(my_out_dir):
            os.makedirs(my_out_dir)

        if is_eval:
            # 保存loss entropy logits及相关信息
            for mode in ['_all_', '_exit_']:
                for k in ['loss_lst', 'entropy_lst', 'logits_lst', 'is_right_lst']:
                    if mode == '_all_':
                        data = self.final_dict_of_arr
                    elif mode == '_exit_':
                        data = self.exit_dict_of_arr
                    # 所有item数据
                    tmp = np.array(data[k]).T
                    name = '_'.join(k.split('_')[:-1])
                    np.save(
                        os.path.join(my_out_dir, name + mode
                                     + "eval.npy"), tmp
                    )
                    # item统计信息
                    column = [name + "_" + str(i) for i in self.config.idx_lst]
                    column.append(name + "_ori")
                    x_all = pd.DataFrame(tmp, columns=column)
                    x_all.describe().to_csv(
                        os.path.join(my_out_dir, name + "_info" + mode +
                                     "eval.csv"))

            # cnt_exit
            column = ["cnt_exit_" + str(i) for i in self.config.idx_lst]
            column.append("cnt_ori")
            cnt_exit = self.exit_dict_of_arr['cnt_exit']
            cnt_exit = cnt_exit.reshape(1, -1)
            x = pd.DataFrame(cnt_exit, columns=column)
            print("cnt_exit", self.exit_dict['cnt_exit'])
            x.to_csv(os.path.join(
                my_out_dir, "cnt_info_" + "_exit_eval.csv"))

            # 保存self.info为json
            info_file = os.path.join(
                my_out_dir, "info_" + "_exit_eval.json")
            with open(info_file, "w") as f:
                f.write(json.dumps(self.info, ensure_ascii=False,
                                   indent=4, separators=(",", ":")))

        else:
            pass
            # x.to_json(os.path.join(
            #     my_out_dir, "cnt_info_" + self.task + "_tr.json"))
            # x.to_csv(os.path.join(
            #     my_out_dir, "cnt_info_" + self.task + "_tr.csv"))
            # info_file = os.path.join(
            #     my_out_dir, "info_" + self.task + "_tr.json")

        print("*****to_file END*****")

    def set_task(self, task):
        self.task = task

    def reset(self):
        self.info = {}
        self.final_dict = {
            "loss_lst": [[] for _ in range(self.data_num)],  # 层数*样本数
            "entropy_lst": [[] for _ in range(self.data_num)],
            "logits_lst": [[] for _ in range(self.data_num)],
            # 和上式形状一致，样本通过各层时预测正确与否
            "is_right_lst": [[] for _ in range(self.data_num)],
            "cnt_correct": [0] * self.data_num,
            "cnt_total": 0,  # 可由上式len出
            "acc_lst": [0] * self.data_num,
        }
        self.exit_dict = {
            # loss等4项最后从final里面按索引取子集
            "loss_lst": [[] for _ in range(self.data_num)],
            "entropy_lst": [[] for _ in range(self.data_num)],
            "logits_lst": [[] for _ in range(self.data_num)],
            # 和上式形状一致，各层exit的样本预测正确与否
            "is_right_lst": [[] for _ in range(self.data_num)],
            # 各层exit的样本序号
            "exit_sample_idx_lst": [[] for _ in range(self.data_num)],
            # 后期计算
            "cnt_correct": [0] * self.data_num,
            "cnt_exit": [0] * self.data_num,  # 可由上式len出
            "acc_lst": [0] * self.data_num,
        }
        self.final_dict_of_arr = {}
        self.exit_dict_of_arr = {
            "loss_lst": [0] * self.data_num,
            "entropy_lst": [0] * self.data_num,
            "logits_lst": [0] * self.data_num,
            # 和上式形状一致，各层exit的样本预测正确与否
            "is_right_lst": [0] * self.data_num,
            # 各层exit的样本序号
            "exit_sample_idx_lst": [0] * self.data_num,
            # 后期计算
            "cnt_correct": [0] * self.data_num,
            "cnt_exit": [0] * self.data_num,  # 可由上式len出
            "acc_lst": [0] * self.data_num,
        }


class AlbertExitForCls(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.exit_config = ExitConfig(config.exit_config_path)
        self.num_labels = config.num_labels
        self.data_num = config.num_exit + 1

        self.albert = AlbertExitModel(config, self.exit_config)

        self.data_writer = DataWriter(config, self.exit_config)

        # loss 加权方式
        self.exit_weight_lst = None
        self.ori_weight = None
        if self.config.weight_name == "dyn":
            self.sigmoid = nn.Sigmoid()
            self.W = torch.tensor([config.w_init for _ in range(config.num_exit)],
                                  device="cuda", requires_grad=True)
            self.M = config.num_exit + 1
            self.w_lr = config.w_lr
        elif self.config.weight_name == "equal":
            self.ori_weight = 1.
            self.exit_weight_lst = [1.] * self.config.num_exit

        self.init_weights()

        self.cnt_exit = [0] * self.data_num
        self.compute_ratio = 0

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        is_eval=False,
    ):
        # 最后一层（eval时可能是提前退出的层）的输出
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            is_eval=is_eval,
        )   # (最后encoder输出的hidden_state，logits_output)+可选的中间量

        # train
        if not is_eval:
            # 获取各exit_out loss的权重
            if self.config.weight_name == "dyn":
                self.exit_weight_lst = self.sigmoid(self.W)
                self.ori_weight = self.M - self.exit_weight_lst.sum()
            # print("exit: ", self.exit_weight_lst)
            # print("ori: ", self.ori_weight)
            # 获取各个exit_logits
            exit_logits_lst = self.albert.encoder.exit_logits_lst
            # 计算loss
            loss_weighted = None
            loss_all = []
            for i, logits in enumerate(exit_logits_lst):
                logits = logits[0]  # tuple to tensor
                if labels is not None:
                    if self.num_labels == 1:
                        #  We are doing regression
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), labels.view(-1))
                    else:
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(
                            logits.view(-1, self.num_labels), labels.view(-1))
                loss_all.append(loss.item())     # save loss
                if loss_weighted is None:
                    loss_weighted = loss * self.exit_weight_lst[i]
                else:
                    try:
                        loss_weighted += loss * self.exit_weight_lst[i]
                    except:  # 最后层的权重
                        loss_weighted += loss * self.ori_weight

            outputs = (loss_weighted, logits,) + outputs[3:]
            return outputs  # (loss), logits, (hidden_states), (attentions)

        # inference
        else:
            exit_idx = outputs[1]
            self.cnt_exit[exit_idx] += 1
            logits = outputs[2][0]
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss, logits) + outputs
            return outputs

    def after_run(self):
        # avg compute layers
        tmp = self.config.idx_lst.copy()
        tmp.append(self.config.num_hidden_layers - 1)   # append对应原来模型中的最后一层
        layer_lst = np.array(tmp)
        layer_lst = layer_lst + 1
        cnt_arr = np.array(self.cnt_exit)
        avg_layer = np.sum(layer_lst * cnt_arr / cnt_arr.sum())
        self.compute_ratio = (avg_layer / self.config.num_hidden_layers)
        print("Computation cost: {:.4f}".format(self.compute_ratio))


class AlbertMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores


class AlbertForMaskedLM(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(
            self.predictions.decoder, self.albert.embeddings.word_embeddings)

    def get_output_embeddings(self):
        return self.predictions.decoder

    # @ add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        # Add hidden states and attention if they are here
        outputs = (prediction_scores,) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs


# @ add_start_docstrings(
#     """Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
#     the hidden-states output to compute `span start logits` and `span end logits`). """,
#     ALBERT_START_DOCSTRING,
# )
class AlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @ add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        # (loss), start_logits, end_logits, (hidden_states), (attentions)
        return outputs


class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AlbertModel(AlbertPreTrainedModel):

    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
            If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
            is a total of 4 different layers.

            These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
            while [2,3] correspond to the two inner groups of the second hidden layer.

            Any layer with in index other than [0,1,2,3] will result in an error.
            See base class PreTrainedModel for more information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx
                                  * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(
                heads)

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Example::

        from transformers import AlbertModel, AlbertTokenizer
        import torch

        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler_activation(
            self.pooler(sequence_output[:, 0]))
        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs


class AlbertForTokenClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="albert-base-v2",
    #     output_type=TokenClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_tuple=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_tuple = return_tuple if return_tuple is not None else self.config.use_return_tuple

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_tuple=return_tuple,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if return_tuple:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
