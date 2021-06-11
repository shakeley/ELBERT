# ELBERT
ELBERT is a fast BERT-like model coupled with a confidence-window based early exit mechanism, which achieves high-speed inference based on ALBERT without introducing additional parameters. It's quite easy for users to migrate from ALBERT to ELBERT for fast inference. 
<img src="https://github.com/shakeley/pics/blob/master/Transmodel.svg" alt="6cebc512b4fe9d336d256d6f4720b1f" style="zoom:67%;" />
## Environment

Our code is implemented in```python=3.6.5, pytorch=1.4.0 (GPU Version)```. To run it, please install ```transformers``` by ```pip```

```bash
$ pip install transformers==2.7.0
```

## Setup

1. Download the pre-trained models of [ALBERT](https://github.com/google-research/ALBERT#albert). 

2. Convert the parameters using ```convert_albert_original_tf_checkpoint_to_pytorch.py``` and change the files' names as 

    ```
    config.json
    pytorch_model.bin
    spiece.model
    spiece.vocab
    ```

    > You may use the ```config.json```in the repo and do some changes according to your demand. 

3. Modify the parameters in ```config.json```, whose meanings are explained as follows

    ```
    my_out_dir: Dir of some statistical outputs. 
    exit_config_path: Needed for future use. You may touch a blank file named "exit_config.json" and set the file dir. 
    weight_name: [dyn,equal]. Dyn corresponds to dynamic weight approach applied in ELBERT. Equal corresponds to the common method which is setting the same weight for every exit layer. 
    thres_name: [bias_1, bias_2, bias_3], which correspond to the different ways mentioned in the second stage of early exit in ELBERT. 
    cnt_thres: The window size N. 
    ```

4. Set the directories in ```glue_run.sh```

    ```
    MODEL: The name of the model. 
    GLUE_DIR: Dir of GLUE datasets. 
    W_DIR: Dir of the pre-trained model parameters. 
    OUT_DIR: Dir of the outputs. 
    ```

## Datasets

The GLUE datasets can be downloaded by running ```download_glue_data.py```. 

## Example

Run training on RTE dataset

```bash
$ python ./run_glue.py \
  --per_gpu_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --save_steps 5000 \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size 1 \
  --overwrite_output_dir \
  --model_type albert \
  --task_name $TASK_NAME \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --model_name_or_path $W_DIR \
  --output_dir $OUT_DIR/$TASK_NAME \
  --config_name $W_DIR \
  --do_train \
```

Set the parameters for early exit (eg., set the confidence threshold in inference)

```
export KEY=exit_thres
export V=0.1
python  ./change_config.py \
  --key $KEY \
  --value $V \
  --task $TASK_NAME \
  --model $MODEL \
  --out_dir ${OUT_DIR} \
```

Run inference on RTE dataset

```bash
$ python ./run_glue.py \
  --out_mode $TEST_MODE \
  --per_gpu_train_batch_size 32 \
  --learning_rate 1e-5 \
  --save_steps 5000 \
  --max_seq_length 128 \
  --num_train_epochs 3.0 \
  --per_gpu_eval_batch_size 1 \
  --overwrite_output_dir \
  --model_type albert \
  --task_name $TASK_NAME \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --model_name_or_path $W_DIR \
  --output_dir $OUT_DIR/$TASK_NAME \
```

Then execute the ```glue_run.sh``` by

```bash
$ bash ./glue_run.sh 1 2
```

The evaluation results are

```
Computation cost: 0.5881
***** Eval results  *****
acc = 0.8122743682310469
```






