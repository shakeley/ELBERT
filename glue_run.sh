export CUDA_VISIBLE_DEVICES=$2
export PRE_DIR=/home/xxx/
export MODEL=albert_large

# required
export TEST_MODE="one"
export TASK_NAME=RTE
export GLUE_DIR=${PRE_DIR}BERT/params/GLUE
export W_DIR=${PRE_DIR}BERT/params/W/${MODEL}
export OUT_DIR=${PRE_DIR}BERT/myoutput/${MODEL}

echo "Task:$TASK_NAME"
if [ $1 = "1" ];then
echo "Train!"
python ./run_glue.py \
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


fi

# 
echo "Eval!"
# loop
if [ $TEST_MODE = "all" ];then

# test confidence
export KEY=exit_thres
for i in $(seq 9|tac)
do 
export V=0.$i

# test window
# export KEY=cnt_thres
# for i in $(seq 2 12)
# do
# export V=$i

python ./change_config.py \
  --key $KEY \
  --value $V \
  --task $TASK_NAME \
  --model $MODEL \
  --out_dir ${OUT_DIR} \

python ./run_glue.py \
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

done

# one-shot test
elif [ $TEST_MODE = "one" ];then
export KEY=exit_thres
export V=0.4
python ./change_config.py \
  --key $KEY \
  --value $V \
  --task $TASK_NAME \
  --model $MODEL \
  --out_dir ${OUT_DIR} \

export KEY=cnt_thres
export V=8
python  ./change_config.py \
  --key $KEY \
  --value $V \
  --task $TASK_NAME \
  --model $MODEL \
  --out_dir ${OUT_DIR} \


python ./run_glue.py \
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


fi
