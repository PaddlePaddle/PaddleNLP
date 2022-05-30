#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

if [ $# != 4 ];then
    echo "USAGE: sh run_train.sh \$TRAIN_SET \$MODEL_PATH \$epoch \$nodes_count"
    exit 1
fi

TRAIN_SET=$1
MODEL_PATH=$2
epoch=$3
node=$4

CHECKPOINT_PATH=output
if [ ! -d output ]; then
    mkdir output
fi
if [ ! -d log ]; then
    mkdir log
fi

lr=1e-5
batch_size=32
train_exampls=`cat $TRAIN_SET | wc -l`
save_steps=$[$train_exampls/$batch_size/$node]
data_size=$[$save_steps*$batch_size*$node]
new_save_steps=$[$save_steps*$epoch/2]

python3 -m paddle.distributed.launch \
    --log_dir log \
    ./src/train_ce.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val false \
                   --do_test false \
                   --use_mix_precision false \
                   --train_data_size ${data_size} \
                   --batch_size ${batch_size} \
                   --init_pretraining_params ${MODEL_PATH} \
                   --train_set ${TRAIN_SET} \
                   --save_steps ${new_save_steps} \
                   --validation_steps ${new_save_steps} \
                   --checkpoints ${CHECKPOINT_PATH} \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --epoch $epoch \
                   --max_seq_len 384 \
                   --for_cn true \
                   --vocab_path config/ernie_base_1.0_CN/vocab.txt \
                   --ernie_config_path config/ernie_base_1.0_CN/ernie_config.json \
                   --learning_rate ${lr} \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1 

