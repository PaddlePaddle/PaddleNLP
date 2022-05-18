#set -x
export PADDLE_WITH_GLOO=0
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit


# add PaddleNLP/paddlenlp to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:../../../../../

MP_DEGREE=${1:-1}
PP_DEGREE=${2:-1}
DP_DEGREE=${3:-1}
MICRO_BATCH_SIZE=${4:-8}
GLOBAL_BATCH_SIZE=${5:-8}

max_steps=30
use_sharding=true
total_degree=$(( $PP_DEGREE * $DP_DEGREE * $MP_DEGREE ))
case $total_degree in
    1) gpus="0"
       max_steps=3
       use_sharding=false
    ;;
    2) gpus="0,1"
       if [ $DP_DEGREE -eq 2 ]
       then
          use_sharding=false
       fi
       max_steps=3
    ;;
    4) gpus="0,1,2,3"
       max_steps=3
    ;;
    8) gpus="0,1,2,3,4,5,6,7"
       if [ $PP_DEGREE -eq 1 ]
       then
          use_sharding=false
       fi
       max_steps=3
    ;;
    32) gpus="0,1,2,3,4,5,6,7"
       if [ $PP_DEGREE -eq 1 ]
       then
          use_sharding=false
       fi
    ;;
    *) echo "Support total_degree is 1, 2, 4, 8 and 32, but you give $total_degree"
    ;;
esac

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*

python3.7 -u  -m paddle.distributed.launch \
    --gpus $gpus \
    --log_dir ./log \
    ../run_pretrain_static.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --input_dir "./data" \
    --output_dir "output/" \
    --max_seq_len 1024 \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --sharding_degree 1\
    --mp_degree $MP_DEGREE \
    --dp_degree $DP_DEGREE \
    --pp_degree $PP_DEGREE \
    --use_sharding $use_sharding \
    --use_amp true \
    --amp_level "O1" \
    --use_recompute true \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps $max_steps \
    --save_steps 5000 \
    --decay_steps 320000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 1\
    --eval_freq 1000 \
    --device "gpu"
