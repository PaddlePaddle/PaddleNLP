set -x
export PADDLE_WITH_GLOO=0
export FLAGS_call_stack_level=2
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*

task_name="ernie-1.0-dp8-gb512"
rm -rf output/$task_name/log


PYTHONPATH=../../../  python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name/log" \
    run_pretrain_static.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 512 \
    --micro_batch_size 64 \
    --global_batch_size 512 \
    --sharding_degree 1\
    --dp_degree 8 \
    --use_sharding false \
    --use_amp true \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 2000000 \
    --save_steps 50000 \
    --checkpoint_steps 5000 \
    --decay_steps 1980000 \
    --weight_decay 0.01 \
    --warmup_rate 0.005 \
    --grad_clip 1.0 \
    --logging_freq 20\
    --num_workers 2 \
    --eval_freq 1000 \
    --device "gpu"\

# --check_accuracy true\

# NOTE: please set use_sharding=True for sharding_degree > 1
