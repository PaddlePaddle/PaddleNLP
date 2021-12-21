unset CUDA_VISIBLE_DEVICES

task_name="ernie-1.0-dp8-gb1024"
rm -rf output/$task_name/log

PYTHONPATH=../../../  python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    --log_dir "output/$task_name/log" \
    run_pretrain_static.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 512 \
    --micro_batch_size 32 \
    --global_batch_size 128 \
    --sharding_degree 1\
    --dp_degree 4 \
    --use_sharding false \
    --use_amp true \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 200000 \
    --save_steps 100000 \
    --checkpoint_steps 5000 \
    --decay_steps 1980000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --num_workers 2 \
    --logging_freq 20\
    --eval_freq 1000 \
    --device "gpu"

# NOTE: please set use_sharding=True for sharding_degree > 1
