set -x
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*

# dp8 for 8 worker of data parallel
# gb512 for the global batch size is 512 = 64 * 8
task_name="ernie-1.0-dp8-gb512"
rm -rf output/$task_name/log

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name/log" \
    run_pretrain_static.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0-base-zh" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_len 512 \
    --micro_batch_size 64 \
    --sharding_degree 1\
    --dp_degree 8 \
    --use_sharding false \
    --use_amp true \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 1000000 \
    --save_steps 50000 \
    --checkpoint_steps 5000 \
    --decay_steps 990000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --num_workers 2 \
    --logging_freq 20\
    --eval_freq 1000 \
    --device "gpu"

# NOTE: please set use_sharding=True for sharding_degree > 1
