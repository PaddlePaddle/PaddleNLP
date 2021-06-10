set -x

task_name="gpt-dygraph"
rm -rf output/$task_name/log

unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    --log_dir "output/$task_name/log"  run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en"\
    --input_dir "./data"\
    --output_dir "output/$task_name"\
    --max_seq_len 1024 \
    --micro_batch_size 4\
    --max_lr 0.00015\
    --min_lr 0.00001\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --weight_decay 0.01\
    --warmup_rate 0.01\
    --grad_clip 1.0\
    --logging_freq 1\
    --eval_freq 1000\
    --device "gpu"
