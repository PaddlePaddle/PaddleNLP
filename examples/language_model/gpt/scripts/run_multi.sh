set -x

task_name="gpt-dygraph"
rm -rf output/$task_name/log

unset CUDA_VISIBLE_DEVICES
PYTHONPATH=../../../ python -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    --log_dir "output/$task_name/log"  run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path "gpt-small-en"\
    --input_dir "./data"\
    --output_dir "output/$task_name"\
    --max_lr 0.00015\
    --min_lr 0.00001\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --eval_steps 1000\
    --logging_steps 10\
    --save_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --micro_bsz 8\
    --device "gpu"
