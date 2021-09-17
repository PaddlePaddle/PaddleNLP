set -x

task_name="gpt-acc-single-dygraph"
base_out="tests/output"
rm -rf $base_out/$task_name/log

unset CUDA_VISIBLE_DEVICES
PYTHONPATH=../../../ python -m paddle.distributed.launch \
    --gpus "0" \
    --log_dir "$base_out/$task_name/log"  run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path "./ckpt/gpt2-small-en-init-checkpoint"\
    --input_dir "./data"\
    --output_dir "$base_out/$task_name"\
    --max_seq_len 1024 \
    --micro_batch_size 8\
    --max_lr 0.00015\
    --min_lr 0.00001\
    --max_steps 20\
    --save_steps 100000\
    --decay_steps 320000\
    --weight_decay 0.01\
    --warmup_rate 0.01\
    --grad_clip 1.0\
    --logging_freq 1\
    --eval_freq 500\
    --check_accuracy  true\
    --device "gpu"
