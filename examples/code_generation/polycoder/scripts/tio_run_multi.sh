set -x

task_name="gpt-dygraph3"
dir_name="finetune"
mdl="./pretrain/gpt-dygraph/model_250000"
rm -rf $dir_name/$task_name/log

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    --log_dir "$dir_name/$task_name/log"  tio_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path "$mdl" \
    --input_dir "./$dir_name" \
    --output_dir "./$dir_name/$task_name" \
    --max_seq_len 1024 \
    --micro_batch_size 4 \
    --max_lr 0.00008 \
    --min_lr 0.000005 \
    --max_steps 400000\
    --save_steps 50000 \
    --decay_steps 250000\
    --weight_decay 0.01\
    --warmup_rate 0.01\
    --grad_clip 1.0\
    --logging_freq 300\
    --eval_freq 1000\
    --device "gpu"
