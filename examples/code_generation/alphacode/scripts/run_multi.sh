set -x

task_name="bart-dygraph"
rm -rf output/$task_name/log

export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus "0,1" \
    --log_dir "output/$task_name/log"  run_pretrain.py \
    --model_type "bart" \
    --model_name_or_path "alphacode-small"\
    --input_dir "./data_tools"\
    --output_dir "output/$task_name"\
    --encoder_max_seq_len 1536 \
    --decoder_max_seq_len 768 \
    --micro_batch_size 3 \
    --num_train_epochs 10 \
    --max_lr 0.00015\
    --min_lr 0.00001\
    --save_steps 10000 \
    --weight_decay 0.01\
    --warmup_rate 0.01\
    --grad_clip 1.0\
    --logging_freq 1\
    --eval_freq 10000 \
    --device "gpu" \
    --use_amp true
