set -x
export CUDA_VISIBLE_DEVICES=0

python -u run_pretrain.py \
    --model_type "bart"\
    --model_name_or_path "bart-base"\
    --input_dir "./data_tools"\
    --output_dir "output"\
    --micro_batch_size 4 \
    --encoder_max_seq_len 1536 \
    --decoder_max_seq_len 768 \
    --max_lr 0.00015\
    --min_lr 0.00001\
    --max_steps 50000 \
    --save_steps 100000\
    --decay_steps 320000\
    --weight_decay 0.01\
    --warmup_rate 0.01\
    --grad_clip 1.0\
    --logging_freq 1\
    --eval_freq 1000 \
    --device "gpu" \

#    --use_amp true
