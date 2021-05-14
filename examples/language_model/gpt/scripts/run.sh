export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=1.0

PYTHONPATH=../../../ python -u run_pretrain.py \
    --model_type "gpt"\
    --model_name_or_path "gpt2-en"\
    --input_dir "./data"\
    --output_dir "output"\
    --max_seq_len 1024 \
    --micro_batch_size 4\
    --max_lr 0.00015\
    --min_lr 0.00001\
    --max_steps 70000\
    --save_steps 70000\
    --decay_steps 320000\
    --weight_decay 0.01\
    --warmup_rate 0.01\
    --grad_clip 1.0\
    --logging_freq 1\
    --eval_freq 500\
    --device "gpu"


