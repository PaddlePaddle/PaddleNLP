export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=1.0

PYTHONPATH=../../../ python -u run_pretrain.py \
    --model_type "gpt2"\
    --model_name_or_path "gpt2-small-en"\
    --input_dir "./data"\
    --output_dir "output"\
    --max_lr 0.00015\
    --min_lr 0.00001\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 70000\
    --save_steps 70000\
    --eval_steps 500\
    --logging_steps 1\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --micro_bsz 16\
    --device "gpu"


