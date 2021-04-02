unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1" run_pretrain.py \
    --model_type gpt2\
    --model_name_or_path gpt2-small-en\
    --input_dir "./data"\
    --output_dir "output"\
    --max_lr 0.00015\
    --min_lr 0.00001\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --batch_size 8\
    --device gpu
