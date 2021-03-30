unset CUDA_VISIBLE_DEVICES
#fleetrun --gpus 0,1,2,3 run_pretrain_static.py --model_name_or_path gpt2-medium-en --input_dir "./input_data"\
PYTHONPATH=../../../ python -u  -m paddle.distributed.fleet.launch --gpus 3 --log_dir ./log run_pretrain_static.py \
    --model_type gpt2\
    --model_name_or_path gpt2-medium-en \
    --input_dir "./new_data" \
    --output_dir "output" \
    --max_lr 0.00015\
    --min_lr 0.00001\
    --weight_decay 0.01\
    --max_steps 1000\
    --save_steps 100\
    --eval_steps 100\
    --use_recompute False\
    --use_amp True\
    --warmup_rate .1\
    --batch_size 6\
    --select_devices gpu
