export PYTHONPATH=$PYTHONPATH:../../../../
export FLAGS_enable_eager_mode=0

log_dir=dp8
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir $log_dir --gpus "0,1,2,3,4,5,6,7" run_moe_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-small-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 50000\
    --save_steps 100000\
    --decay_steps 320000\
    --device gpu\
    --eval_freq 1000\
    --warmup_rate 0.01\
    --local_batch_size 8\
    --dp_degree 8\
    --mp_degree 1\
    --pp_degree 1\
    --sharding_degree 1\
    --sharding_offload False\
    --expert_mode True\
    --logging_freq 1 \
    --num_experts 8\
    --use_pure_fp16 True\
    --use_recompute True\
    --recompute_partition False\
    --recompute_offload False\
    --resume_dir ""\
    --scale_loss 32768 \
    --gate gshard \
    --balance_loss_weight 1.0

