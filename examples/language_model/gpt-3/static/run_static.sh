set -x
export PADDLE_WITH_GLOO=0
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*

task_name="gpt-mp-sharding"
rm -rf output/$task_name/log

python -u  -m paddle.distributed.fleet.launch \
    --gpus "4,5,6,7" \
    --log_dir "output/$task_name/log" run_pretrain_static.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 1024 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --sharding_degree 2 \
    --mp_degree 2 \
    --dp_degree 1 \
    --pp_degree 1 \
    --use_sharding true \
    --use_amp true \
    --amp_level "O1" \
    --use_recompute true \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 5000 \
    --save_steps 1000 \
    --decay_steps 320000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 1\
    --eval_freq 1000 \
    --device "gpu"

 # Not support pipeline for this version, don't change pp_degree.
