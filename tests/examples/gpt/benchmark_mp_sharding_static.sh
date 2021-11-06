set -x
export PADDLE_WITH_GLOO=0
export FLAGS_call_stack_level=2
#export FLAGS_allocator_strategy=naive_best_fit
export GLOG_v=-1
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*

task_name="gpt-benchmark-mp-sharding-static"
rm -rf output/$task_name/log

PYTHONPATH=../../../ python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3" \
    --log_dir "output/$task_name/log" run_pretrain_static.py \
    --model_type "gpt" \
    --model_name_or_path "./ckpt/gpt2-small-en-init-checkpoint"\
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 1024 \
    --micro_batch_size 16 \
    --global_batch_size 32 \
    --sharding_degree 2\
    --mp_degree 2 \
    --dp_degree 1 \
    --pp_degree 1 \
    --use_sharding true \
    --use_amp true \
    --use_recompute true \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 70000 \
    --save_steps 10000 \
    --decay_steps 320000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 1\
    --eval_freq 500 \
    --device "gpu"

