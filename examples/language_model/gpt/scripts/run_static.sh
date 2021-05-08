set -x
export PADDLE_WITH_GLOO=0
export GLOG_v=0
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.98
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*
fuser -kv /dev/nvidia*

task_name="gpt-mp-sharding"
rm -rf output/$task_name/log

PYTHONPATH=../../../ python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name/log" run_pretrain_static.py \
    --model_type "gpt" \
    --model_name_or_path "gpt-small-en" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --weight_decay 0.01\
    --max_steps 70000 \
    --save_steps 50 \
    --decay_steps 320000 \
    --eval_steps 50 \
    --grad_clip 1.0 \
    --use_recompute true \
    --use_amp true \
    --use_sharding true \
    --warmup_rate 0.01 \
    --micro_bsz 32 \
    --global_bsz 128 \
    --sharding_degree 4\
    --mp_degree 2 \
    --pp_degree 1 \
    --dp_degree 1 \
    --device "gpu"

 # Not support pipeline for this version, don't change pp_degree.
