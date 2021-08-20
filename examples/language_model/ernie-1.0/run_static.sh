set -x
export PADDLE_WITH_GLOO=0
export FLAGS_call_stack_level=2
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*

task_name="ernie-1.0-dp16-gb1024"
rm -rf output/$task_name/log

#export CUDA_VISIBLE_DEVICES=1
#export CUDA_LAUNCH_BLOCKING=1
#export PYTHONPATH=../../../ 
# cuda-memcheck  python -u  \


PYTHONPATH=../../../  python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name/log" \
    run_pretrain_static.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 512 \
    --micro_batch_size 64 \
    --global_batch_size 1024 \
    --sharding_degree 1\
    --dp_degree 8 \
    --mp_degree 1 \
    --pp_degree 1 \
    --use_sharding false \
    --use_amp true \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 2000000 \
    --save_steps 100000 \
    --checkpoint_steps 5000 \
    --decay_steps 1980000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 20\
    --eval_freq 1000 \
    --device "gpu"

# Not support pipeline for this version, don't change pp_degree.
#/usr/local/cuda-11.2/bin/nsys  profile -t cuda,nvtx -o ernie_fp32 --capture-range=cudaProfilerApi --stop-on-range-end=true \
