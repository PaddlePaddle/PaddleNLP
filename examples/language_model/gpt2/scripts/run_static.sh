export PADDLE_WITH_GLOO=0
export GLOG_v=0
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*
rm -rf start_sharding*
rm -rf main_sharding*


task_name="gpt2-mp-sharing"
rm -rf output/$task_name/log

#fleetrun --gpus 0,1,2,3 run_pretrain_static.py --model_name_or_path gpt2-medium-en --input_dir "./data"\
PYTHONPATH=../../../ python -u  -m paddle.distributed.fleet.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    --log_dir output/$task_name/log run_pretrain_static.py \
    --model_type "gpt2"\
    --model_name_or_path "gpt2-small-en" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_lr 0.00015\
    --min_lr 0.00001\
    --weight_decay 0.01\
    --max_steps 70000\
    --save_steps 70000\
    --decay_steps 320000\
    --eval_steps 500\
    --grad_clip 1.0\
    --use_recompute false \
    --use_amp false \
    --use_sharding true \
    --warmup_rate 0.01 \
    --batch_size 8 \
    --sharding_degree 2\
    --mp_degree 2\
    --pp_degree 2\
    --dp_degree 1\
    --device gpu
