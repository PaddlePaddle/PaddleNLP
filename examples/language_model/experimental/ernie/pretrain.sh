set -x

export PADDLE_WITH_GLOO=0
export GLOG_v=0
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit
unset CUDA_VISIBLE_DEVICES

rm -rf *.prototxt
rm -rf core.*

task_name='ernie-base-2pp2dp2mp'
output_dir=output/${task_name}
rm -rf ${output_dir}

PYTHONPATH=../../../ python -m paddle.distributed.fleet.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    --log_dir ${output_dir}/log \
    run_pretraining.py \
    --global_bsz 64 \
    --micro_bsz 1 \
    --max_seq_len 512 \
    --ernie_config_file config/ernie_base_config.json \
    --learning_rate 1e-4 \
    --log_steps 1 \
    --num_train_steps 1000000 \
    --save_steps 100000 \
    --output_dir ${output_dir} \
    --use_recompute true \
    --use_sharding true \
    --use_sop false \
    --num_mp=2 \
    --num_sharding=2 \
    --num_pp=2 \
    --num_dp=1 \

