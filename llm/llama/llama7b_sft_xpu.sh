export PYTHONPATH=/paddle/PaddleNLP/
export XPU_PADDLE_FC_INT32_WITH_LL=1
export XPU_PADDLE_L3_SIZE=40060288

export XPU_LLAMA_FFN=True
export LD_LIBRARY_PATH=/paddle/baidu/xpu/fast_paddle/build/kernel_build/so:$LD_LIBRARY_PATH
#export XPU_DEBUG=0X20
#export XPUAPI_DEBUG=0x21
#export XPU_PADDLE_DEBUG=1
#export GLOG_v=10

export BKCL_PCIE_RING=1
export BKCL_CCIX_RING=1
export BKCL_SOCKET_IFNAME=docker0
export BKCL_RING_BUFFER_SIZE=33554432
export BKCL_RING_BUFFER_SIZE=8388608
export BKCL_RING_BUFFER_SIZE=4194304
export FLAGS_fuse_parameter_memory_size=128
export FLAGS_fuse_parameter_groups_size=128
export BKCL_SOCKET_FORCE_TREE=1

unset PADDLE_MASTER
unset PADDLE_NNODES
unset PADDLE_JOB_ID

# 根据需要修改如下的多机ip地址
    #--master  10.93.200.26:47789  --ips="10.93.200.75,10.93.200.26" --nnodes 2 \
task_name="llama_sft"
python -u  -m paddle.distributed.launch \
    --devices "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    finetune_generation.py \
    --device "xpu" \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 16 \
    --model_name_or_path facebook/llama-7b \
    --task_name squad \
    --max_steps 2000 \
    --learning_rate 3e-5 \
    --warmup_steps 2 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 512 \
    --tgt_length 512 \
    --do_train \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --tensor_parallel_degree 8 \
    --pipeline_parallel_degree 1 \
    --do_eval \
    --disable_tqdm True
