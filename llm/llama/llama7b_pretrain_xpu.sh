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

# down loaddata
#mkdir data
#cd data
#wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
#wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
#cd ..

# 根据需要修改如下的多机ip地址
    #--master  10.93.200.26:47789  --ips="10.93.200.75,10.93.200.26" --nnodes 2 \
task_name="llama_pretrain"
python -u  -m paddle.distributed.launch \
    --devices "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-7b" \
    --tokenizer_name_or_path "facebook/llama-7b" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 64 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 4 \
    --virtual_pp_degree 1 \
    --sequence_parallel 0 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 1\
    --recompute 1 \
    --do_train \
    --do_eval \
    --device "xpu"
    # --pipeline_parallel_config "disable_partial_send_recv"  # if set sequence_parallel True, please note off this line.
    #--fp16  \
    #--fp16_opt_level "O2"  \
    #--scale_loss 512 \
