export FLAGS_prim_all=true
export FLAGS_enable_pir_api=True
export FLAGS_cudnn_deterministc=1
export ENABLE_FALL_BACK=False 
export CUDA_VISIBLE_DEVICES=6

if [ ! -d ./data ]
then
    mkdir ./data
    cd ./data
    wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
    wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
    cd ..
fi

if [ -d ./output ]
then
    rm -rf ./output
fi

task_name_or_path="llama_output"
GLOG_vmodule=generated_vjp=4 python run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "__internal_testing__/tiny-random-llama" \
    --tokenizer_name_or_path "__internal_testing__/tiny-random-llama" \
    --input_dir "./data" \
    --output_dir "./output/$task_name_or_path" \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --lr_scheduler_type "cosine" \
    --max_steps 5000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --dataloader_num_workers 1 \
    --eval_steps 5000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 0\
    --recompute 0 \
    --do_train \
    --device "gpu" \
    --seed 2023 \
    --use_fused_rms_norm False

