export CUDA_VISIBLE_DEVICES=1

python3.7 -m paddle.distributed.launch --gpus="1" ./run_xfun_ser.py \
    --model_name_or_path "./layoutxlm-base-paddle/" \
    --max_seq_length 512 \
    --train_data_dir "zh.train/" \
    --train_label_path "zh.train/xfun_normalize_train.json" \
    --eval_data_dir "zh.val/" \
    --eval_label_path "zh.val/xfun_normalize_val.json" \
    --num_train_epochs 200 \
    --eval_steps 10 \
    --save_steps 500 \
    --output_dir "output_v9/" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --evaluate_during_training
