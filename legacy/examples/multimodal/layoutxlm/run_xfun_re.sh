export CUDA_VISIBLE_DEVICES=0

python ./run_xfun_re.py \
    --model_name_or_path "layoutxlm-base-uncased" \
    --max_seq_length 512 \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --num_train_epochs 200 \
    --eval_steps 50 \
    --save_steps 500 \
    --output_dir "./output/re/" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --evaluate_during_training \
    --seed 2048
