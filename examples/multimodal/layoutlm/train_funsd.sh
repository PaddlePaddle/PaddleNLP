export CUDA_VISIBLE_DEVICES=7

python3.7 train_funsd.py \
    --data_dir "./data/" \
    --model_name_or_path "layoutlm-base-uncased" \
    --do_lower_case \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --num_train_epochs 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --output_dir "output/" \
    --labels "./data/labels.txt" \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --evaluate_during_training
