python -u ./ernie_ctm_finetune.py \
    --max_seq_len 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 100 \
    --output_dir ./tmp/$TASK_NAME/ \
    --n_gpu 1 \