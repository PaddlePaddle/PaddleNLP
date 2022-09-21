# GPU training
root_path=inbatch
data_path=data
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    train.py \
    --device gpu \
    --save_dir ./checkpoints/${root_path} \
    --batch_size 24 \
    --learning_rate 5E-5 \
    --epochs 100 \
    --output_emb_size 0 \
    --save_steps 50 \
    --max_seq_length 384 \
    --warmup_proportion 0.0 \
    --margin 0.2 \
    --recall_result_dir "recall_result_dir" \
    --recall_result_file "recall_result.txt" \
    --train_set_file ${data_path}/train.txt \
    --corpus_file ${data_path}/label.txt   \
    --similar_text_pair ${data_path}/dev.txt \
    --evaluate True
