# gpu
text_file='./data/val_data.txt'
corpus_file='./data/train_data.txt'
python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "checkpoints/model_50/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 128 \
        --text_tag_fle "${text_file}" \
        --corpus_file "${corpus_file}"  
