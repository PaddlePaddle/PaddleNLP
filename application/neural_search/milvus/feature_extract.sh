root_dir="checkpoints" 
python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        feature_extract.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_40/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 4096 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "recall/test.csv" \
        --corpus_file "milvus/milvus_data.csv" 