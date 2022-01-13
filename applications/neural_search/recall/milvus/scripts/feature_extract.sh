root_dir="checkpoints" 
python -u -m paddle.distributed.launch --gpus "4" --log_dir "recall_log/" \
        feature_extract.py \
        --device gpu \
        --params_path "${root_dir}/model_40/model_state.pdparams" \
        --batch_size 4096 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --corpus_file "milvus/milvus_data.csv" 