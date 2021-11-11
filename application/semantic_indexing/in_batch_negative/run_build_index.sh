corpus_num=580000
# --params_path "checkpoints/simcse_inbatch_negative/model_45000/model_state.pdparams" \
python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "checkpoints/train_0.001/model_10/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus_${corpus_num}.csv"
        # --corpus_file "data/corpus.csv" 