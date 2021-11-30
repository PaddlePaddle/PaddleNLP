# root_dir=dropout_0.100000_batch_size_64 
# root_dir="checkpoints/dropout_0.100000_batch_size_64 " 
root_dir="checkpoints/dropout_0.200000_batch_size_64" 
# root_dir="checkpoints/dropout_0.400000_batch_size_64 " 
# root_dir="checkpoints/dropout_0.300000_batch_size_64 " 
# root_dir="checkpoints/model_base"

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_10000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus.csv" 

python -u evaluate.py \
    --similar_text_pair "data/test.csv" \
    --recall_result_file "./recall_result_dir/recall_result.txt" \
    --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_20000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus.csv" 

python -u evaluate.py \
    --similar_text_pair "data/test.csv" \
    --recall_result_file "./recall_result_dir/recall_result.txt" \
    --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_25000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus.csv" 

python -u evaluate.py \
    --similar_text_pair "data/test.csv" \
    --recall_result_file "./recall_result_dir/recall_result.txt" \
    --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_32000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus.csv" 

python -u evaluate.py \
    --similar_text_pair "data/test.csv" \
    --recall_result_file "./recall_result_dir/recall_result.txt" \
    --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_45000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus.csv" 

python -u evaluate.py \
    --similar_text_pair "data/test.csv" \
    --recall_result_file "./recall_result_dir/recall_result.txt" \
    --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_63000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus.csv" 

python -u evaluate.py \
    --similar_text_pair "data/test.csv" \
    --recall_result_file "./recall_result_dir/recall_result.txt" \
    --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_93000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus.csv" 

python -u evaluate.py \
    --similar_text_pair "data/test.csv" \
    --recall_result_file "./recall_result_dir/recall_result.txt" \
    --recall_num 50