
# root_dir="checkpoints/train_0.01" 
root_dir="checkpoints/train_0.001" 
# root_dir="checkpoints/train_0.0001" 
corpus_num=280000
python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_10/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus_${corpus_num}.csv" 

python -u evaluate.py \
        --similar_text_pair "data/test.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50


python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_20/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus_${corpus_num}.csv"

python -u evaluate.py \
        --similar_text_pair "data/test.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_30/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus_${corpus_num}.csv"  

python -u evaluate.py \
        --similar_text_pair "data/test.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_40/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus_${corpus_num}.csv"  

python -u evaluate.py \
        --similar_text_pair "data/test.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_5/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus_${corpus_num}.csv"  

python -u evaluate.py \
        --similar_text_pair "data/test.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_6/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus_${corpus_num}.csv"  

python -u evaluate.py \
        --similar_text_pair "data/test.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50

# python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
#         recall.py \
#         --device gpu \
#         --recall_result_dir "recall_result_dir" \
#         --recall_result_file "recall_result.txt" \
#         --params_path "checkpoints/model_45000/model_state.pdparams" \
#         --hnsw_m 100 \
#         --hnsw_ef 100 \
#         --batch_size 64 \
#         --output_emb_size 256\
#         --max_seq_length 60 \
#         --recall_num 50 \
#         --similar_text_pair "data/test.csv" \
#         --corpus_file "data/corpus.csv" 

# python -u evaluate.py \
#         --similar_text_pair "data/test.csv" \
#         --recall_result_file "./recall_result_dir/recall_result.txt" \
#         --recall_num 50