unset http_proxy && unset https_proxy
export CUDA_VISIBLE_DEVICES=0
# linux
python utils/offline_ann.py --index_name dureader_robust_query_encoder \
                            --doc_dir data/dureader_dev \
                            --query_embedding_model rocketqa-zh-nano-query-encoder \
                            --passage_embedding_model rocketqa-zh-nano-para-encoder \
                            --port 9200 \
                            --host localhost \
                            --embedding_dim 312 \
                            --delete_index 
# windows & macos                             
# python utils/offline_ann.py --index_name dureader_robust_query_encoder \
#                             --doc_dir data/dureader_dev \
#                             --query_embedding_model rocketqa-zh-nano-query-encoder \
#                             --passage_embedding_model rocketqa-zh-nano-para-encoder \
#                             --port 9200 \
#                             --host host.docker.internal \
#                             --embedding_dim 312 \
#                             --delete_index 
