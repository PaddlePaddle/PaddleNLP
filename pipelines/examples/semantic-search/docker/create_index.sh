unset http_proxy && unset https_proxy
export CUDA_VISIBLE_DEVICES=0
# linux
python utils/offline_ann.py --index_name dureader_robust_query_encoder \
                            --doc_dir data/dureader_dev \
                            --port 9200 \
                            --host localhost
# windows                             
# python utils/offline_ann.py --index_name dureader_robust_query_encoder \
#                             --doc_dir data/dureader_dev \
#                             --port 9200 \
#                             --host host.docker.internal
