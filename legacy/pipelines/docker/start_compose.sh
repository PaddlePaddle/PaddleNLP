#!/bin/bash
cd /root/PaddleNLP/pipelines/
# linux
python utils/offline_ann.py --index_name dureader_robust_query_encoder \
                            --doc_dir data/dureader_dev \
                            --query_embedding_model rocketqa-zh-nano-query-encoder \
                            --passage_embedding_model rocketqa-zh-nano-para-encoder \
                            --port 9200 \
                            --host localhost \
                            --embedding_dim 312 \
                            --delete_index 
# 使用端口号 8891 启动模型服务
nohup python rest_api/application.py 8891 > server.log 2>&1 &
# 在指定端口 8502 启动 WebUI
nohup python -m streamlit run ui/webapp_semantic_search.py --server.port 8502 > client.log 2>&1 &
