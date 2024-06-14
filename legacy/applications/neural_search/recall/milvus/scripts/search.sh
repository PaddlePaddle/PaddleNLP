python milvus_ann_search.py --data_path milvus/milvus_data.csv \
                            --embedding_path corpus_embedding.npy \
                            --batch_size 100000 \
                            --index 18 \
                            --insert \
                            --search