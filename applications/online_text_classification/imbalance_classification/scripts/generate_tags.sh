# python generate_result_milvus.py \
#                         --recall_path ./data/answer.txt \
#                         --feature_path ./data/val_embedding.npy \
#                         --corpus_path ./data/train_data.txt \
#                         --tag_path ./data/answer_tag.txt \
#                         --threshold 0.3

python generate_tags.py \
                        --recall_path ./recall_result_dir/recall_result.txt \
                        --tag_path ./data/answer_tag.txt \
                        --threshold 0.3