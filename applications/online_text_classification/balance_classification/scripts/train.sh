# 一级分类，类别均衡
python -m paddle.distributed.launch --gpus "1" train.py \
                                    --device gpu \
                                    --task_name level1 \
                                    --save_dir ./checkpoints \
                                    --valid_steps 50 \
                                    --epochs 6 \
                                    --train_file ./data/train_data.txt \
                                    --test_file ./data/val_data.txt  \
                                    --label_file ./data/label_level1.txt \
                                    --label_index -2 \
                                    --logging_steps 10 \
                                    --use_amp False

# 二级分类，类别不均衡
# python -m paddle.distributed.launch --gpus "1" train.py \
#                                     --device gpu \
#                                     --task_name level2 \
#                                     --save_dir ./checkpoints \
#                                     --valid_steps 50 \
#                                     --epochs 6 \
#                                     --train_file ./related_data/train_sub_set_base.txt \
#                                     --test_file ./related_data/val_data.txt  \
#                                     --label_file ./related_data/label_level2.txt \
#                                     --label_index -1 \
#                                     --logging_steps 10 \
#                                     --use_amp False