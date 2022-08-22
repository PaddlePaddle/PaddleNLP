# gpu
python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_pairwise.py \
        --device gpu \
        --save_dir ./checkpoints \
        --batch_size 32 \
        --learning_rate 2E-5 \
        --margin 0.1 \
        --eval_step 100 \
        --train_file sort/train_pairwise.csv \
        --test_file sort/dev_pairwise.csv

# cpu
# python train_pairwise.py \
#         --device cpu \
#         --save_dir ./checkpoints \
#         --batch_size 32 \
#         --learning_rate 2E-5 \
#         --margin 0.1 \
#         --eval_step 100 \
#         --train_file sort/train_pairwise.csv \
#         --test_file sort/dev_pairwise.csv