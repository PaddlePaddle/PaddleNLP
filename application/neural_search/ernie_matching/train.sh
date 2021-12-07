# pointwise
# python -u -m paddle.distributed.launch --gpus "0" train_pointwise.py \
#         --device gpu \
#         --save_dir ./checkpoints \
#         --batch_size 32 \
#         --learning_rate 2E-5

python -u -m paddle.distributed.launch --gpus "0,2,3,4" train_pairwise.py \
        --device gpu \
        --save_dir ./checkpoints \
        --batch_size 32 \
        --learning_rate 2E-5 \
        --margin 0.1 \
        --eval_step 100 \
        --train_file data/train_pairwise.csv \
        --test_file data/test_pairwise.csv