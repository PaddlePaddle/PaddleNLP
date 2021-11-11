# root_path=train_0.01
root_path=train_0.001
# python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
#     train_batch_neg.py \
#     --device gpu \
#     --save_dir ./checkpoints/${root_path} \
#     --batch_size 64 \
#     --learning_rate 5E-5 \
#     --epochs 3 \
#     --output_emb_size 256 \
#     --save_steps 1 \
#     --max_seq_length 64 \
#     --margin 0.2 \
#     --train_set_file data/${root_path}/train.csv 

# python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
#     train_batch_neg.py \
#     --device gpu \
#     --save_dir ./checkpoints/simcse_inbatch_negative \
#     --batch_size 64 \
#     --learning_rate 5E-5 \
#     --epochs 3 \
#     --output_emb_size 256 \
#     --save_steps 10 \
#     --max_seq_length 64 \
#     --margin 0.2 \
#     --train_set_file data/${root_path}/train.csv  \
#     --init_from_ckpt simcse/model_20000/model_state.pdparams


python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    train_batch_neg.py \
    --device gpu \
    --save_dir ./checkpoints/post_simcse_inbatch_negative \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 3 \
    --output_emb_size 256 \
    --save_steps 10 \
    --max_seq_length 64 \
    --margin 0.2 \
    --train_set_file data/${root_path}/train.csv  \
    --init_from_ckpt simcse/post_model_10000/model_state.pdparams
