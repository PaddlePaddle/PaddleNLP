# GPU training
root_path=inbatch
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    train_batch_neg.py \
    --device gpu \
    --save_dir ./checkpoints/${root_path} \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 3 \
    --output_emb_size 256 \
    --save_steps 10 \
    --max_seq_length 64 \
    --margin 0.2 \
    --train_set_file recall/train.csv


# cpu training
# root_path=inbatch
# python train_batch_neg.py \
#     --device cpu \
#     --save_dir ./checkpoints/${root_path} \
#     --batch_size 64 \
#     --learning_rate 5E-5 \
#     --epochs 3 \
#     --output_emb_size 256 \
#     --save_steps 10 \
#     --max_seq_length 64 \
#     --margin 0.2 \
#     --train_set_file recall/train.csv 



# 加载simcse训练的模型，模型放在simcse/model_20000
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
#     --evaluate \
#     --train_set_file recall/train.csv  \
#     --init_from_ckpt simcse/model_20000/model_state.pdparams

# 加载post training的模型，模型放在simcse/post_model_10000
# python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
#     train_batch_neg.py \
#     --device gpu \
#     --save_dir ./checkpoints/post_simcse_inbatch_negative \
#     --batch_size 64 \
#     --learning_rate 5E-5 \
#     --epochs 3 \
#     --output_emb_size 256 \
#     --save_steps 10 \
#     --max_seq_length 64 \
#     --margin 0.2 \
#     --train_set_file recall/train.csv  \
#     --init_from_ckpt simcse/post_model_10000/model_state.pdparams
