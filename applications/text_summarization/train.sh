# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
unset CUDA_VISIBLE_DEVICES

log_dir=output
rm -rf ${log_dir}
mkdir -p ${log_dir}

python -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir ${log_dir} train.py \
    --model_name_or_path=unimo-text-1.0-summary \
    --train_file train.json \
    --eval_file test.json \
    --save_dir=${log_dir}/checkpoints \
    --logging_steps=100 \
    --save_steps=10000 \
    --epochs=10 \
    --batch_size=32 \
    --learning_rate=5e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=60 \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --do_train \
    --do_eval \
    --device=gpu \
