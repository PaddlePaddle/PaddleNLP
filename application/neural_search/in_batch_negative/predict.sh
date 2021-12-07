root_dir="checkpoints/train_0.001" 

python -u -m paddle.distributed.launch --gpus "3" \
    predict.py \
    --device gpu \
    --params_path "${root_dir}/model_40/model_state.pdparams" \
    --output_emb_size 256 \
    --batch_size 128 \
    --max_seq_length 64 \
    --text_pair_file "data/test.csv"
