# gpu version

root_dir="checkpoints/inbatch" 
python -u -m paddle.distributed.launch --gpus "3" \
    predict.py \
    --device gpu \
    --params_path "${root_dir}/model_40/model_state.pdparams" \
    --output_emb_size 256 \
    --batch_size 128 \
    --max_seq_length 64 \
    --text_pair_file "recall/test.csv"


# cpu
# root_dir="checkpoints/inbatch" 
# python predict.py \
#     --device cpu \
#     --params_path "${root_dir}/model_40/model_state.pdparams" \
#     --output_emb_size 256 \
#     --batch_size 128 \
#     --max_seq_length 64 \
#     --text_pair_file "recall/test.csv"
