# gpu
python -u -m paddle.distributed.launch --gpus "0" \
        predict_pairwise.py \
        --device gpu \
        --params_path "./checkpoints/model_10000/model_state.pdparams"\
        --batch_size 128 \
        --max_seq_length 64 \
        --input_file 'sort/test_pairwise.csv'
# cpu
# python predict_pairwise.py \
#         --device gpu \
#         --params_path "./checkpoints/model_10000/model_state.pdparams"\
#         --batch_size 128 \
#         --max_seq_length 64 \
#         --input_file 'sort/test_pairwise.csv'