unset CUDA_VISIBLE_DEVICES
# gpu
python -u -m paddle.distributed.launch --gpus "0" evaluate.py \
        --device gpu \
        --batch_size 32 \
        --learning_rate 2E-5 \
        --init_from_ckpt "./checkpoints/model_30000/model_state.pdparams" \
        --test_file sort/dev_pairwise.csv

# cpu
# python  evaluate.py \
#         --device cpu \
#         --batch_size 32 \
#         --learning_rate 2E-5 \
#         --init_from_ckpt "./checkpoints/model_30000/model_state.pdparams" \
#         --test_file sort/dev_pairwise.csv