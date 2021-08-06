
unset CUDA_VISIBLE_DEVICES

PYTHON_BIN="/usr/local/bin/python3.7"
export PYTHONPATH="/ssd2/tianxin04/PaddleNLP/"

test_set="data_v2/test/public_test_A"

python -u -m paddle.distributed.launch --gpus "7" \
        predict.py \
        --device gpu \
        --params_path "./checkpoints/model_1600/model_state.pdparams"\
        --batch_size 128 \
        --input_file ${test_set}
