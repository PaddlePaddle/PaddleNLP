

# set paddle env
export PYTHONPATH=/paddle/workspace/Paddle/build/python:/paddle/new_env/deepseek/PaddleNLP



python -u  -m paddle.distributed.launch --gpus "2" run_pretrain.py ./config/deepseek-v2/pretrain_argument.json --continue_training False
