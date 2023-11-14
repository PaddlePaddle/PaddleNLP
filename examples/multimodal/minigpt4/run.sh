export PYTHONPATH=/root/paddlejob/workspace/env_run/zhengshifeng/vitllm/pr/PaddleNLP/:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=3

python run_predict.py --pretrained_name_or_path=/root/paddlejob/workspace/env_run/zhengshifeng/vitllm/pr/PaddleNLP/examples/multimodal/minigpt4/vit_model
