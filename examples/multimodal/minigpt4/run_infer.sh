export PYTHONPATH=../../../../PaddleNLP/:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=3

python infer.py --model_dir=./vit_model/