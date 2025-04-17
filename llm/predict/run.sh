#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

export PYTHONPATH=/root/paddlejob/workspace/env_run/output/changwenbin/dongyazhu/PaddleNLP:$PYTHONPATH

# python ./predictor.py --model_name_or_path Qwen/Qwen2.5-14B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --batch_size 2

seqlen=1024
nsys profile -o seqlen-1.5-linear-${seqlen} python ./predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --batch_size 2 --src_length ${seqlen} --min_length 100 --total_max_length 8400

seqlen=4096
nsys profile -o seqlen-1.5-linear-${seqlen} python ./predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --batch_size 2 --src_length ${seqlen} --min_length 100 --total_max_length 8400

# seqlen=8192
# nsys profile -o seqlen${seqlen} python ./predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --batch_size 2 --src_length ${seqlen} --min_length 100 --total_max_length 16384

# seqlen=4096
# python ./predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --batch_size 2

# python ./predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --batch_size 2

seqlen=1024
nsys profile -o seqlen-40-linear-${seqlen} python ./predictor.py --model_name_or_path Qwen/Qwen2.5-14B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --batch_size 2 --src_length ${seqlen} --min_length 100 --total_max_length 8400

seqlen=4096
nsys profile -o seqlen-40-linear-${seqlen} python ./predictor.py --model_name_or_path Qwen/Qwen2.5-14B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --batch_size 2 --src_length ${seqlen} --min_length 100 --total_max_length 8400