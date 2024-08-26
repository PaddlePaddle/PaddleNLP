# !/bin/bash

python predict/predictor.py \
    --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --data_file ./data/humaneval.json  \
    --dtype bfloat16 \
    --output_file qwen2-1.5b.json