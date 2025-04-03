#!/bin/bash

IP=127.0.0.1
PORT=9965

# 定义类别列表
categories=('business' 'law' 'psychology' 'biology' 'chemistry' 'history' 'other' 'health' 'economics' 'math' 'physics' 'philosophy' 'engineering')

# 遍历每个类别并执行 Python 脚本
for category in "${categories[@]}"; do
    echo "Evaluating category: $category" > ./txt_log/mmlu_${category}.log
    nohup python3 evaluate_from_api.py --backend paddle --ip "$IP" --port "$PORT" --output_dir ./r1_mla_eval --assigned_subjects "$category" >> ./txt_log/mmlu_${category}.log 2>&1 &
done


categories=('computer science')

# 遍历每个类别并执行 Python 脚本
for category in "${categories[@]}"; do
    echo "Evaluating category: $category" > ./txt_log/mmlu_cs.log
    nohup python3 evaluate_from_api.py --backend paddle --ip "$IP" --port "$PORT" --output_dir ./r1_mla_eval --assigned_subjects "$category" >> ./txt_log/mmlu_cs.log 2>&1 &
done