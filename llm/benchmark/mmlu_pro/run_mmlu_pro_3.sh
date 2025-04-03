#!/bin/bash

IP=127.0.0.1
PORT=9965

# 定义类别列表
categories=('physics' 'computer science' 'philosophy' 'engineering')

# 遍历每个类别并执行 Python 脚本
for category in "${categories[@]}"; do
    echo "Evaluating category: $category"
    python3 evaluate_from_api.py --backend paddle --ip "$IP" --port "$PORT" --output_dir ./r1_mla_eval --assigned_subjects "$category"
done