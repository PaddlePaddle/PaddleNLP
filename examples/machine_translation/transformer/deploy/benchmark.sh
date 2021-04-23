#!/bin/bash
model_dir=$1
log_path="output_pipeline"

for batch_size in "1" "2" "4"; do
    python python/inference.py \
        --config="../configs/transformer.base.yaml" \
        --model-dir=${model_dir} \
        --batch-size=${batch_size} \
        --profile > ${log_path}/transformer_base_cpu_nomkl_bs${batch_size}_inference.log 2>&1

    for threads in "1" "6"; do
        python python/inference.py \
            --config="../configs/transformer.base.yaml" \
            --model-dir=${model_dir} \
            --use-mkl \
            --threads=${threads} \
            --batch-size=${batch_size} \
            --profile > ${log_path}/transformer_base_cpu_mkl_threads${threads}_bs${batch_size}_inference.log 2>&1 
    done

    python python/inference.py \
        --config="../configs/transformer.base.yaml" \
        --model-dir=${model_dir} \
        --use-gpu \
        --batch-size=${batch_size} \
        --profile > tee ${log_path}/transformer_base_gpu_bs${batch_size}_inference.log 2>&1 
done
