#!/bin/bash
model_dir=$1
log_path="output_pipeline"

for batch_size in "1" "2" "4"; do
    for use_mkl in "True" "False"; do
        for threads in "1" "6"; do
                python3.6 python/inference.py \
                    --config="../configs/transformer.base.yaml" \
                    --model_dir=${model_dir} \
                    --use_mkl \
                    --threads=${threads} \
                    --batch_size=${batch_size} \
                    2>&1 | tee ${log_path}/transformer_base_cpu_usemkl_${use_mkl}_threads_${threads}_bs${batch_size}_inference.log
        done
    done

    python3.6 python/inference.py \
        --config="../configs/transformer.base.yaml" \
        --model_dir=${model_dir} \
        --use_gpu \
        --batch_size=${batch_size} \
        2>&1 | tee ${log_path}/transformer_base_gpu_bs${batch_size}_inference.log
done
