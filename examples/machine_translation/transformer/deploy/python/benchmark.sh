#!/bin/bash
model_dir=${1}
model=${2}
mkdir -p output_pipeline
log_path="output_pipeline"

for batch_size in "1" "2" "4"; do
    python inference.py \
        --config="../../configs/transformer.${model}.yaml" \
        --device cpu \
        --model_dir=${model_dir} \
        --batch_size=${batch_size} \
        --profile > ${log_path}/transformer_${model}_cpu_nomkl_bs${batch_size}_inference.log 2>&1

    for threads in "1" "6"; do
        python inference.py \
            --config="../../configs/transformer.${model}.yaml" \
            --model_dir=${model_dir} \
            --device cpu \
            --use_mkl True \
            --threads=${threads} \
            --batch_size=${batch_size} \
            --profile > ${log_path}/transformer_${model}_cpu_mkl_threads${threads}_bs${batch_size}_inference.log 2>&1 
    done

    python inference.py \
        --config="../../configs/transformer.${model}.yaml" \
        --model_dir=${model_dir} \
        --device gpu \
        --batch_size=${batch_size} \
        --profile > tee ${log_path}/transformer_${model}_gpu_bs${batch_size}_inference.log 2>&1 
done
