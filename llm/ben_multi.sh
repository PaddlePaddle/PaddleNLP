set -x
temp_dir=./tmp
temp_dir_full=./tmp_full

mkdir -p ${temp_dir}
mkdir -p ${temp_dir_full}

export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1

#for task in "piqa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy"; do
#for task in "data_slim" "data"; do
#for task in "boolq"; do
for task in "data"; do
    export BREAK=1
    export QUANT=1
    for i in $(seq 17 20); do
        echo "${task}: ${i}"
        pids=""
        python -u  -m paddle.distributed.launch  \
            --gpus "0,1,2,3,4,5,6,7" \
            run_finetune.py \
            config/llama/sft_argument.json \
            --output_dir ./checkpoint/${task} \
            --logging_dir ./graph/${task}/${i} \
            --dataset_name_or_path ${task}  \
            --save_total_limit 1 \
            &>> ${temp_dir}/workerlog.${task} & 
        pids="$pids $!"
        wait $pids
    done
    export BREAK=0
    export QUANT=0
    pids=""
    python -u  -m paddle.distributed.launch  \
        --gpus "0,1,2,3,4,5,6,7" \
        run_finetune.py \
        config/llama/sft_argument.json \
        --output_dir ./checkpoint/${task}_full \
        --logging_dir ./graph/${task}/full \
        --dataset_name_or_path ${task}  \
        --save_total_limit 1 \
        &>> ${temp_dir_full}/workerlog.${task} & 
    pids="$pids $!"
    wait $pids
    rm -rf ./checkpoint/*
done

