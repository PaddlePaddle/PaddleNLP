set -x
temp_dir=./tmp
temp_dir_full=./tmp_full
graph_dir=./graph

mkdir -p ${temp_dir}
mkdir -p ${temp_dir_full}
export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1
export PYTHONPATH=../:$PYTHONPATH

#for task in "piqa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy"; do
#for task in "data_slim" "data"; do
#for task in "boolq"; do
#for task in "data"; do
#for task in  "ARC-Challenge"; do
#for task in  "ARC-Challenge" "ARC-Easy" "data_slim" "data" "boolq" "winogrande" "piqa" "hellaswag"; do
#for task in "boolq" "winogrande" "piqa" "hellaswag"; do
for task in "winogrande"; do
    #export BREAK=1
    #export QUANT=2
    #export REMOVE_MW=1
    #for i in $(seq 10); do
    #    echo "${task}: ${i}"
    #    pids=""
    #    python -u  -m paddle.distributed.launch  \
    #            --log_dir log1  \
    #            --gpus "0,1,2,3,4,5,6,7" \
    #            run_finetune.py \
    #            config/llama/${task}.json \
    #            --output_dir ../../../checkpoint/${task} \
    #            --logging_dir ./${graph_dir}/${task}/${i} \
    #            --dataset_name_or_path ${task}  \
    #            --save_total_limit 1 \
    #            &>> ${temp_dir}/workerlog.${task} & 
    #    #if echo "$task" | grep -q "winogrande"; then
    #    #    python -u  -m paddle.distributed.launch  \
    #    #            --gpus "0,1,2,3,4,5,6,7" \
    #    #            run_finetune.py \
    #    #            config/llama/sft_argument.json \
    #    #            --output_dir ../../../checkpoint/${task} \
    #    #            --logging_dir ./graph/${task}/${i} \
    #    #            --dataset_name_or_path ${task}  \
    #    #            --save_total_limit 1 \
    #    #            --zero_padding true \
    #    #            --per_device_train_batch_size 1    \
    #    #            --gradient_accumulation_steps 2  \
    #    #            --per_device_eval_batch_size 1    \
    #    #            --eval_accumulation_steps 2  \
    #    #            &>> ${temp_dir}/workerlog.${task} & 
    #    #else
    #    #    python -u  -m paddle.distributed.launch  \
    #    #            --gpus "0,1,2,3,4,5,6,7" \
    #    #            run_finetune.py \
    #    #            config/llama/sft_argument.json \
    #    #            --output_dir ../../../checkpoint/${task} \
    #    #            --logging_dir ./graph/${task}/${i} \
    #    #            --dataset_name_or_path ${task}  \
    #    #            --save_total_limit 1 \
    #    #            --zero_padding false \
    #    #            --per_device_train_batch_size 8    \
    #    #            --gradient_accumulation_steps 2   \
    #    #            --per_device_eval_batch_size 4    \
    #    #            --eval_accumulation_steps 16    \
    #    #            &>> ${temp_dir}/workerlog.${task} & 
    #    #fi
    #    
    #    pids="$pids $!"
    #    wait $pids
    #done

    export BREAK=0
    export QUANT=2
    export REMOVE_MW=0
    pids=""
    python -u  -m paddle.distributed.launch  \
            --log_dir log1  \
            --gpus "0,1,2,3,4,5,6,7" \
            run_finetune.py \
            config/llama/${task}_1.json \
            --output_dir ../../../checkpoint/${task} \
            --logging_dir ./${graph_dir}_tmp/${task}/full \
            --adam_beta1 0.5    \
            --adam_beta2 0.555    \
            --dataset_name_or_path ${task}  \
            --save_total_limit 1 \
            &>> ${temp_dir_full}/workerlog.${task} & 
            #--output_dir ../../../checkpoint/${task}_full \
    #if echo "$task" | grep -q "winogrande"; then
    #    python -u  -m paddle.distributed.launch  \
    #            --gpus "0,1,2,3,4,5,6,7" \
    #            run_finetune.py \
    #            config/llama/sft_argument.json \
    #            --output_dir ../../../checkpoint/${task}_full \
    #            --logging_dir ./graph/${task}/full \
    #            --dataset_name_or_path ${task}  \
    #            --zero_padding true \
    #            --save_total_limit 1 \
    #            --per_device_train_batch_size 1    \
    #            --gradient_accumulation_steps 2  \
    #            --per_device_eval_batch_size 1    \
    #            --eval_accumulation_steps 2     \
    #            &>> ${temp_dir_full}/workerlog.${task} & 
    #else
    #    python -u  -m paddle.distributed.launch  \
    #            --gpus "0,1,2,3,4,5,6,7" \
    #            run_finetune.py \
    #            config/llama/sft_argument.json \
    #            --output_dir ../../../checkpoint/${task}_full \
    #            --logging_dir ./graph/${task}/full \
    #            --dataset_name_or_path ${task}  \
    #            --save_total_limit 1 \
    #            --zero_padding false \
    #            --per_device_train_batch_size 8    \
    #            --gradient_accumulation_steps 2   \
    #            --per_device_eval_batch_size 4    \
    #            --eval_accumulation_steps 16    \
    #            &>> ${temp_dir_full}/workerlog.${task} & 
    #fi

    pids="$pids $!"
    wait $pids

    #rm -rf ../../../checkpoint/${task}/checkpoint-*
    #rm -rf ../../../checkpoint/${task}_full/checkpoint-*
done

#for task in  "ARC-Challenge" "ARC-Easy" "data_slim" "data" "boolq" "winogrande" "piqa" "hellaswag"; do
#    if [ "$task" != "data" ] && [ "$task" != "data_slim" ]; then
#        python evaluate.py --base_model ../../../checkpoint/${task}/ --dataset ./${task}/dev.json --model quant_${task} &>> ${temp_dir}/workerlog.${task} 
#        python evaluate.py --base_model ../../../checkpoint/${task}_full/ --dataset ./${task}/dev.json --model ${task} &>> ${temp_dir_full}/workerlog.${task} 
#    fi
#done
