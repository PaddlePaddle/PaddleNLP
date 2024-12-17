#!/usr/bin/env bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

export log_path=/workspace/case_logs
export root_path=/workspace/PaddleNLP

export gpt_case_path=$root_path/slm/model_zoo/gpt-3
export gpt_data_path=/fleetx_data

export llama_case_path=$root_path/llm/auto_parallel/llama
export llama_data_path=/llama_data
export llm_gpt_case_path=$root_path/llm/auto_parallel/gpt-3

unset CUDA_VISIBLE_DEVICES

function is_a100() {
    if [ $(nvidia-smi|grep A100|wc -l)  -ne 0 ];then
        echo 1
    else
        echo 0
    fi
}

IS_A100=$(is_a100)

function track_case_status() {  
    local case_name="$1"  
    local prefix="$2"  
    local original_path  
  
    original_path=$(pwd)  
    cd ${log_path} || { echo "Failed to enter log_path: $log_path"; return 1; }  
  
    total_count=$(ls -1 "$prefix"* 2>/dev/null | grep -Ev 'result\.log|functions\.txt' | wc -l)
    run_fail_count=$(ls -1 "$prefix"*_FAIL* 2>/dev/null | wc -l)  
    loss_fail_count=$(grep 'check failed! ' result.log | awk -v prefix="$prefix" '{if ($2 ~ "^" prefix) print $2}'| wc -l)
    
    echo -e "\033[31m ---- $case_name total tests :  $total_count \033"
    if [ $run_fail_count -eq 0 ] && [ $loss_fail_count  -eq 0 ]; then
        echo -e "\033[32m ---- all cases Success  \033"
    else
        if [[ $run_fail_count -ne 0 ]] ; then
            echo -e "\033[31m ---- $case_name runtime failed test  :  $run_fail_count \033"
            ls -1 "$prefix"*_FAIL* 2>/dev/null | awk -v OFS="\t" '{print "\t" $0 "(failed)"}'
        fi
        if [[ $loss_fail_count -ne 0 ]] ; then
            echo -e "\033[31m ---- $case_name verification failed test  :  $loss_fail_count \033"
            grep 'check failed! ' result.log | awk -v prefix="$prefix" 'BEGIN {OFS="\t"} {if ($2 ~ "^" prefix) print "\t" $2 "(failed)"}'
        fi
        return 2
    fi
    cd "$original_path" || { echo "Failed to return to original path: $original_path"; return 1; }  
    return 0
}

function restore_func() {
    fun_list=$1
    cd ${log_path} || { echo "Failed to enter log_path: $log_path"; return 1; } 
    if [ -e "functions.txt" ]; then
        rm "functions.txt"
        echo "Deleted existing functions.txt"
    fi
    for function in ${fun_list[@]};do
        echo "$function" >> functions.txt
    done
}



# NOTE: Please place the new tests as much as possible after the existing tests
function llama_case_list_auto() {
    fun_list=(
        # The test name must have "llama_" as a prefix, which will 
        # be used for tracking the execution status of the case.
        llama_dygraph_auto_bs8_fp32_DP2
        llama_dygraph_auto_bs8_fp32_DP2-MP2
        llama_dygraph_auto_bs8_fp32_DP2-MP2-PP2
        llama_dygraph_auto_bs8_fp16_DP2-MP2-PP2
        llama_dy2st_auto_bs4_bf16_DP1-MP1-PP4-SD2-VPP3_split_bw
        llama_dy2st_auto_bs4_bf16_DP1-MP1-PP4-SD2
        llama_align_dygraph_dy2st_auto_bs2_bf16_DP2-MP1-PP1
        llama_pir_auto_fuse_ffn_attention_qkv_MP2
        llama_convert_hybrid_ckpt_to_auto_parallel_bs2_fp32_DP2-MP1-PP1
        llama_align_dygraph_dy2st_pir_auto_bs2_bf16_DP2-MP2-PP1-SP
        llama_align_dygraph_dy2st_pir_auto_bs2_bf16_DP2-MP2-PP2-SP
        llama_align_dygraph_dy2st_pir_auto_grad_merge_bs2_fp32_DP1-MP1-PP1
        llama_align_dy2st_fthenb_and_vpp_auto_bs2_fp32_DP1-MP1-PP4
        llama_align_dygraph_dy2st_pir_auto_pp_bs2_bf16_DP1-MP1-PP4
        llama_baichuan_pir_auto_fuse_ffn_attention_qkv_DP2_MP2_PP2
        llama_dy2st_auto_bs2_bf16_DP2-MP1-PP1-CINN
    )
    if [ $1 = "prepare_case" ]; then
        restore_func $fun_list  
    elif [ $1 = "exec_case" ]; then
        for fun in "${fun_list[@]}"; do
            eval "$fun"
        done
        track_case_status $FUNCNAME "llama_"
    else 
        echo -e "\033[31m ---- Invalid status $1 \033[0m"
        return 1
    fi
}


function llm_gpt_case_list_auto() {
    fun_list=(
        # The test name must have "llm_gpt_dygraph_auto_" as a prefix, 
        # which will be used for tracking the execution status of the case.
        llm_gpt_dygraph_auto_bs8_fp32_DP2
        llm_gpt_dygraph_auto_bs8_fp32_DP2-MP2
        llm_gpt_dygraph_auto_bs8_fp32_DP2-MP2-PP2
        llm_gpt_dygraph_auto_bs8_fp16_DP2-MP2-PP2
        llm_gpt_pir_auto_bs4_TP2
        llm_gpt_pir_auto_bs4_TP2_PP2
        llm_gpt_pir_auto_bs8_DP2_TP2_PP2
    )
    if [ $1 = "prepare_case" ]; then
        restore_func $fun_list  
    elif [ $1 = "exec_case" ]; then
        for fun in "${fun_list[@]}"; do
            eval "$fun"
        done
        track_case_status $FUNCNAME "llm_gpt"
    else 
        echo -e "\033[31m ---- Invalid status $1 \033[0m"
        return 1
    fi
}

function llm_qwen_case_list_auto() {
    fun_list=(
        # The test name must have "llm_qwen_dygraph_auto_" as a prefix, 
        # which will be used for tracking the execution status of the case.
        llm_qwen_dygraph_auto_bs1_fp32_DP2
        llm_qwen_dygraph_auto_bs1_fp32_DP2-MP2
        llm_qwen_dygraph_auto_bs1_fp32_DP2-MP2-PP2
        llm_qwen_dygraph_auto_bs1_bf16_DP2-MP2-PP2
        llm_qwen_pir_auto_bs1_bf16_TP2
        llm_qwen_pir_auto_bs1_bf16_TP2_PP2
    )
    if [ $1 = "prepare_case" ]; then
        restore_func $fun_list  
    elif [ $1 = "exec_case" ]; then
        for fun in "${fun_list[@]}"; do
            eval "$fun"
        done
        track_case_status $FUNCNAME "llm_qwen"
    else 
        echo -e "\033[31m ---- Invalid status $1 \033[0m"
        return 1
    fi
}

############ case start ############

function llama_dygraph_auto_bs8_fp32_DP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0

    task_name="llama_auto_bs8_dp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1" --log_dir $case_log_dir run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        --scale_loss 1024 \
        --pipeline_parallel_degree 1 \
        --tensor_parallel_degree 1 \
        --sharding_parallel_degree 1 \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --sharding "" \
        --eval_steps 1000000 \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 0 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --max_grad_norm 1.0 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.4992733
    if [ $IS_A100 -ne 0 ];then
        loss_base=9.50651741
    fi
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_dygraph_auto_bs8_fp32_DP2-MP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0

    task_name="llama_auto_bs8_dp2mp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir $case_log_dir run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        --scale_loss 1024 \
        --pipeline_parallel_degree 1 \
        --tensor_parallel_degree 2 \
        --sharding_parallel_degree 1 \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --sharding "" \
        --eval_steps 1000000 \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 0 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --max_grad_norm 1.0 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.3507843
    if [ $IS_A100 -ne 0 ];then
        loss_base=9.38577747
    fi
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_dygraph_auto_bs8_fp32_DP2-MP2-PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0

    task_name="llama_auto_bs8_dp2mp2pp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" --log_dir $case_log_dir run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        --scale_loss 1024 \
        --pipeline_parallel_degree 2 \
        --tensor_parallel_degree 2 \
        --sharding_parallel_degree 1 \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --sharding "" \
        --eval_steps 1000000 \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 0 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --max_grad_norm 1.0 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.3513937
    if [ $IS_A100 -ne 0 ];then
        loss_base=9.39356422
    fi
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_dygraph_auto_bs8_fp16_DP2-MP2-PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0

    task_name="llama_auto_bs8_fp16_dp2mp2pp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" --log_dir $case_log_dir run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 1 \
        --fp16_opt_level "O2" \
        --amp_master_grad 1 \
        --scale_loss 1024 \
        --pipeline_parallel_degree 2 \
        --tensor_parallel_degree 2 \
        --sharding_parallel_degree 1 \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --sharding "" \
        --eval_steps 1000000 \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 0 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --max_grad_norm 1.0 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.35162258
    if [ $IS_A100 -ne 0 ];then
        loss_base=9.39368534
    fi
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_dy2st_auto_bs4_bf16_DP1-MP1-PP4-SD2() {
    # Only A100 support this case.
    echo IS_A100 is $IS_A100
    if [ $IS_A100 -ne 0 ]; then
        echo "=========== $FUNCNAME run begin ==========="
        export PYTHONPATH=$root_path/:$PYTHONPATH
        export FLAGS_call_stack_level=3
        export NVIDIA_TF32_OVERRIDE=0

        export FLAGS_cudnn_deterministic=1
        export FLAGS_embedding_deterministic=1 
        
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export PARALLEL_CROSS_ENTROPY=true

        task_name="llama_dy2st_auto_bs4_bf16_DP1-MP1-PP4-SD2"
        case_out_dir="output/$task_name"
        case_log_dir="output/$task_name""_log"
        rm -rf $case_out_dir
        rm -rf $case_log_dir

        python -u  -m paddle.distributed.launch \
            --gpus "0,1,2,3,4,5,6,7" \
            --log_dir  "output/$task_name""_log" \
            ./run_pretrain_auto.py \
            --model_name_or_path "meta-llama/Llama-2-13b" \
            --tokenizer_name_or_path "meta-llama/Llama-2-13b" \
            --input_dir "./data" \
            --output_dir "./output" \
            --split 949,50,1 \
            --weight_decay 0.01 \
            --warmup_ratio 0.01 \
            --max_grad_norm 1.0 \
            --learning_rate 3e-05 \
            --min_learning_rate 3e-06 \
            --max_steps 30 \
            --logging_steps 10 \
            --eval_steps 1000 \
            --save_steps 50000 \
            --continue_training 0 \
            --do_train true \
            --do_eval false \
            --do_predict false \
            --disable_tqdm true \
            --skip_profile_timer true \
            --save_total_limit 2 \
            --device gpu \
            --disable_tqdm true \
            --dataloader_num_workers 1 \
            --distributed_dataloader 0 \
            --enable_auto_parallel 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --per_device_eval_batch_size 1 \
            --recompute false \
            --recompute_use_reentrant true \
            --recompute_granularity full \
            --pp_recompute_interval 0 \
            --bf16 true \
            --fp16_opt_level "O2"  \
            --amp_master_grad true \
            --fuse_attention_ffn false \
            --fuse_attention_qkv true \
            --fused_linear_param_grad_add 1 \
            --fuse_sequence_parallel_allreduce false \
            --use_flash_attention true \
            --use_fused_rope true \
            --use_fused_rms_norm true \
            --max_seq_length 4096 \
            --sep_parallel_degree 1 \
            --sequence_parallel false \
            --pipeline_parallel_degree 4 \
            --sharding_parallel_degree 2 \
            --tensor_parallel_degree 1 \
            --virtual_pp_degree 3 \
            --pipeline_schedule_mode "VPP" \
            --sharding "stage2" \
            --pipeline_parallel_config "enable_send_recv_overlap" \
            --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate" \
            --sharding_parallel_config "enable_stage2_overlap" \
            --tensor_parallel_config "enable_mp_async_allreduce" \
            --to_static 1 \
            --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
            --amp_custom_white_list "lookup_table" "lookup_table_v2" \
            --num_hidden_layers 12 \
            --skip_memory_metrics 0 \
            >>${log_path}/$FUNCNAME 2>&1
        loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
        ips=`cat $case_log_dir/workerlog.0 | grep 'global_step: 30' | awk -F 'interval_tokens_per_second_per_device: ' '{print $2}' | awk -F ',' '{print $1}'`
        mem=`cat $case_log_dir/workerlog.0 | grep 'global_step: 30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ',' '{print $1}'`
        echo "result: loss=$loss ips=$ips mem=$mem"
        loss_base=7.57775269
        ips_base=5442.5208
        mem_base=25.066193342208862
        check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
        echo "=========== $FUNCNAME run  end ==========="
    fi
}

function llama_dy2st_auto_bs4_bf16_DP1-MP1-PP4-SD2-VPP3_split_bw() {
    # Only A100 support this case.
    echo IS_A100 is $IS_A100
    if [ $IS_A100 -ne 0 ]; then
        echo "=========== $FUNCNAME run begin ==========="
        export PYTHONPATH=$root_path/:$PYTHONPATH
        export FLAGS_call_stack_level=3
        export NVIDIA_TF32_OVERRIDE=0

        export FLAGS_cudnn_deterministic=1
        export FLAGS_embedding_deterministic=1 

        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export PARALLEL_CROSS_ENTROPY=true
        export FLAGS_enable_pir_api=True # 功能已经实现并监控，具体显存数值对齐 @卢畅

        task_name="llama_dy2st_auto_bs4_bf16_DP1-MP1-PP4-SD2-VPP3_split_bw"
        case_out_dir="output/$task_name"
        case_log_dir="output/$task_name""_log"
        rm -rf $case_out_dir
        rm -rf $case_log_dir

        python -u  -m paddle.distributed.launch \
            --gpus "0,1,2,3,4,5,6,7" \
            --log_dir  "output/$task_name""_log" \
            ./run_pretrain_auto.py \
            --model_name_or_path "meta-llama/Llama-2-13b" \
            --tokenizer_name_or_path "meta-llama/Llama-2-13b" \
            --input_dir "./data" \
            --output_dir "./output" \
            --split 949,50,1 \
            --weight_decay 0.01 \
            --warmup_ratio 0.01 \
            --max_grad_norm 1.0 \
            --learning_rate 3e-05 \
            --min_learning_rate 3e-06 \
            --max_steps 30 \
            --logging_steps 10 \
            --eval_steps 1000 \
            --save_steps 50000 \
            --continue_training 0 \
            --do_train true \
            --do_eval false \
            --do_predict false \
            --disable_tqdm true \
            --skip_profile_timer true \
            --save_total_limit 2 \
            --device gpu \
            --disable_tqdm true \
            --dataloader_num_workers 1 \
            --distributed_dataloader 0 \
            --enable_auto_parallel 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --per_device_eval_batch_size 1 \
            --recompute false \
            --recompute_use_reentrant true \
            --recompute_granularity full \
            --pp_recompute_interval 0 \
            --bf16 true \
            --fp16_opt_level "O2"  \
            --amp_master_grad true \
            --fuse_attention_ffn false \
            --fuse_attention_qkv true \
            --fused_linear_param_grad_add 1 \
            --fuse_sequence_parallel_allreduce false \
            --use_flash_attention true \
            --use_fused_rope true \
            --use_fused_rms_norm true \
            --max_seq_length 4096 \
            --sep_parallel_degree 1 \
            --sequence_parallel false \
            --pipeline_parallel_degree 4 \
            --sharding_parallel_degree 2 \
            --tensor_parallel_degree 1 \
            --virtual_pp_degree 3 \
            --pipeline_schedule_mode "VPP" \
            --sharding "stage2" \
            --pipeline_parallel_config "enable_send_recv_overlap enable_split_backward" \
            --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate" \
            --sharding_parallel_config "enable_stage2_overlap" \
            --tensor_parallel_config "enable_mp_async_allreduce" \
            --to_static 1 \
            --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
            --amp_custom_white_list "lookup_table" "lookup_table_v2" \
            --num_hidden_layers 12 \
            --skip_memory_metrics 0 \
            >>${log_path}/$FUNCNAME 2>&1
        loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
        ips=`cat $case_log_dir/workerlog.0 | grep 'global_step: 30' | awk -F 'interval_tokens_per_second_per_device: ' '{print $2}' | awk -F ',' '{print $1}'`
        mem=`cat $case_log_dir/workerlog.0 | grep 'global_step: 30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ',' '{print $1}'`
        echo "result: loss=$loss ips=$ips mem=$mem"
        loss_base=7.57775269 # record new data
        ips_base=5825.427
        mem_base=25.562287092208862
        check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
        echo "=========== $FUNCNAME run  end ==========="  
    fi
}

function llama_align_dygraph_dy2st_pir_auto_bs2_bf16_DP2-MP2-PP1-SP() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export PYTHONPATH=/paddle/Paddle/build_gpu/python/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export FLAGS_enable_pir_api=1
    export FLAGS_dynamic_static_unified_comm=1
    export FLAGS_enable_auto_parallel_align_mode=1

    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_cudnn_deterministic=1
    export FLAGS_embedding_deterministic=1

    task_name="llama_align_dygraph_dy2st_pir_auto_bs2_bf16_dp2mp2pp1_sp"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"

    for to_static in "0" "1"; do
        for use_recompute in "1" "0"; do
            rm -rf $case_out_dir
            rm -rf $case_log_dir
            python -u -m paddle.distributed.launch \
                --gpus "0,1,2,3" \
                --log_dir $case_log_dir \
                run_pretrain_auto.py \
                --model_type "llama" \
                --model_name_or_path "facebook/llama-7b" \
                --tokenizer_name_or_path "facebook/llama-7b" \
                --input_dir "./data" \
                --output_dir $case_out_dir \
                --split 949,50,1 \
                --weight_decay 0.01 \
                --warmup_ratio 0.01 \
                --max_grad_norm 0.0 \
                --learning_rate 3e-05 \
                --min_learning_rate 3e-06 \
                --max_steps 10 \
                --logging_steps 10 \
                --eval_steps 1000 \
                --save_steps 50000 \
                --continue_training 0 \
                --do_train true \
                --do_eval false \
                --do_predict false \
                --disable_tqdm true \
                --skip_profile_timer true \
                --save_total_limit 2 \
                --device gpu \
                --disable_tqdm true \
                --dataloader_num_workers 1 \
                --enable_auto_parallel 1 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 1 \
                --per_device_eval_batch_size 2 \
                --recompute ${use_recompute} \
                --bf16 1\
                --fp16_opt_level "O2"  \
                --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
                --amp_custom_white_list "lookup_table" "lookup_table_v2" \
                --amp_master_grad 1 \
                --fuse_attention_ffn false \
                --fuse_attention_qkv false \
                --fuse_sequence_parallel_allreduce false \
                --use_flash_attention 0 \
                --use_fused_rope false \
                --use_fused_rms_norm 0 \
                --max_seq_length 4096 \
                --sep_parallel_degree 1 \
                --sequence_parallel true \
                --pipeline_parallel_degree 1 \
                --sharding_parallel_degree 1 \
                --tensor_parallel_degree 2 \
                --virtual_pp_degree 1 \
                --sharding "" \
                --to_static ${to_static} \
                --num_hidden_layers 4 \
                >>${log_path}/$FUNCNAME 2>&1
            loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
            loss_md5=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss_md5: ' '{print $2}' | awk -F ',' '{print $1}'`
            ips=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'interval_tokens_per_second_per_device: ' '{print $2}' | awk -F ',' '{print $1}'`
            mem=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ',' '{print $1}'`
            echo "result: to_static=$to_static loss=$loss ips=$ips mem=$mem"
            loss_base=9.16783295
            loss_md5_base=8ea72495fba4e1b9ba004b4431e27218
            if [ $IS_A100 -ne 0 ] && [ $to_static -eq 0 ];then
                loss_base=9.37966919
            elif [ $IS_A100 -ne 0 ] && [ $to_static -eq 1 ];then
                loss_base=9.38012543
            fi
            ips=-1
            mem=-1
            ips_base=-1
            mem_base=-1
            check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
            # check_md5_result $FUNCNAME ${loss_md5_base} ${loss_md5}
        done
    done
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_pir_auto_fuse_ffn_attention_qkv_MP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export FLAGS_max_inplace_grad_add=100
    export FLAGS_cudnn_deterministic=1
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_embedding_deterministic=1
    export FLAGS_flash_attn_version=v1
    export PARALLEL_CROSS_ENTROPY=true
    export FLAGS_enable_auto_parallel_align_mode=1

    export FLAGS_enable_pir_api=1
    export FLAGS_enable_fused_ffn_qkv_pass=1

    auto_task_name="llama_pir_auto_fuse_ffn_attention_qkv_MP2"
    auto_case_out_dir="auto_output/$auto_task_name"
    auto_case_log_dir="auto_output/$auto_task_name""_log"
    
    tp_configs=(
        "--tensor_parallel_config replace_with_parallel_cross_entropy"
        " "
    )
    for tp_config in "${tp_configs[@]}"; do
        rm -rf $auto_case_out_dir
        rm -rf $auto_case_log_dir
        python -u -m paddle.distributed.launch \
            --gpus "0,1" \
            --log_dir $auto_case_log_dir \
            run_pretrain_auto.py \
            --model_name_or_path "facebook/llama-7b" \
            --tokenizer_name_or_path "facebook/llama-7b" \
            --input_dir "./data" \
            --output_dir $auto_case_out_dir \
            --split 949,50,1 \
            --weight_decay 0.01 \
            --warmup_ratio 0.01 \
            --warmup_steps 30 \
            --max_grad_norm 0.0 \
            --learning_rate 3e-05 \
            --min_learning_rate 3e-06 \
            --max_steps 10 \
            --logging_steps 1 \
            --eval_steps 1000 \
            --save_steps 3 \
            --continue_training 0 \
            --do_train true \
            --do_eval false \
            --do_predict false \
            --disable_tqdm true \
            --skip_profile_timer true \
            --save_total_limit 2 \
            --device gpu \
            --disable_tqdm true \
            --dataloader_num_workers 1 \
            --distributed_dataloader 0 \
            --enable_auto_parallel 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --per_device_eval_batch_size 2 \
            --recompute false \
            --recompute_use_reentrant true \
            --recompute_granularity full \
            --pp_recompute_interval 0 \
            --bf16 0 \
            --fp16_opt_level "O2"  \
            --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
            --amp_custom_white_list "lookup_table" "lookup_table_v2" \
            --amp_master_grad false \
            --fuse_attention_ffn false \
            --fuse_attention_qkv false \
            --use_flash_attention false \
            --use_fused_rope true \
            --use_fused_rms_norm true \
            --max_seq_length 4096 \
            --sequence_parallel false \
            --pipeline_parallel_degree 1 \
            --sharding_parallel_degree 1 \
            --tensor_parallel_degree 2 \
            ${tp_config} \
            --virtual_pp_degree 1 \
            --pipeline_schedule_mode "VPP" \
            --sharding "" \
            --to_static 1 \
            --num_hidden_layers 2 \
            >>${log_path}/$FUNCNAME 2>&1

        auto_loss_2=`cat $auto_case_log_dir/workerlog.0 | grep 'global_step: 2' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
        loss_md5_2=`cat $auto_case_log_dir/workerlog.0 | grep 'global_step: 2' | awk -F 'loss_md5: ' '{print $2}' | awk -F ',' '{print $1}'`
        auto_ips_2=`cat $auto_case_log_dir/workerlog.0 | grep 'global_step: 2' | awk -F 'interval_tokens_per_second_per_device: ' '{print $2}' | awk -F ',' '{print $1}'`
        auto_mem_2=`cat $auto_case_log_dir/workerlog.0 | grep 'global_step: 2' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ',' '{print $1}'`
        echo "auto result: step 2 loss=$auto_loss_2 ips=$auto_ips_2 mem=$auto_mem_2"
        auto_loss_10=`cat $auto_case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
        loss_md5_10=`cat $auto_case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss_md5: ' '{print $2}' | awk -F ',' '{print $1}'`
        auto_ips_10=`cat $auto_case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'interval_tokens_per_second_per_device: ' '{print $2}' | awk -F ',' '{print $1}'`
        auto_mem_10=`cat $auto_case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ',' '{print $1}'`
        echo "auto result: step 10 loss=$auto_loss_10 ips=$auto_ips_10 mem=$auto_mem_10"
        if [[ $tp_config =~ "replace_with_parallel_cross_entropy" ]];then
            # This optimization may result in a discrepancy in accuracy.
            loss_base_2=10.53477287
            loss_base_10=9.4961338
        else
            loss_base_2=10.53477192
            loss_base_10=9.4961338
        fi
        auto_ips=-1
        auto_mem=-1
        ips_base=-1
        mem_base=-1
        if [ $IS_A100 -ne 0 ];then
            loss_base_2=10.58283806
            loss_base_10=10.58283806
        fi
        check_result $FUNCNAME ${loss_base_2} ${auto_loss_2} ${ips_base} ${auto_ips} ${mem_base} ${auto_mem}
        check_result $FUNCNAME ${loss_base_10} ${auto_loss_10} ${ips_base} ${auto_ips} ${mem_base} ${auto_mem}
    done
    export FLAGS_enable_fused_ffn_qkv_pass=0
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_align_dygraph_dy2st_pir_auto_bs2_bf16_DP2-MP2-PP2-SP() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export PYTHONPATH=/paddle/Paddle/build_gpu/python/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export FLAGS_enable_pir_api=1
    export FLAGS_dynamic_static_unified_comm=1
    export FLAGS_enable_auto_parallel_align_mode=1

    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_cudnn_deterministic=1
    export FLAGS_embedding_deterministic=1

    task_name="llama_align_dygraph_dy2st_pir_auto_bs2_bf16_dp2mp2pp2_sp"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"

    for to_static in "0" "1"; do
        rm -rf $case_out_dir
        rm -rf $case_log_dir
        python -u -m paddle.distributed.launch \
            --gpus "0,1,2,3,4,5,6,7" \
            --log_dir $case_log_dir \
            run_pretrain_auto.py \
            --model_type "llama" \
            --model_name_or_path "facebook/llama-7b" \
            --tokenizer_name_or_path "facebook/llama-7b" \
            --input_dir "./data" \
            --output_dir $case_out_dir \
            --split 949,50,1 \
            --weight_decay 0.01 \
            --warmup_ratio 0.01 \
            --max_grad_norm 0.0 \
            --learning_rate 3e-05 \
            --min_learning_rate 3e-06 \
            --max_steps 10 \
            --logging_steps 10 \
            --eval_steps 1000 \
            --save_steps 50000 \
            --continue_training 0 \
            --do_train true \
            --do_eval false \
            --do_predict false \
            --disable_tqdm true \
            --skip_profile_timer true \
            --save_total_limit 2 \
            --device gpu \
            --disable_tqdm true \
            --dataloader_num_workers 1 \
            --enable_auto_parallel 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --per_device_eval_batch_size 2 \
            --recompute false \
            --bf16 1\
            --fp16_opt_level "O2"  \
            --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
            --amp_custom_white_list "lookup_table" "lookup_table_v2" \
            --amp_master_grad 1 \
            --fuse_attention_ffn false \
            --fuse_attention_qkv false \
            --fuse_sequence_parallel_allreduce false \
            --use_flash_attention 0 \
            --use_fused_rope false \
            --use_fused_rms_norm 0 \
            --max_seq_length 4096 \
            --sep_parallel_degree 1 \
            --sequence_parallel true \
            --pipeline_parallel_degree 2 \
            --sharding_parallel_degree 1 \
            --tensor_parallel_degree 2 \
            --virtual_pp_degree 1 \
            --sharding "" \
            --to_static ${to_static} \
            --num_hidden_layers 4 \
            >>${log_path}/$FUNCNAME 2>&1
        loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
        loss_md5=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss_md5: ' '{print $2}' | awk -F ',' '{print $1}'`
        ips=-1
        mem=-1
        echo "result: to_static=$to_static loss=$loss loss_md5=$loss_md5 ips=$ips mem=$mem"
        loss_base=9.25199432
        loss_md5_base=83531e98ee11cd271db175150ab254bb
        if [ $IS_A100 -ne 0 ] && [ $to_static -eq 0 ];then
            loss_base=9.44203949
        elif [ $IS_A100 -ne 0 ] && [ $to_static -eq 1 ];then
            loss_base=9.44225311
        fi
        ips_base=-1
        mem_base=-1
        check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
        # check_md5_result $FUNCNAME ${loss_md5_base} ${loss_md5}
    done
    echo "=========== $FUNCNAME run  end ==========="
}


function llama_align_dygraph_dy2st_auto_bs2_bf16_DP2-MP1-PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_enable_pir_api=1
    export FLAGS_max_inplace_grad_add=4

    task_name="llama_align_dygraph_dy2st_auto_bs2_bf16_dp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"

    for to_static in "0" "1"; do
        rm -rf $case_out_dir
        rm -rf $case_log_dir
        python -u -m paddle.distributed.launch \
            --gpus "0,1" \
            --log_dir $case_log_dir \
            run_pretrain_auto.py \
            --model_type "llama" \
            --model_name_or_path "facebook/llama-7b" \
            --tokenizer_name_or_path "facebook/llama-7b" \
            --input_dir "./data" \
            --output_dir $case_out_dir \
            --split 949,50,1 \
            --weight_decay 0.01 \
            --warmup_ratio 0.01 \
            --warmup_steps 30 \
            --max_grad_norm 1.0 \
            --learning_rate 3e-05 \
            --min_learning_rate 3e-06 \
            --max_steps 10 \
            --logging_steps 10 \
            --eval_steps 1000 \
            --save_steps 50000 \
            --continue_training 0 \
            --do_train true \
            --do_eval false \
            --do_predict false \
            --disable_tqdm true \
            --skip_profile_timer true \
            --save_total_limit 2 \
            --device gpu \
            --disable_tqdm true \
            --dataloader_num_workers 1 \
            --distributed_dataloader 0 \
            --enable_auto_parallel 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --per_device_eval_batch_size 2 \
            --recompute false \
            --recompute_use_reentrant true \
            --recompute_granularity full \
            --pp_recompute_interval 0 \
            --bf16 1 \
            --fp16_opt_level "O2"  \
            --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
            --amp_custom_white_list "lookup_table" "lookup_table_v2" \
            --amp_master_grad 1 \
            --fuse_attention_ffn true \
            --fuse_attention_qkv true \
            --fuse_sequence_parallel_allreduce false \
            --use_flash_attention 0 \
            --use_fused_rope false \
            --use_fused_rms_norm 1 \
            --max_seq_length 4096 \
            --sep_parallel_degree 1 \
            --sequence_parallel false \
            --pipeline_parallel_degree 1 \
            --sharding_parallel_degree 1 \
            --tensor_parallel_degree 1 \
            --virtual_pp_degree 1 \
            --pipeline_schedule_mode "VPP" \
            --sharding "" \
            --to_static ${to_static} \
            --num_hidden_layers 2 \
            >>${log_path}/$FUNCNAME 2>&1
        loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
        ips=-1
        mem=-1
        echo "result: to_static=$to_static loss=$loss ips=$ips mem=$mem"
        loss_base=9.99302673
        if [ $IS_A100 -ne 0 ];then
            loss_base=10.20991516
        fi
        ips_base=-1
        mem_base=-1
        check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    done
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_dy2st_auto_bs2_bf16_DP2-MP1-PP1-CINN() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export FLAGS_cudnn_deterministic=1
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_embedding_deterministic=1
    export FLAGS_flash_attn_version=v1
    export FLAGS_enable_pir_api=1
    export FLAGS_max_inplace_grad_add=4
    export PARALLEL_CROSS_ENTROPY=true

    export FLAGS_use_cinn=1
    export FLAGS_dist_prim_all=1
    export FLAGS_prim_forward_blacklist="pd_op.stack;pd_op.squeeze;pd_op.swiglu;pd_op.squared_l2_norm"
    export FLAGS_prim_backward_blacklist="swiglu_grad"

    task_name="llama_dy2st_auto_bs2_bf16_DP2-MP1-PP1-CINN"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch \
        --gpus "0,1" \
        --log_dir $case_log_dir \
        run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --warmup_steps 30 \
        --max_grad_norm 1.0 \
        --learning_rate 3e-05 \
        --min_learning_rate 3e-06 \
        --max_steps 10 \
        --logging_steps 10 \
        --eval_steps 1000 \
        --save_steps 50000 \
        --continue_training 0 \
        --do_train true \
        --do_eval false \
        --do_predict false \
        --disable_tqdm true \
        --skip_profile_timer true \
        --save_total_limit 2 \
        --device gpu \
        --disable_tqdm true \
        --dataloader_num_workers 1 \
        --distributed_dataloader 0 \
        --enable_auto_parallel 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 2 \
        --recompute false \
        --recompute_use_reentrant true \
        --recompute_granularity full \
        --pp_recompute_interval 0 \
        --bf16 1 \
        --fp16_opt_level "O2"  \
        --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
        --amp_custom_white_list "lookup_table" "lookup_table_v2" \
        --amp_master_grad 1 \
        --fuse_attention_ffn true \
        --fuse_attention_qkv true \
        --fuse_sequence_parallel_allreduce false \
        --use_flash_attention 0 \
        --use_fused_rope false \
        --use_fused_rms_norm false \
        --max_seq_length 4096 \
        --sep_parallel_degree 1 \
        --sequence_parallel false \
        --pipeline_parallel_degree 1 \
        --sharding_parallel_degree 1 \
        --tensor_parallel_degree 1 \
        --virtual_pp_degree 1 \
        --pipeline_schedule_mode "VPP" \
        --sharding "" \
        --to_static ${to_static} \
        --num_hidden_layers 2 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: to_static=$to_static loss=$loss ips=$ips mem=$mem"
    loss_base=9.99302597
    if [ $IS_A100 -ne 0 ];then
        loss_base=10.20988007
    fi
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}

    unset FLAGS_use_cinn
    unset FLAGS_dist_prim_all
    unset FLAGS_prim_forward_blacklist
    unset FLAGS_prim_backward_blacklist
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_align_dygraph_dy2st_pir_auto_grad_merge_bs2_fp32_DP1-MP1-PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_max_inplace_grad_add=3

    task_name="llama_align_dygraph_dy2st_pir_auto_grad_merge_bs2_fp32_DP2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"

    loss1=0
    loss2=0
    use_pir=1
    max_step=12

    for to_static in "0" "1"; do
        export FLAGS_enable_pir_api=${use_pir}
        export FLAGS_enable_pir_in_executor=${use_pir}
        rm -rf $case_out_dir
        rm -rf $case_log_dir
        rm -rf ${log_path}/$FUNCNAME

        /usr/bin/python -u -m paddle.distributed.launch \
            --gpus "0" \
            --log_dir $case_log_dir \
            run_pretrain_auto.py \
            --model_type "llama" \
            --model_name_or_path "facebook/llama-7b" \
            --tokenizer_name_or_path "facebook/llama-7b" \
            --input_dir "./data" \
            --output_dir $case_out_dir \
            --split 949,50,1 \
            --weight_decay 0.01 \
            --warmup_ratio 0.01 \
            --warmup_steps 30 \
            --max_grad_norm 0.0 \
            --learning_rate 3e-05 \
            --min_learning_rate 3e-06 \
            --max_steps $max_step \
            --logging_steps 1 \
            --eval_steps 1000 \
            --save_steps 50000 \
            --continue_training 0 \
            --do_train true \
            --do_eval false \
            --do_predict false \
            --disable_tqdm true \
            --skip_profile_timer true \
            --save_total_limit 2 \
            --device gpu \
            --disable_tqdm true \
            --dataloader_num_workers 1 \
            --distributed_dataloader 0 \
            --enable_auto_parallel 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 2 \
            --recompute false \
            --recompute_use_reentrant true \
            --recompute_granularity full \
            --pp_recompute_interval 0 \
            --fp16 0 \
            --fp16_opt_level "O2" \
            --fuse_attention_ffn true \
            --fuse_attention_qkv false \
            --fuse_sequence_parallel_allreduce false \
            --use_flash_attention 0 \
            --use_fused_rope false \
            --use_fused_rms_norm 0 \
            --max_seq_length 2048 \
            --sep_parallel_degree 1 \
            --sequence_parallel false \
            --pipeline_parallel_degree 1 \
            --sharding_parallel_degree 1 \
            --tensor_parallel_degree 1 \
            --virtual_pp_degree 1 \
            --sharding "" \
            --to_static ${to_static} \
            --num_hidden_layers 2 \
            --data_parallel_config "gradient_sync_after_accumulate" \
            >>${log_path}/$FUNCNAME 2>&1

        loss=$(grep "global_step: $max_step" "$case_log_dir/workerlog.0" | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}')
        if [ $to_static -eq 0 ];then
            loss1=($loss)
        else
            loss2=($loss)
        fi
        echo "result: to_static=$to_static loss=$loss"
    done

    ips=-1
    mem=-1
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss1} ${loss2} ${ips_base} ${ips} ${mem_base} ${mem}
}

function llama_align_dy2st_fthenb_and_vpp_auto_bs2_fp32_DP1-MP1-PP4() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_max_inplace_grad_add=3

    task_name="llama_align_dy2st_fthenb_and_vpp_auto_bs2_fp32_DP1_MP1_PP4"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    loss1=0
    loss2=0
    use_pir=1

    max_step=10
    to_static=1
    loss1_array=()
    loss2_array=()

    for pp_mode in "FThenB" "VPP"; do
        export FLAGS_enable_pir_api=${use_pir}
        export FLAGS_enable_pir_in_executor=${use_pir}
        rm -rf $case_out_dir
        rm -rf $case_log_dir
        rm -rf ${log_path}/$FUNCNAME
        if [ "$pp_mode" == "FThenB" ]; then
            vpp_degree=1
        else
            vpp_degree=2
        fi

        python -u -m paddle.distributed.launch \
            --gpus "0,1,2,3" \
            --log_dir $case_log_dir \
            run_pretrain_auto.py \
            --model_type "llama" \
            --model_name_or_path "facebook/llama-7b" \
            --tokenizer_name_or_path "facebook/llama-7b" \
            --input_dir "./data" \
            --output_dir $case_out_dir \
            --split 949,50,1 \
            --weight_decay 0.01 \
            --warmup_ratio 0.01 \
            --warmup_steps 30 \
            --max_grad_norm 0.0 \
            --learning_rate 3e-05 \
            --min_learning_rate 3e-06 \
            --max_steps $max_step \
            --logging_steps 1 \
            --eval_steps 1000 \
            --save_steps 50000 \
            --continue_training 0 \
            --do_train true \
            --do_eval false \
            --do_predict false \
            --disable_tqdm true \
            --skip_profile_timer true \
            --save_total_limit 2 \
            --device gpu \
            --disable_tqdm true \
            --dataloader_num_workers 1 \
            --distributed_dataloader 0 \
            --enable_auto_parallel 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --per_device_eval_batch_size 2 \
            --recompute false \
            --recompute_use_reentrant true \
            --recompute_granularity full \
            --fp16 0 \
            --fp16_opt_level "O2" \
            --fuse_attention_ffn true \
            --fuse_attention_qkv true \
            --fuse_sequence_parallel_allreduce false \
            --use_flash_attention 0 \
            --use_fused_rope false \
            --use_fused_rms_norm 0 \
            --max_seq_length 2048 \
            --hidden_size 1024 \
            --sep_parallel_degree 1 \
            --sequence_parallel false \
            --pipeline_parallel_degree 4 \
            --sharding_parallel_degree 1 \
            --tensor_parallel_degree 1 \
            --sharding "" \
            --to_static ${to_static} \
            --num_hidden_layers 8 \
            --data_parallel_config "gradient_sync_after_accumulate" \
            --pipeline_schedule_mode $pp_mode \
            --virtual_pp_degree $vpp_degree \
            >>${log_path}/$FUNCNAME 2>&1

        for step in $(seq 1 $max_step); do
            loss=$(grep "global_step: $step," "$case_log_dir/workerlog.0" | grep -oP '(?<=loss: )\d+(\.\d+)?' | awk -F ',' '{print $1}')
            if [ "$pp_mode" == "FThenB" ]; then
                loss1_array+=($loss)
            else
                loss2_array+=($loss)
            fi
        done

        loss=$(grep "global_step: 10," "$case_log_dir/workerlog.0" | grep -oP '(?<=loss: )\d+(\.\d+)?' | awk -F ',' '{print $1}')
        if [ "$pp_mode" == "FThenB" ]; then
            loss1=($loss)
        else
            loss2=($loss)
        fi
        echo "result: $pp_mode loss=$loss"
    done
    ips=-1
    mem=-1
    ips_base=-1
    mem_base=-1
    for step in $(seq 1 $max_step); do
        echo "step=$step fthenb loss: ${loss1_array[$step-1]}, vpp loss: ${loss2_array[$step-1]}"
    done
    check_result $FUNCNAME ${loss1} ${loss2} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_align_dygraph_dy2st_pir_auto_pp_bs2_bf16_DP1-MP1-PP4() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_max_inplace_grad_add=3

    task_name="llama_align_dygraph_dy2st_pir_auto_pp_bs2_bf16_DP1_MP1_PP4"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    loss1=0
    loss2=0
    loss1_array=()
    loss2_array=()
    use_pir=1

    max_step=15
    to_static=1

    for to_static in "0" "1"; do
        export FLAGS_enable_pir_api=${use_pir}
        export FLAGS_enable_pir_in_executor=${use_pir}

        case_out_dir="output/$task_name"
        case_log_dir="output/$task_name""_log$to_static"
        rm -rf $case_out_dir
        rm -rf $case_log_dir
        rm -rf ${log_path}/$FUNCNAME

        python -u -m paddle.distributed.launch \
            --gpus "0,1,2,3" \
            --log_dir $case_log_dir \
            run_pretrain_auto.py \
            --model_type "llama" \
            --model_name_or_path "facebook/llama-7b" \
            --tokenizer_name_or_path "facebook/llama-7b" \
            --input_dir "./data" \
            --output_dir $case_out_dir \
            --split 949,50,1 \
            --weight_decay 0.01 \
            --warmup_ratio 0.01 \
            --warmup_steps 30 \
            --max_grad_norm 0.0 \
            --learning_rate 3e-05 \
            --min_learning_rate 3e-06 \
            --max_steps $max_step \
            --logging_steps 1 \
            --eval_steps 1000 \
            --save_steps 50000 \
            --continue_training 0 \
            --do_train true \
            --do_eval false \
            --do_predict false \
            --disable_tqdm true \
            --skip_profile_timer true \
            --save_total_limit 2 \
            --device gpu \
            --disable_tqdm true \
            --dataloader_num_workers 1 \
            --distributed_dataloader 0 \
            --enable_auto_parallel 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 2 \
            --recompute false \
            --recompute_use_reentrant true \
            --recompute_granularity full \
            --bf16 true \
            --fp16_opt_level "O2" \
            --amp_master_grad true \
            --amp_custom_black_list ["reduce_sum", "c_softmax_with_cross_entropy"] \
            --amp_custom_white_list ["lookup_table", "lookup_table_v2"] \
            --fuse_attention_ffn true \
            --fuse_attention_qkv true \
            --fuse_sequence_parallel_allreduce false \
            --use_flash_attention 0 \
            --use_fused_rope false \
            --use_fused_rms_norm 0 \
            --max_seq_length 2048 \
            --hidden_size 1024 \
            --sep_parallel_degree 1 \
            --sequence_parallel false \
            --pipeline_parallel_degree 4 \
            --sharding_parallel_degree 1 \
            --tensor_parallel_degree 1 \
            --sharding "" \
            --to_static ${to_static} \
            --num_hidden_layers 8 \
            --data_parallel_config "gradient_sync_after_accumulate" \
            --pipeline_schedule_mode "FThenB" \
            >>${log_path}/$FUNCNAME 2>&1
        loss=$(grep "global_step: 15," "$case_log_dir/workerlog.0" | grep -oP '(?<=loss: )\d+(\.\d+)?' | awk -F ',' '{print $1}')
        if [ $to_static -eq 0 ]; then
            loss1=($loss)
        else
            loss2=($loss)
        fi
        echo "result: to_static=$to_static loss=$loss"
    done
    ips=-1
    mem=-1
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss1} ${loss2} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_convert_hybrid_ckpt_to_auto_parallel_bs2_fp32_DP2-MP1-PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_enable_pir_api=1
    export FLAGS_max_inplace_grad_add=3

    echo "---- run hybrid and save ckpt ----"
    dy_task_name="llama_hybrid_ckpt_bs2_fp32_DP2-MP1-PP1"
    dy_case_out_dir="dy_output/$dy_task_name"
    dy_case_log_dir="dy_output/$dy_task_name""_log"
    rm -rf $dy_case_out_dir
    rm -rf $dy_case_log_dir

    python -u -m paddle.distributed.launch \
        --gpus "0,1" \
        --log_dir $dy_case_log_dir \
        ../../run_pretrain.py \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $dy_case_out_dir \
        --split 949,50,1 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --warmup_steps 30 \
        --max_grad_norm 0.0 \
        --learning_rate 3e-05 \
        --min_learning_rate 3e-06 \
        --max_steps 5 \
        --logging_steps 1 \
        --eval_steps 1000 \
        --save_steps 3 \
        --continue_training 0 \
        --do_train true \
        --do_eval false \
        --do_predict false \
        --disable_tqdm true \
        --skip_profile_timer true \
        --save_total_limit 2 \
        --device gpu \
        --disable_tqdm true \
        --dataloader_num_workers 1 \
        --distributed_dataloader 0 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 2 \
        --recompute false \
        --recompute_use_reentrant true \
        --recompute_granularity full \
        --pp_recompute_interval 0 \
        --bf16 0 \
        --fp16_opt_level "O2"  \
        --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
        --amp_custom_white_list "lookup_table" "lookup_table_v2" \
        --amp_master_grad false \
        --enable_linear_fused_grad_add false \
        --fuse_attention_ffn true \
        --fuse_attention_qkv false \
        --fuse_sequence_parallel_allreduce false \
        --use_flash_attention 0 \
        --use_fused_rope false \
        --use_fused_rms_norm 0 \
        --max_seq_length 4096 \
        --sep_parallel_degree 1 \
        --sequence_parallel false \
        --pipeline_parallel_degree 1 \
        --sharding_parallel_degree 1 \
        --tensor_parallel_degree 1 \
        --virtual_pp_degree 1 \
        --sharding "" \
        --to_static 0 \
        --num_hidden_layers 2 \
        --unified_checkpoint false \
        >>${log_path}/$FUNCNAME 2>&1
    dy_loss=`cat $dy_case_log_dir/workerlog.0 | grep 'global_step: 4' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    dy_ips=-1
    dy_mem=-1
    echo "hybrid result: loss=$dy_loss ips=$dy_ips mem=$dy_mem"

    echo "---- run auto parallel resueme from hybrid ckpt ----"
    auto_task_name="llama_auto_parallel_bs2_fp32_DP2-MP1-PP1"
    auto_case_out_dir="auto_output/$auto_task_name"
    auto_case_log_dir="auto_output/$auto_task_name""_log"
    rm -rf $auto_case_out_dir
    rm -rf $auto_case_log_dir

    python -u -m paddle.distributed.launch \
        --gpus "0,1" \
        --log_dir $auto_case_log_dir \
        run_pretrain_auto.py \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $auto_case_out_dir \
        --split 949,50,1 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --warmup_steps 30 \
        --max_grad_norm 0.0 \
        --learning_rate 3e-05 \
        --min_learning_rate 3e-06 \
        --max_steps 4 \
        --logging_steps 1 \
        --eval_steps 1000 \
        --save_steps 1000 \
        --continue_training 0 \
        --do_train true \
        --do_eval false \
        --do_predict false \
        --disable_tqdm true \
        --skip_profile_timer true \
        --save_total_limit 2 \
        --device gpu \
        --disable_tqdm true \
        --dataloader_num_workers 1 \
        --distributed_dataloader 0 \
        --enable_auto_parallel 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 2 \
        --recompute false \
        --recompute_use_reentrant true \
        --recompute_granularity full \
        --pp_recompute_interval 0 \
        --bf16 0 \
        --fp16_opt_level "O2"  \
        --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
        --amp_custom_white_list "lookup_table" "lookup_table_v2" \
        --amp_master_grad false \
        --fuse_attention_ffn true \
        --fuse_attention_qkv false \
        --fuse_sequence_parallel_allreduce false \
        --use_flash_attention 0 \
        --use_fused_rope false \
        --use_fused_rms_norm 0 \
        --max_seq_length 4096 \
        --sep_parallel_degree 1 \
        --sequence_parallel false \
        --pipeline_parallel_degree 1 \
        --sharding_parallel_degree 1 \
        --tensor_parallel_degree 1 \
        --virtual_pp_degree 1 \
        --pipeline_schedule_mode "VPP" \
        --sharding "" \
        --to_static 1 \
        --num_hidden_layers 2 \
        --resume_from_checkpoint "dy_output/llama_hybrid_ckpt_bs2_fp32_DP2-MP1-PP1/checkpoint-3" \
        --auto_parallel_resume_form_hybrid_parallel 1 \
        >>${log_path}/$FUNCNAME 2>&1
    auto_loss=`cat $auto_case_log_dir/workerlog.0 | grep 'global_step: 4' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    auto_ips=-1
    auto_mem=-1
    echo "auto result: loss=$auto_loss ips=$auto_ips mem=$auto_mem"

    check_result $FUNCNAME ${dy_loss} ${auto_loss} ${dy_ips} ${auto_ips} ${dy_mem} ${auto_mem}
    echo "=========== $FUNCNAME run  end ==========="
}
function llama_baichuan_pir_auto_fuse_ffn_attention_qkv_DP2_MP2_PP2(){
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_enable_pir_api=1

    task_name="llama_baichuan_pir_auto_fuse_ffn_attention_qkv_DP2_MP2_PP2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" --log_dir $case_log_dir run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "baichuan-inc/Baichuan2-13B-Base" \
        --tokenizer_name_or_path "baichuan-inc/Baichuan2-13B-Base" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --to_static true \
        --pipeline_parallel_degree 2 \
        --tensor_parallel_degree 2 \
        --virtual_pp_degree 2\
        --pipeline_schedule_mode "1F1B" \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 0.0 \
        --learning_rate 3e-05 \
        --min_learning_rate 3e-06 \
        --max_steps 10 \
        --logging_steps 1 \
        --eval_steps 10000 \
        --save_steps 1000 \
        --continue_training 0 \
        --do_train true \
        --do_eval false \
        --do_predict false \
        --disable_tqdm true \
        --save_total_limit 2 \
        --device gpu \
        --dataloader_num_workers 4 \
        --distributed_dataloader 0 \
        --enable_auto_parallel 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 32 \
        --per_device_eval_batch_size 1 \
        --recompute false \
        --recompute_use_reentrant true \
        --recompute_granularity full \
        --pp_recompute_interval 0 \
        --bf16 true \
        --fp16_opt_level "O2"  \
        --amp_master_grad true \
        --fuse_attention_ffn true \
        --fuse_attention_qkv true \
        --use_flash_attention false \
        --use_fused_rope true \
        --use_fused_rms_norm false \
        --max_seq_length 4096 \
        --sequence_parallel false \
        --sharding "stage1" \
        --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate " \
        --sharding_parallel_config "enable_stage1_overlap" \
        --tensor_parallel_config "enable_mp_async_allreduce" \
        --pipeline_parallel_config "enable_send_recv_overlap" \
        --auto_parallel_resume_form_hybrid_parallel true \
        --num_hidden_layers 2 \
        >>${log_path}/$FUNCNAME 2>&1
    echo "=========== $FUNCNAME run  end ==========="
}
function llm_gpt_dygraph_auto_bs8_fp32_DP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0

    cd ${llm_gpt_case_path}
    task_name="gpt3_auto_bs8_dp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1" \
        --log_dir $case_log_dir \
        run_pretrain_auto.py \
        --model_name_or_path gpt2-medium-en \
        --tokenizer_name_or_path gpt2-medium-en \
        --input_dir "$gpt_data_path/data" \
        --output_dir "output/$task_name" \
        --split 949,50,1 \
        --max_seq_length 1024 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --sharding "" \
        --tensor_parallel_degree 1 \
        --pipeline_parallel_degree 1 \
        --sequence_parallel 0 \
        --fuse_attention_qkv 0 \
        --use_flash_attention 0 \
        --scale_loss 1024 \
        --learning_rate 0.00001 \
        --min_learning_rate 0.000005 \
        --max_steps 10 \
        --save_steps 50000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1\
        --continue_training 0\
        --dataloader_num_workers 1 \
        --eval_steps 100000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --recompute 0 \
        --gradient_accumulation_steps 4 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --model_type "gpt" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    loss_md5=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss_md5: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem loss_md5=$loss_md5"
    loss_base=10.59486389 # output of dropout is different after supporting spmd
    ips_base=-1
    mem_base=-1
    if [ $IS_A100 -ne 0 ];then
        loss_base=10.60063553 # after add dropout spmd
    fi
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_gpt_dygraph_auto_bs8_fp32_DP2-MP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_cudnn_deterministic=1
    export FLAGS_embedding_deterministic=1 

    cd ${llm_gpt_case_path}
    task_name="gpt3_auto_bs8_dp2mp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_log_dir
    rm -rf $case_out_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
        --log_dir $case_log_dir \
        run_pretrain_auto.py \
        --model_name_or_path gpt2-medium-en \
        --tokenizer_name_or_path gpt2-medium-en \
        --input_dir "$gpt_data_path/data" \
        --output_dir $case_out_dir  \
        --split 949,50,1 \
        --max_seq_length 1024 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --sharding "" \
        --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 1 \
        --sequence_parallel 0 \
        --fuse_attention_qkv 0 \
        --use_flash_attention 0 \
        --scale_loss 1024 \
        --learning_rate 0.00001 \
        --min_learning_rate 0.000005 \
        --max_steps 10 \
        --save_steps 50000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1\
        --continue_training 0\
        --dataloader_num_workers 1 \
        --eval_steps 100000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --recompute 0 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --model_type "gpt" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    loss_md5=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss_md5: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem loss_md5=$loss_md5"
    loss_base=10.58862114 # output of dropout is different after supporting spmd
    ips_base=-1
    mem_base=-1
    if [ $IS_A100 -ne 0 ];then
        loss_base=10.59354877 # after add dropout spmd
    fi
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_gpt_dygraph_auto_bs8_fp32_DP2-MP2-PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_cudnn_deterministic=1
    export FLAGS_embedding_deterministic=1 

    cd ${llm_gpt_case_path}
    task_name="gpt3_auto_bs8_dp2mp2pp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_log_dir
    rm -rf $case_out_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
        --log_dir $case_log_dir \
        run_pretrain_auto.py \
        --model_name_or_path gpt2-medium-en \
        --tokenizer_name_or_path gpt2-medium-en \
        --input_dir "$gpt_data_path/data" \
        --output_dir $case_out_dir  \
        --split 949,50,1 \
        --max_seq_length 1024 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --sharding "" \
        --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --sequence_parallel 0 \
        --fuse_attention_qkv 0 \
        --use_flash_attention 0 \
        --scale_loss 1024 \
        --learning_rate 0.00001 \
        --min_learning_rate 0.000005 \
        --max_steps 10 \
        --save_steps 50000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1\
        --continue_training 0\
        --dataloader_num_workers 1 \
        --eval_steps 100000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --recompute 0 \
        --gradient_accumulation_steps 4 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --model_type "gpt" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    loss_md5=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss_md5: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem loss_md5=$loss_md5"
    # loss_base=10.59993172     # note: need to debug
    loss_base=10.58122158 # output of dropout is different after supporting spmd
    ips_base=-1
    mem_base=-1
    if [ $IS_A100 -ne 0 ];then
        loss_base=10.58605194 # after add dropout spmd
    fi
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_gpt_dygraph_auto_bs8_fp16_DP2-MP2-PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_cudnn_deterministic=1
    export FLAGS_embedding_deterministic=1 

    cd ${llm_gpt_case_path}
    task_name="gpt3_auto_bs8_fp16_dp2mp2pp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_log_dir
    rm -rf $case_out_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
        --log_dir $case_log_dir \
        run_pretrain_auto.py \
        --model_name_or_path gpt2-medium-en \
        --tokenizer_name_or_path gpt2-medium-en \
        --input_dir "$gpt_data_path/data" \
        --output_dir $case_out_dir  \
        --split 949,50,1 \
        --max_seq_length 1024 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --sharding "" \
        --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --sequence_parallel 0 \
        --fuse_attention_qkv 0 \
        --use_flash_attention 0 \
        --scale_loss 1024 \
        --learning_rate 0.00001 \
        --min_learning_rate 0.000005 \
        --max_steps 10 \
        --save_steps 50000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1\
        --continue_training 0\
        --dataloader_num_workers 1 \
        --eval_steps 100000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --recompute 0 \
        --gradient_accumulation_steps 4 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --model_type "gpt" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --fp16 1 \
        --fp16_opt_level "O2" \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    loss_md5=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss_md5: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem loss_md5=$loss_md5"
    # loss_base=10.58456802     # note: need to debug
    loss_base=10.58163357
    ips_base=-1
    mem_base=-1
    if [ $IS_A100 -ne 0 ];then
        loss_base=10.58635044 # after add dropout spmd
    fi
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_gpt_pir_auto_bs4_TP2(){
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_enable_pir_api=1
    cd ${llm_gpt_case_path}

    task_name="gpt3_auto_bs4_tp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1" \
        --log_dir $case_log_dir \
        run_pretrain_auto.py \
        --model_name_or_path gpt3-13B-en \
        --tokenizer_name_or_path gpt3-13B-en \
        --input_dir "$gpt_data_path/data" \
        --output_dir "output/$task_name" \
        --split 949,50,1 \
        --max_seq_length 1024 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --sharding "" \
        --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 1 \
        --sequence_parallel 0 \
        --fuse_attention_qkv 0 \
        --use_flash_attention 0 \
        --scale_loss 1024 \
        --learning_rate 0.00001 \
        --min_learning_rate 0.000005 \
        --max_steps 10 \
        --save_steps 50000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1\
        --continue_training 0\
        --dataloader_num_workers 1 \
        --eval_steps 100000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --recompute 0 \
        --gradient_accumulation_steps 4 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --model_type "gpt" \
        --enable_auto_parallel 1 \
        --to_static 1 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        --num_hidden_layers 2 \
        --intermediate_size 1024 \
        >>${log_path}/$FUNCNAME 2>&1
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_gpt_pir_auto_bs4_TP2_PP2(){
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_enable_pir_api=1
    cd ${llm_gpt_case_path}

    task_name="gpt3_auto_bs4_tp2_pp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
        --log_dir $case_log_dir \
        run_pretrain_auto.py \
        --model_name_or_path gpt3-13B-en \
        --tokenizer_name_or_path gpt3-13B-en \
        --input_dir "$gpt_data_path/data" \
        --output_dir "output/$task_name" \
        --split 949,50,1 \
        --max_seq_length 1024 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --sharding "" \
        --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --sequence_parallel 0 \
        --fuse_attention_qkv 1 \
        --use_flash_attention 0 \
        --scale_loss 1024 \
        --learning_rate 0.00001 \
        --min_learning_rate 0.000005 \
        --max_steps 10 \
        --save_steps 50000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1\
        --continue_training 0\
        --dataloader_num_workers 1 \
        --eval_steps 100000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --recompute 0 \
        --gradient_accumulation_steps 4 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --model_type "gpt" \
        --enable_auto_parallel 1 \
        --to_static 1 \
        --fp16 1 \
        --fp16_opt_level "O2" \
        --num_hidden_layers 2 \
        --intermediate_size 1024 \
        >>${log_path}/$FUNCNAME 2>&1
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_gpt_pir_auto_bs8_DP2_TP2_PP2(){
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_enable_pir_api=1
    cd ${llm_gpt_case_path}

    task_name="gpt3_auto_bs8_dp2_tp2_pp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
        --log_dir $case_log_dir \
        run_pretrain_auto.py \
        --model_name_or_path gpt3-13B-en \
        --tokenizer_name_or_path gpt3-13B-en \
        --input_dir "$gpt_data_path/data" \
        --output_dir "output/$task_name" \
        --split 949,50,1 \
        --max_seq_length 1024 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --sharding "stage1" \
        --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --pipeline_schedule_mode "1F1B" \
        --sequence_parallel 0 \
        --fuse_attention_qkv 1 \
        --use_flash_attention 0 \
        --fused_linear_param_grad_add 1\
        --scale_loss 1024 \
        --learning_rate 0.00001 \
        --min_learning_rate 0.000005 \
        --max_steps 10 \
        --save_steps 50000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1\
        --continue_training 0\
        --dataloader_num_workers 1 \
        --eval_steps 100000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --recompute 0 \
        --gradient_accumulation_steps 4 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --model_type "gpt" \
        --enable_auto_parallel 1 \
        --to_static 1 \
        --fp16 1 \
        --fp16_opt_level "O2" \
        --num_hidden_layers 2 \
        --intermediate_size 1024 \
        --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
        --tensor_parallel_config "enable_mp_async_allreduce" \
        --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate" \
        --pipeline_parallel_config "enable_send_recv_overlap enable_split_backward" \
        >>${log_path}/$FUNCNAME 2>&1
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_qwen_dygraph_auto_bs1_fp32_DP2() {
    set -x

    config_json="pretrain_argument_for_ci_auto_dp2.json"

    cat <<EOF >"$config_json"
{
    "model_name_or_path": "qwen/qwen-7b",
    "tokenizer_name_or_path": "qwen/qwen-7b",
    "input_dir": "./data",
    "output_dir": "./checkpoints/qwen_pretrain_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 16,
    "tensor_parallel_degree": 1,
    "pipeline_parallel_degree": 1,
    "virtual_pp_degree": 1,
    "sequence_parallel": 0,   
    "use_flash_attention": false,
    "use_fused_rms_norm": false,
    "use_fused_rope": false,
    "max_seq_length": 4096,
    "learning_rate": 3e-05,
    "num_hidden_layers": 8,
    "min_learning_rate": 3e-06,
    "scale_loss": 1024,
    "warmup_steps": 30,
    "logging_steps": 1,
    "max_steps": 12,
    "save_steps": 1000,
    "eval_steps": 10000,
    "weight_decay": 0.01,
    "bf16": false,
    "fp16_opt_level": "O0",
    "warmup_ratio": 0.01,
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 1,
    "continue_training": 0,
    "do_train": true,
    "do_eval": false,
    "do_predict": false,
    "disable_tqdm": true,
    "recompute": true,
    "recompute_granularity": "core_attn",
    "recompute_use_reentrant": true,
    "distributed_dataloader": 0,
    "save_total_limit": 2,
    "enable_auto_parallel": 1,
    "to_static": 0
}
EOF

    unset CUDA_VISIBLE_DEVICES

    export FLAGS_call_stack_level=3
    export FLAGS_use_cuda_managed_memory=true

    task_name="llama_auto_dp2"
    case_log_dir="qwen_auto_3d_fp32_dp2"
    rm -rf output/$task_name/
    rm -rf "output/$task_name""_log"

    export SOT_LOG_LEVEL=4
    export PYTHONPATH=../../../:$PYTHONPATH


    rm -rf $case_log_dir

    export FLAGS_embedding_deterministic=1        
    export FLAGS_cudnn_deterministic=1
    export NVIDIA_TF32_OVERRIDE=0

    python -u  -m paddle.distributed.launch \
            --gpus "0,1" \
            --log_dir "$case_log_dir" \
        run_pretrain_3D_auto.py ./$config_json \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem loss_md5=$loss_md5"
    loss_base=9.83757591
    ips_base=-1
    mem_base=-1
    if [ $IS_A100 -ne 0 ];then
        check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    else
        echo "qwen auto just compare loss in A100 machine."
    fi
    rm -f $config_json
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_qwen_dygraph_auto_bs1_fp32_DP2-MP2() {
    set -x

    config_json="pretrain_argument_for_ci_auto_dp2_mp2.json"

    cat <<EOF >"$config_json"
{
    "model_name_or_path": "qwen/qwen-7b",
    "tokenizer_name_or_path": "qwen/qwen-7b",
    "input_dir": "./data",
    "output_dir": "./checkpoints/qwen_pretrain_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 16,
    "tensor_parallel_degree": 2,
    "pipeline_parallel_degree": 1,
    "virtual_pp_degree": 1,
    "sequence_parallel": 0,   
    "use_flash_attention": false,
    "use_fused_rms_norm": false,
    "use_fused_rope": false,
    "max_seq_length": 4096,
    "learning_rate": 3e-05,
    "num_hidden_layers": 8,
    "min_learning_rate": 3e-06,
    "scale_loss": 1024,
    "warmup_steps": 30,
    "logging_steps": 1,
    "max_steps": 12,
    "save_steps": 1000,
    "eval_steps": 10000,
    "weight_decay": 0.01,
    "bf16": false,
    "fp16_opt_level": "O0",
    "warmup_ratio": 0.01,
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 1,
    "continue_training": 0,
    "do_train": true,
    "do_eval": false,
    "do_predict": false,
    "disable_tqdm": true,
    "recompute": true,
    "recompute_granularity": "core_attn",
    "recompute_use_reentrant": true,
    "distributed_dataloader": 0,
    "save_total_limit": 2,
    "enable_auto_parallel": 1,
    "to_static": 0
}
EOF

    set -x
    unset CUDA_VISIBLE_DEVICES

    export FLAGS_call_stack_level=3
    export FLAGS_use_cuda_managed_memory=true

    task_name="llama_auto_dp2_mp2"
    case_log_dir="qwen_auto_3d_fp32_dp2_mp2"
    rm -rf output/$task_name/
    rm -rf "output/$task_name""_log"

    export SOT_LOG_LEVEL=4
    export PYTHONPATH=../../../:$PYTHONPATH

    rm -rf $case_log_dir

    export FLAGS_embedding_deterministic=1        
    export FLAGS_cudnn_deterministic=1
    export NVIDIA_TF32_OVERRIDE=0

    python -u  -m paddle.distributed.launch \
            --gpus "0,1,2,3" \
            --log_dir "$case_log_dir" \
        run_pretrain_3D_auto.py $config_json \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem loss_md5=$loss_md5"
    loss_base=9.83757591
    ips_base=-1
    mem_base=-1
    if [ $IS_A100 -ne 0 ];then
        check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    else
        echo "qwen auto just compare loss in A100 machine."
    fi
    rm -f $config_json
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_qwen_dygraph_auto_bs1_fp32_DP2-MP2-PP2() {
    set -x

    config_json="pretrain_argument_for_ci_auto_dp2_mp2_pp2.json"

    cat <<EOF >"$config_json"
{
    "model_name_or_path": "qwen/qwen-7b",
    "tokenizer_name_or_path": "qwen/qwen-7b",
    "input_dir": "./data",
    "output_dir": "./checkpoints/qwen_pretrain_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 16,
    "tensor_parallel_degree": 2,
    "pipeline_parallel_degree": 2,
    "virtual_pp_degree": 1,
    "sequence_parallel": 0,   
    "use_flash_attention": false,
    "use_fused_rms_norm": false,
    "use_fused_rope": false,
    "max_seq_length": 4096,
    "learning_rate": 3e-05,
    "num_hidden_layers": 8,
    "min_learning_rate": 3e-06,
    "scale_loss": 1024,
    "warmup_steps": 30,
    "logging_steps": 1,
    "max_steps": 12,
    "save_steps": 1000,
    "eval_steps": 10000,
    "weight_decay": 0.01,
    "bf16": false,
    "fp16_opt_level": "O0",
    "warmup_ratio": 0.01,
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 1,
    "continue_training": 0,
    "do_train": true,
    "do_eval": false,
    "do_predict": false,
    "disable_tqdm": true,
    "recompute": true,
    "recompute_granularity": "core_attn",
    "recompute_use_reentrant": true,
    "distributed_dataloader": 0,
    "save_total_limit": 2,
    "enable_auto_parallel": 1,
    "to_static": 0
}
EOF

    unset CUDA_VISIBLE_DEVICES

    export FLAGS_call_stack_level=3
    export FLAGS_use_cuda_managed_memory=true

    task_name="llama_auto_dp2_mp2_pp2"
    case_log_dir="qwen_auto_3d_fp32_dp2_mp2_pp2"
    rm -rf output/$task_name/
    rm -rf "output/$task_name""_log"

    export SOT_LOG_LEVEL=4
    export PYTHONPATH=../../../:$PYTHONPATH


    rm -rf $case_log_dir

    export FLAGS_embedding_deterministic=1
    export FLAGS_cudnn_deterministic=1
    export NVIDIA_TF32_OVERRIDE=0

    python -u  -m paddle.distributed.launch \
            --gpus "0,1,2,3,4,5,6,7" \
            --log_dir "$case_log_dir" \
        run_pretrain_3D_auto.py $config_json \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem loss_md5=$loss_md5"
    loss_base=9.83757591
    ips_base=-1
    mem_base=-1
    if [ $IS_A100 -ne 0 ];then
        check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    else
        echo "qwen auto just compare loss in A100 machine."
    fi
    rm -f $config_json
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_qwen_dygraph_auto_bs1_bf16_DP2-MP2-PP2() {
    set -x

    config_json="pretrain_argument_for_ci_auto_dp2_mp2_pp2.json"

    cat <<EOF >"$config_json"
{
    "model_name_or_path": "qwen/qwen-7b",
    "tokenizer_name_or_path": "qwen/qwen-7b",
    "input_dir": "./data",
    "output_dir": "./checkpoints/qwen_pretrain_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 16,
    "tensor_parallel_degree": 2,
    "pipeline_parallel_degree": 2,
    "virtual_pp_degree": 1,
    "sequence_parallel": 0,   
    "use_flash_attention": false,
    "use_fused_rms_norm": false,
    "use_fused_rope": false,
    "max_seq_length": 4096,
    "learning_rate": 3e-05,
    "num_hidden_layers": 8,
    "min_learning_rate": 3e-06,
    "scale_loss": 1024,
    "warmup_steps": 30,
    "logging_steps": 1,
    "max_steps": 12,
    "save_steps": 1000,
    "eval_steps": 10000,
    "weight_decay": 0.01,
    "bf16": true,
    "fp16_opt_level": "O2",
    "warmup_ratio": 0.01,
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 1,
    "continue_training": 0,
    "do_train": true,
    "do_eval": false,
    "do_predict": false,
    "disable_tqdm": true,
    "recompute": true,
    "recompute_granularity": "core_attn",
    "recompute_use_reentrant": true,
    "distributed_dataloader": 0,
    "save_total_limit": 2,
    "enable_auto_parallel": 1,
    "to_static": 0
}
EOF

    unset CUDA_VISIBLE_DEVICES

    export FLAGS_call_stack_level=3
    export FLAGS_use_cuda_managed_memory=true

    task_name="llama_auto_dp2_mp2_pp2"
    case_log_dir="qwen_auto_3d_bf16_dp2_mp2_pp2"
    rm -rf output/$task_name/
    rm -rf "output/$task_name""_log"

    export SOT_LOG_LEVEL=4
    export PYTHONPATH=../../../:$PYTHONPATH


    rm -rf $case_log_dir

    export FLAGS_embedding_deterministic=1
    export FLAGS_cudnn_deterministic=1
    export NVIDIA_TF32_OVERRIDE=0

    python -u  -m paddle.distributed.launch \
            --gpus "0,1,2,3,4,5,6,7" \
            --log_dir "$case_log_dir" \
        run_pretrain_3D_auto.py $config_json \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem loss_md5=$loss_md5"
    loss_base=9.88092232
    ips_base=-1
    mem_base=-1
    if [ $IS_A100 -ne 0 ];then
        check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    else
        echo "qwen auto just compare loss in A100 machine."
    fi
    rm -f $config_json
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_qwen_pir_auto_bs1_bf16_TP2(){
    echo "=========== $FUNCNAME run  begin ==========="

    set -x
    unset CUDA_VISIBLE_DEVICES

    export FLAGS_call_stack_level=3

    task_name="llama_auto_tp2"
    case_log_dir="qwen_auto_pir_bf16_tp2"
    rm -rf output/$task_name/
    rm -rf "output/$task_name""_log"

    export SOT_LOG_LEVEL=4
    export PYTHONPATH=../../../:$PYTHONPATH


    rm -rf $case_log_dir

    export FLAGS_embedding_deterministic=1
    export FLAGS_cudnn_deterministic=1
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_enable_pir_in_executor=1
    export FLAGS_enable_pir_api=1

    python -u  -m paddle.distributed.launch \
        --gpus "0,1" \
        --log_dir "$case_log_dir" \
        run_pretrain_3D_auto.py \
        --model_name_or_path "qwen/qwen-14b" \
        --tokenizer_name_or_path "qwen/qwen-14b" \
        --input_dir "./data" \
        --output_dir "output/$task_name/" \
        --per_device_train_batch_size 1\
        --gradient_accumulation_steps 2\
        --per_device_eval_batch_size 16\
        --sharding "stage1" \
        --sharding_parallel_degree 1\
        --tensor_parallel_degree 2\
        --pipeline_parallel_degree 1\
        --pipeline_schedule_mode "VPP" \
        --virtual_pipeline_seg_method 'QWenBlockAuto' \
        --virtual_pp_degree 2\
        --use_flash_attention true\
        --use_fused_rms_norm false\
        --use_fused_rope true\
        --max_seq_length 4096\
        --learning_rate 3e-05\
        --min_learning_rate 3e-06\
        --scale_loss 1024\
        --warmup_steps 30\
        --logging_steps 1\
        --max_steps 10\
        --save_steps 1000\
        --eval_steps 10000\
        --weight_decay 0.01\
        --bf16 true\
        --fp16_opt_level "O2"\
        --amp_master_grad true \
        --warmup_ratio 0.01\
        --max_grad_norm 0.0\
        --dataloader_num_workers 4\
        --continue_training 0\
        --do_train true\
        --do_eval false\
        --do_predict false \
        --disable_tqdm true\
        --recompute false\
        --recompute_granularity "core_attn"\
        --recompute_use_reentrant true\
        --distributed_dataloader 0\
        --save_total_limit 2\
        --enable_auto_parallel 1\
        --to_static 1 \
        --num_hidden_layers 4 \
        >>${log_path}/$FUNCNAME 2>&1
    echo "=========== $FUNCNAME run  end ==========="
}

function llm_qwen_pir_auto_bs1_bf16_TP2_PP2(){
    echo "=========== $FUNCNAME run  begin ==========="

    set -x
    unset CUDA_VISIBLE_DEVICES

    export FLAGS_call_stack_level=3

    task_name="llama_auto_tp2_pp2"
    case_log_dir="qwen_auto_pir_bf16_tp2_pp2"
    rm -rf output/$task_name/
    rm -rf "output/$task_name""_log"

    export SOT_LOG_LEVEL=4
    export PYTHONPATH=../../../:$PYTHONPATH


    rm -rf $case_log_dir

    export FLAGS_embedding_deterministic=1
    export FLAGS_cudnn_deterministic=1
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_enable_pir_in_executor=1
    export FLAGS_enable_pir_api=1

    python -u  -m paddle.distributed.launch \
        --gpus "0,1,2,3" \
        --log_dir "$case_log_dir" \
        run_pretrain_3D_auto.py \
        --model_name_or_path "qwen/qwen-14b" \
        --tokenizer_name_or_path "qwen/qwen-14b" \
        --input_dir "./data" \
        --output_dir "output/$task_name/" \
        --per_device_train_batch_size 1\
        --gradient_accumulation_steps 4\
        --per_device_eval_batch_size 16\
        --sharding "stage1" \
        --sharding_parallel_degree 1\
        --tensor_parallel_degree 2\
        --pipeline_parallel_degree 2\
        --pipeline_schedule_mode "1F1B" \
        --use_flash_attention true\
        --use_fused_rms_norm false\
        --use_fused_rope true\
        --max_seq_length 4096\
        --learning_rate 3e-05\
        --min_learning_rate 3e-06\
        --scale_loss 1024\
        --warmup_steps 30\
        --logging_steps 1\
        --max_steps 10\
        --save_steps 1000\
        --eval_steps 10000\
        --weight_decay 0.01\
        --bf16 true\
        --fp16_opt_level "O2"\
        --amp_master_grad true \
        --warmup_ratio 0.01\
        --max_grad_norm 0.0\
        --dataloader_num_workers 4\
        --continue_training 0\
        --do_train true\
        --do_eval false\
        --do_predict false \
        --disable_tqdm true\
        --recompute false\
        --recompute_granularity "core_attn"\
        --recompute_use_reentrant true\
        --distributed_dataloader 0\
        --save_total_limit 2\
        --enable_auto_parallel 1\
        --to_static 1 \
        --num_hidden_layers 4 \
        >>${log_path}/$FUNCNAME 2>&1
    echo "=========== $FUNCNAME run  end ==========="
}


############ case end ############

function check_md5_result() {
    echo -e "$1" >> ${log_path}/result.log

    if [ $# -ne 3 ]; then
        echo -e "\033[31m $1 parameter transfer failed: $@ \033[0m" | tee -a ${log_path}/result.log
        exit -1
    fi

    echo -e "loss_md5_base: $2 loss_md5: $3" | tee -a ${log_path}/result.log
    if [ $2 != $3 ];then
        echo -e "\033[31m $1 loss_md5 diff check failed! \033[0m" | tee -a ${log_path}/result.log
        exit -1
    fi
}

function check_result() {
    echo -e "$1" >> ${log_path}/result.log
    if [ $? -ne 0 ];then
        echo -e "\033[31m $1 run failed! \033[0m" | tee -a ${log_path}/result.log
        exit 2
    fi

    if [ $# -ne 7 ] && [ $# -ne 8 ]; then
        echo -e "\033[31m $1 parameter transfer failed: $@ \033[0m" | tee -a ${log_path}/result.log
        exit 2
    fi

    diff_loss=$(echo $2 $3|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
    echo -e "loss_base: $2 loss_test: $3 loss_diff: $diff_loss%" | tee -a ${log_path}/result.log
    if [ $2 != $3 ];then
        if [ -z "$8" ] || [ $8 -ne 1 ] ;then
            echo -e "\033[31m $1 loss diff check failed! \033[0m" | tee -a ${log_path}/result.log
            exit 2
        else
            diff=$(echo "$2 $3" | awk '{print $1-$2}')
            gt=$(echo "${diff#-} 1e-5" | awk '{print ($1>$2)?"1":"0"}')
            if [ $gt -eq 1 ];then
                echo -e "\033[31m $1 loss diff check failed! \033[0m" | tee -a ${log_path}/result.log
                exit 2
            fi
        fi
    fi

    diff_ips=$(echo $4 $5|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
    echo -e "ips_base: $4 ips_test: $5 ips_diff: $diff_ips% " | tee -a $log_path/result.log
    v1=$(echo $diff_ips 5.0|awk '{print($1>=$2)?"0":"1"}')
    v2=$(echo $diff_ips -5.0|awk '{print($1<=$2)?"0":"1"}')
    if [[ $v1 == 0 ]];then
        echo -e "$1 IPS increase greater than 5%, not exit " | tee -a $log_path/result.log
    fi
    if [[ $v2 == 0 ]];then
        echo -e "\033[31m $1 IPS diff check failed! \033[0m" | tee -a $log_path/result.log
        exit 2
    fi

    diff_mem=$(echo $6 $7|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
    echo -e "mem_base: $6 mem_test: $7 mem_diff: $diff_mem% " | tee -a $log_path/result.log
    w1=$(echo $diff_mem 5.0|awk '{print($1>=$2)?"0":"1"}')
    w2=$(echo $diff_mem -5.0|awk '{print($1<=$2)?"0":"1"}')
    if [[ $w1 == 0 ]];then
        echo -e "\033[31m $1 MEM diff check failed! \033[0m" | tee -a $log_path/result.log
        exit 2
    fi
    if [[ $w2 == 0 ]];then
        echo -e "$1 MEM decreases greater than 5%, not exit " | tee -a $log_path/result.log
    fi
}

function before_hook_for_gpt() {
    echo -e "\033[31m ---- Set FLAGS for GPT auto cases  \033[0m"
    cd ${gpt_case_path}
    export FLAGS_new_executor_micro_batching=True  # True：打开新执行器
    export FLAGS_embedding_deterministic=1         # 1：关闭随机性
    export FLAGS_cudnn_deterministic=1             # 1：关闭随机性
    unset CUDA_MODULE_LOADING
    env | grep FLAGS
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    export no_proxy=bcebos.com
    if [[ $FLAGS_install_deps == 0 ]];then
        echo -e "\033[31m ---- Install requirements for GPT auto cases  \033[0m"
        cp requirements.txt requirements_nlp.txt
        sed -i '/paddlenlp/d' requirements.txt
        python -m pip install -r requirements.txt --force-reinstall
        sed -i '/paddlenlp/!d' requirements_nlp.txt
        python -m pip install -r requirements_nlp.txt
        python -m pip install -r $root_path/requirements.txt
        python -m pip install -r $root_path/requirements-dev.txt
        python -m pip install --no-cache-dir https://paddlenlp.bj.bcebos.com/wheels/paddlenlp-ci-py3-none-any.whl --force-reinstall --no-dependencies
        python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)";
    else
        echo -e "\033[31m ---- Skip install requirements for GPT auto cases  \033[0m"
    fi
    unset http_proxy && unset https_proxy
    if [[ ! $FLAGS_download_data =~ "gpt" ]];then
        echo -e "\033[31m ---- Download GPT data  \033[0m"
        rm -rf data
        if [[ -e ${gpt_data_path}/data ]]; then
            echo "GPT data downloaded"
        else
            # download data for gpt
            mkdir -p ${gpt_data_path}/data;
            wget -O ${gpt_data_path}/data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy;
            wget -O ${gpt_data_path}/data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz;
        fi
        cp -r ${gpt_data_path}/data ${gpt_case_path}/
    else
        echo -e "\033[31m ---- Skip download gpt data \033[0m"
    fi
}

function export_env() {
    export FLAGS_new_executor_micro_batching=True  # True：打开新执行器
    export FLAGS_embedding_deterministic=1         # 1：关闭随机性
    export FLAGS_cudnn_deterministic=1             # 1：关闭随机性
    export FLAGS_program_topo_reorder=1            # 1: 反向对齐动手拓扑排序
    unset CUDA_MODULE_LOADING
    env | grep FLAGS
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    export no_proxy=bcebos.com
}

function before_hook_for_llama() {
    echo -e "\033[31m ---- Set FLAGS for LLaMA auto cases  \033[0m"
    cd ${llama_case_path}
    export FLAGS_new_executor_micro_batching=True  # True：打开新执行器
    export FLAGS_embedding_deterministic=1         # 1：关闭随机性
    export FLAGS_cudnn_deterministic=1             # 1：关闭随机性
    export FLAGS_program_topo_reorder=1            # 1: 反向对齐动手拓扑排序
    unset CUDA_MODULE_LOADING
    env | grep FLAGS
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    export no_proxy=bcebos.com
    python -m pip install -r $root_path/requirements.txt
    python -m pip install -r $root_path/requirements-dev.txt
    unset http_proxy && unset https_proxy
    if [[ ! $FLAGS_download_data =~ "llama" ]];then
        echo -e "\033[31m ---- Download LLaMA data  \033[0m"
        rm -rf data
        if [[ -e ${llama_data_path}/data ]]; then
            echo "LLaMA data downloaded"
        else
            # download data for llama
            mkdir ${llama_data_path}/data;
            wget -O ${llama_data_path}/data/llama_openwebtext_100k_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy;
            wget -O ${llama_data_path}/data/llama_openwebtext_100k_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz;
        fi
        cp -r ${llama_data_path}/data ${llama_case_path}/
    else
        echo -e "\033[31m ---- Skip download LLaMA data \033[0m"
    fi
}




export status=$1
if [[ $status = "prepare_case" ]];then
    export FLAGS_install_deps=$3
    export FLAGS_download_data=$4
    if [[ $2 = "llama_case_list_auto" ]];then
        before_hook_for_llama 
        llama_case_list_auto prepare_case
    elif [[ $2 = "llm_gpt_case_list_auto" ]];then
        before_hook_for_gpt
        llm_gpt_case_list_auto prepare_case
    else
        echo -e "\033[31m ---- Invalid exec_case $2 \033[0m"
    fi
elif [[ $status = "exec_case" ]];then
    export FLAGS_install_deps=$3
    export FLAGS_download_data=$4
    export_env
    if [[ $2 =~ "gpt" ]];then
        cd ${gpt_case_path}
    elif [[ $2 =~ "llama" ]];then
        cd ${llama_case_path}
    fi
    $2
else
    echo -e "\033[31m ---- Start executing  $status \033[0m"
    export exec_case=$1
    export FLAGS_install_deps=$2
    export FLAGS_download_data=$3
    if [[ $status =~ "gpt" ]];then
        cd ${gpt_case_path}
        before_hook_for_gpt
    elif [[ $status =~ "llama" ]];then
        cd ${llama_case_path}
        before_hook_for_llama
    else
        echo -e "\033[31m ---- Invalid exec_case $exec_case \033[0m"
    fi
    $1 exec_case
fi
