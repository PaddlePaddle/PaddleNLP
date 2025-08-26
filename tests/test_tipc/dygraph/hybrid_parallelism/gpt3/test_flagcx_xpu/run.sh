# this is the script for training gpt3 on XPU machines using flagcx as communication backend
export root_path=/workspace
#export PYTHONPATH=$root_path/PaddleNLP:$PYTHONPATH
export PADDLE_DISTRI_BACKEND=flagcx

# log
export GLOG_v=0
export FLAGCX_DEBUG=INFO
export FLAGCX_DEBUG_SUBSYS=INIT
export XPU_FORCE_SHARED_DEVICE_CONTEXT=1
# export FLAGS_log_memory_stats=0
# export GLOG_minloglevel=3
# export TRANSLATOR_CODE_LEVEL=100

current_date=$(date +"%m%d")
task_name="gpt13b_dynamic_hand_nosp_ly4_debug_$current_date"
# task_name="gpt13b_dynamic_baseline_ly20_nosp_nofusedrop$current_date"
# task_name="gpt13b_dynamic_baseline_$current_date"
log_dir="log_$current_date/${task_name}_1"
output_dir="output_$current_date/${task_name}_1"

rm -rf ${log_dir}
rm -rf ${output_dir}


python -u -m paddle.distributed.launch \
    --xpus "0,1,2,3,4,5,6,7" \
    --log_dir ${log_dir} \
    run_pretrain.py \
    ${root_path}/PaddleNLP/tests/test_tipc/dygraph/hybrid_parallelism/gpt3/auto_config_gpt3_13b/pretrain-gpt3_13b-config.json

echo "---- $task_name performance:"
echo "throughput(tokens/s/card):"
cat ${log_dir}/workerlog.0 | grep "interval_tokens_per_second_per_device:" | awk -F ',' '{print $11}' | awk -F ' ' '{print $2}' | awk 'NR > 10 {print $1}' |sort -n | awk '{values[NR] = $1} END {for (i = 3; i <= NR-2; i++) sum += values[i]; print sum / (NR-4)}'

echo "max_memory_allocated(GB):"
cat ${log_dir}/workerlog.0 | grep "interval_tokens_per_second_per_device:" | awk -F ',' '{print $7}' | tail -n 1

echo "max_memory_reserved(GB):"
cat ${log_dir}/workerlog.0 | grep "interval_tokens_per_second_per_device:" | awk -F ',' '{print $8}' | tail -n 1