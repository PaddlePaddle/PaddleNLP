model_item=gpt3_moe
dp_degree=1
sharding_degree=1
bs_item=8
fp_item=fp16
run_mode=DP_MoE_C1
device_num=N1C1

model=gpt

cd ./tests
bash ./test_tipc/dygraph/moe/${model}/benchmark_common/prepare.sh
# run
bash ./test_tipc/dygraph/moe/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${sharding_degree} ${bs_item} ${run_mode} ${device_num} 2>&1;
