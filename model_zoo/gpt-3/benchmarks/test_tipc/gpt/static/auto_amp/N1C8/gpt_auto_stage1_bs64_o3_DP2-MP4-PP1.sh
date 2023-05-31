model_item=gpt_auto_stage1
dp_degree=2
mp_degree=4
pp_degree=1
bs_item=64
fp_item=o3
run_mode=DP2-MP4-PP1
device_num=N1C8
sharding_degree=2
sharding_stage=1

model=gpt
micro_bs=8

cd ./benchmarks
bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${sharding_degree} ${sharding_stage} 2>&1;
