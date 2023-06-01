model_item=gpt_auto_stage2
dp_degree=1
mp_degree=1
pp_degree=8
bs_item=64
fp_item=o1
run_mode=DP1-MP1-PP8
device_num=N1C8
sharding_degree=1
sharding_stage=2

model=gpt
micro_bs=8

cd ./benchmarks
bash ./test_tipc/gpt/static/auto_amp/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/static/auto_amp/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${sharding_degree} ${sharding_stage} 2>&1;
