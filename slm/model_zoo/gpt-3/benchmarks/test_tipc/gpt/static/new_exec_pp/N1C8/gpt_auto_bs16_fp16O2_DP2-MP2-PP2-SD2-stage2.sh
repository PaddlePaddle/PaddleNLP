model_item=gpt_auto
dp_degree=2
mp_degree=2
pp_degree=2
bs_item=16 # micro * dp * pp
fp_item=fp16O2
run_mode=DP2-MP2-PP2-SD2-stage2
device_num=N1C8
sharding_degree=2 # sharding_degree = dp_degree
sharding_stage=2
level=o2
local_batch_size=8

model=gpt
micro_bs=4 # local_batch_size / pp_degree

cd ./benchmarks
bash ./test_tipc/gpt/static/new_exec_pp/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/static/new_exec_pp/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${sharding_degree} ${sharding_stage} ${level} 2>&1;
