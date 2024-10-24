model_item=gpt_auto
dp_degree=1
mp_degree=2
pp_degree=4
bs_item=8 # micro * dp * pp
fp_item=fp16O2
run_mode=DP1-MP2-PP4-SD1-stage1
device_num=N1C8
sharding_degree=1 # sharding_degree = dp_degree
sharding_stage=1
level=o2
local_batch_size=8

model=gpt
micro_bs=2 # local_batch_size / pp_degree

cd ./benchmarks
bash ./test_tipc/gpt/static/new_exec_pp/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/static/new_exec_pp/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${sharding_degree} ${sharding_stage} ${level} 2>&1;
