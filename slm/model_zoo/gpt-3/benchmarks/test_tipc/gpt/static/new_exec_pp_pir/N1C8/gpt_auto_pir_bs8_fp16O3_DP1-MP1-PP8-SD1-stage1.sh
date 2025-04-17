model_item=gpt_auto_pir
dp_degree=1
mp_degree=1
pp_degree=8
bs_item=8 # micro * dp * pp
fp_item=fp16O3
run_mode=DP1-MP1-PP8-SD1-stage1
device_num=N1C8
sharding_degree=1
sharding_stage=1
level=o3
local_batch_size=8

model=gpt
micro_bs=1 # local_batch_size / pp_degree

cd ./benchmarks
bash ./test_tipc/gpt/static/new_exec_pp_pir/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/static/new_exec_pp_pir/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${sharding_degree} ${sharding_stage} ${level} 2>&1;
