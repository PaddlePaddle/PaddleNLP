model_item=gpt_auto
dp_degree=1
mp_degree=8
pp_degree=2
bs_item=16 # micro * dp * pp
fp_item=fp32
run_mode=DP1-MP8-PP2-SD1-stage1
device_num=N2C16
sharding_degree=1
sharding_stage=1
local_batch_size=16

model=gpt
micro_bs=8 # local_batch_size / pp_degree

cd ./benchmarks
bash ./test_tipc/gpt/static/new_exec_pp/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/static/new_exec_pp/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${sharding_degree} ${sharding_stage} 2>&1;
