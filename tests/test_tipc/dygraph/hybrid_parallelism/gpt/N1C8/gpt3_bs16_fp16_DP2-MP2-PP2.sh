model_item=gpt3
dp_degree=2
mp_degree=2
pp_degree=2
bs_item=16
fp_item=fp16
run_mode=DP2-MP2-PP2
device_num=N1C8

model=gpt
micro_bs=8
# get data
cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/${model}/benchmark_common/prepare.sh
# run
bash ./test_tipc/dygraph/hybrid_parallelism/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} 2>&1;
