model_item=gpt3
model=gpt
fp_item=fp16
mp_degree=4
pp_degree=1
dp_degree=1
micro_batch_size=32
global_batch_size=32
run_mode=DP1-MP4-PP1
device_num=N1C4
max_iter=3

# run
bash ./test_tipc/static/hybrid_parallelism/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${mp_degree} ${pp_degree} ${dp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${max_iter} 2>&1;
