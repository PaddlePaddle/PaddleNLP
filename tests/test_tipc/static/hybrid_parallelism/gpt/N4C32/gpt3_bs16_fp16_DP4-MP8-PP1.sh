model_item=gpt3
model=gpt
fp_item=fp16
mp_degree=8
pp_degree=1
dp_degree=4
micro_batch_size=4
global_batch_size=16
run_mode=DP4-MP8-PP1
device_num=N4C32
max_iter=1000030
use_sharding=false

# run
bash ./test_tipc/static/hybrid_parallelism/${model}/benchmark_common/prepare.sh
bash ./test_tipc/static/hybrid_parallelism/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${mp_degree} ${pp_degree} ${dp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${max_iter} ${use_sharding} 2>&1;
