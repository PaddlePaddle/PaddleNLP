model_item=gpt3
model=gpt
fp_item=fp16
mp_degree=1
pp_degree=1
dp_degree=1
micro_batch_size=16
global_batch_size=16
run_mode=DP1-MP1-PP1
device_num=N1C1
max_iter=10000500
use_sharding=false

# run
bash ./test_tipc/static/hybrid_parallelism/${model}/benchmark_common/prepare.sh
bash ./test_tipc/static/hybrid_parallelism/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${mp_degree} ${pp_degree} ${dp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${max_iter} ${use_sharding} 2>&1;

sleep 10
export PROFILING=true
bash ./test_tipc/static/hybrid_parallelism/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${mp_degree} ${pp_degree} ${dp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${max_iter} ${use_sharding} 2>&1;
