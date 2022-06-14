model_item=gpt3
model=gpt
fp_item=fp16
mp_degree=8
pp_degree=2
dp_degree=2
micro_batch_size=4
global_batch_size=16
run_mode=DP2-MP8-PP2
device_num=N4C32
max_iter=1000030
use_sharding=true

# run
bash ./test_tipc/static/hybrid_parallelism/${model}/benchmark_common/prepare.sh
bash ./test_tipc/static/hybrid_parallelism/${model}/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${mp_degree} ${pp_degree} ${dp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${max_iter} ${use_sharding} 2>&1;
