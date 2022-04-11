model_item=transformer_base
model=transformer
bs_item=4096
fp_item=fp32
run_mode=DP
device_num=N1C1
max_epochs=500
num_workers=0

# get data
bash ./test_tipc/static/${model}/benchmark_common/prepare.sh
# run
bash ./test_tipc/static/${model}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_epochs} ${num_workers} 2>&1;
