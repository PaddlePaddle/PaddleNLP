model_item=llama-7b_auto_dp2mp2pp2
bs_item=1
fp_item=fp16
run_mode=DP
device_num=N1C1

max_iter=100

# prepare
bash ./test_tipc/dygraph/${model_item}/benchmark_common/prepare.sh
# run
bash ./test_tipc/dygraph/${model_item}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} 2>&1;
