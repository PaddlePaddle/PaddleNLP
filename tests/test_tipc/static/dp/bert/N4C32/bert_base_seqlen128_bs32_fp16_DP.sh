model_item=bert_base_seqlen128
model=bert
bs_item=32
fp_item=fp16
run_mode=DP
device_num=N4C32
max_epochs=1000
num_workers=0

# get data
bash ./test_tipc/static/dp/${model}/benchmark_common/prepare.sh
# run
bash ./test_tipc/static/dp/${model}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_epochs} ${num_workers} 2>&1;
