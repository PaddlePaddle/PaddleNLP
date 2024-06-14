modelname="transformer"
export FLAGS_profile_pipeline=1
# HTTP
ps -ef | grep web_service | awk '{print $2}' | xargs kill -9
sleep 3
rm -rf profile_log_$modelname
for thread_num in "1" "8" "16"; do
  for batch_size in "1" "2" "4"; do
    python transformer_web_server.py --config ../../configs/transformer.base.yaml --device gpu --model_dir ./transformer_server --profile &
    sleep 3
    echo "----Transformer thread num: ${thread_num} batch size: ${batch_size} mode:http ----" >> profile_log_$modelname
    nvidia-smi --id=2 --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    nvidia-smi --id=2 --query-gpu=utilization.gpu --format=csv -lms 100 > gpu_utilization.log 2>&1 &
    echo "import psutil\ncpu_utilization=psutil.cpu_percent(1,False)\nprint('CPU_UTILIZATION:', cpu_utilization)\n" > cpu_utilization.py
    python transformer_web_client.py --config ../../configs/transformer.base.yaml --batch_size ${batch_size} --threads ${thread_num} --profile
    python cpu_utilization.py >> profile_log_$modelname
    ps -ef | grep web_server | awk '{print $2}' | xargs kill -9
    python benchmark.py benchmark.log benchmark.tmp
    mv benchmark.tmp benchmark.log
    awk 'BEGIN {max = 0} {if(NR>1){if ($modelname > max) max=$modelname}} END {print "MAX_GPU_MEMORY:", max}' gpu_use.log >> profile_log_$modelname
    awk 'BEGIN {max = 0} {if(NR>1){if ($modelname > max) max=$modelname}} END {print "GPU_UTILIZATION:", max}' gpu_utilization.log >> profile_log_$modelname
    cat benchmark.log >> profile_log_$modelname
  done
done
