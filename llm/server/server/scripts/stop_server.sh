# /bin/bash

pids=($(ps aux | grep -E 'tritonserver' | grep -v grep | awk '{print $2}'))

if [ ${#pids[@]} -eq 0 ]; then
    echo "Can not find tritonserver."
    timeout=1
else
    timeout=300
fi

# kill processor
for pid in "${pids[@]}"; do
    echo "killing $pid"
    kill -2 "$pid"
done

timeout_interval=$1
if [ ! "$timeout_interval" == "" ]; then
    timeout=$timeout_interval
    echo $timeout
fi

start_time=$(date +%s)

while : ; do
  current_time=$(date +%s)

  elapsed_time=$((current_time - start_time))

  if [ $elapsed_time -ge $timeout ]; then
    echo "forcibly kill all process ..."
    pids=$(ps auxww | grep -E "tritonserver|triton_python_backend_stub|infer|multiprocessing.resource_tracker|paddle.distributed.launch|task_queue_manager|app.py|spawn_main" | grep -v grep | grep -v start_both | awk '{print $2}');
    echo $pids;
    for pid in ${pids[@]}; do
    kill -9 ${pid}
    done
    break
  fi

  pids=$(ps auxww | grep -E "tritonserver|triton_python_backend_stub|multiprocessing.resource_tracker|paddle.distributed.launch|app.py|spawn_main" | grep -v grep | awk '{print $2}');
  array=($(echo "$pids" | tr ' ' '\n'))

  if [ ${#array[*]} -ne 0 ]; then
    echo "cleaning process, please wait ..."
    sleep 1
  else
    echo "clean finished."
    break
  fi
done

manager_pids=$(ps auxww | grep "task_queue_manager" | grep -v grep | awk '{print $2}')
echo $manager_pids
for in_pid in ${manager_pids[@]}; do
    kill -9 ${in_pid}
done
echo 'end kill queue manager'

health_checker_pids=$(ps auxww | grep "health.py" | grep -v grep | awk '{print $2}')
echo $health_checker_pids
for in_pid in ${health_checker_pids[@]}; do
    kill -9 ${in_pid}
done
echo 'end kill health checker'

echo "all process terminated."
exit 0
