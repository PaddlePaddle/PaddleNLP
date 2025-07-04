set -x
unset CUDA_VISIBLE_DEVICES

task_name="pp"
rm -rf $task_name/
rm -rf "$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

TRAINER="./pp.py"
LAUNCHER="python -u -m paddle.distributed.launch"
LAUNCHER="${LAUNCHER} --gpus 0,1,2,3,4,5,6,7"  # 设置需要使用的GPU
LAUNCHER="${LAUNCHER} --log_dir $task_name""_log ${TRAINER}"

$LAUNCHER