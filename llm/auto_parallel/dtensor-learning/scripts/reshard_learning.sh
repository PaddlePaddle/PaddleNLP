set -x
unset CUDA_VISIBLE_DEVICES

task_name="reshard_learning"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

TRAINER="./reshard_learning.py"
LAUNCHER="python -u -m paddle.distributed.launch"
LAUNCHER="${LAUNCHER} --gpus 0,1,2,3,4,5,6,7"  # 设置需要使用的GPU
LAUNCHER="${LAUNCHER} --log_dir output/$task_name""_log ${TRAINER} --output_dir "./output""

$LAUNCHER