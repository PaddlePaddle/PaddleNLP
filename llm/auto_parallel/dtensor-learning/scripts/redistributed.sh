set -x
unset CUDA_VISIBLE_DEVICES

task_name="redistributed"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

TRAINER="./redistributed.py"
LAUNCHER="python -u -m paddle.distributed.launch"
LAUNCHER="${LAUNCHER} --gpus 0,1,2,3"  # 设置需要使用的GPU
LAUNCHER="${LAUNCHER} --log_dir output/$task_name""_log ${TRAINER} --output_dir "./output""

$LAUNCHER