if [ "$USE_EXPORT_VARIABLE" = "1" ]; then
echo "USE_EXPORT_VARIABLE is set to 1, using the exported variables."
else
echo "USE_EXPORT_VARIABLE is not set to 1, using the variables defined in script."
NUM_NODES=2
NUM_GPUS_PER_NODE=8
START_MB=32
END_MB=512
SCALE=2
NCCLTEST_FILE="../site_package/nccl-tests/build/all_reduce_perf"
HOSTNAMES="job-83e1033f-9636-44b3-bf8b-2b627707b95f-master-0,\
job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0"
NCCLTEST_OTHER_ARGS="-x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_2,mlx5_5"
DEVICES="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
OUTPUT_TO_LOG=0
fi

if [ "$NUM_NODES" = "1" ]; then
CMD="${NCCLTEST_FILE} \
-b ${START_MB}M \
-e ${END_MB}M \
-f ${SCALE} \
-g ${NUM_GPUS_PER_NODE}"
export ${DEVICES}
echo "Running: ${DEVICES} ${CMD}"
OUTPUT_TO_LOG=1
else
CMD="mpirun --allow-run-as-root \
-H ${HOSTNAMES} \
-np ${NUM_NODES} \
--output-filename nccl_log \
${NCCLTEST_OTHER_ARGS} \
-x ${DEVICES} \
${NCCLTEST_FILE} \
-b ${START_MB}M \
-e ${END_MB}M \
-f ${SCALE} \
-g ${NUM_GPUS_PER_NODE}"
OUTPUT_TO_LOG=0
echo "Running: ${CMD}"
fi

if [ "$OUTPUT_TO_LOG" = "1" ]; then
mkdir -p nccl_log/1/rank.0/
OUTPUT_FILE="nccl_log/1/rank.0/stdout"
${CMD} 1> ${OUTPUT_FILE} 2>&1
else
${CMD}
fi