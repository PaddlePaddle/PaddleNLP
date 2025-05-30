NUM_NODES=1
NUM_GPUS_PER_NODE=8
BACKEND=nccl

MPI_PATH=/usr/local/mpi/
START_MB=16
END_MB=256
SCALE=2
HOSTFILE="hostfile"
export NCCLTEST_OTHER_ARGS="-x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_2,mlx5_5"

PROFILE_ARGS="
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
    --backend ${BACKEND} \
    --mpi_path ${MPI_PATH} \
    --start_mb ${START_MB} \
    --end_mb ${END_MB} \
    --scale ${SCALE} \
    --hostfile ${HOSTFILE} \
    --avg_or_min_or_first first \
    --max_pp_deg 16 \
"
python3 profile_hardware.py ${PROFILE_ARGS}