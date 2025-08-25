interpreter="/apdcephfs_fsgm/share_303760348/anaconda3/envs/lgm-paddle/bin/python3"
launch="${interpreter} -u -m paddle.distributed.launch"
launch="${launch} --ips 28.12.131.41"
launch="${launch} --gpus 0,1,2,3,4,5,6,7"

export INTERPRETER=${interpreter}
export LAUNCHER=${launch}
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5
export NCCL_IB_DISABLE=0

PROFILE_HARDWARE_ARGS=(
    --num_nodes 1
    --num_gpus_per_node 8
    --backend 'paddle'
    --max_pp_deg 8
    --max_tp_deg 8
)

${interpreter} profile_hardware.py \
    "${PROFILE_HARDWARE_ARGS[@]}" \