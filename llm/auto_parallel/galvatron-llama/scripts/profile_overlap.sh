export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5
export NCCL_IB_DISABLE=0
echo "Running /apdcephfs_fsgm/share_303760348/anaconda3/envs/lgm-paddle/bin/python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 --log_dir output/profile_overlap /apdcephfs_fsgm/share_303760348/guangming/WorkSpace/paddle3.0/paddlenlp/experimental/galvatron/profiler/profile_overlap.py --output_dir "./output""
/apdcephfs_fsgm/share_303760348/anaconda3/envs/lgm-paddle/bin/python3 -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 --log_dir output/profile_overlap /apdcephfs_fsgm/share_303760348/guangming/WorkSpace/paddle3.0/paddlenlp/experimental/galvatron/profiler/profile_overlap.py --output_dir "./output"
sleep 1
rm -r ./profiler_log