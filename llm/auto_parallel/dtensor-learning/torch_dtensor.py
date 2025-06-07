import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')  # 使用NCCL后端（GPU专用）

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # all_data_list = [[i + 1 for _ in range(16)] for i in range(world_size)]
    all_data_list = [rank for _ in range(16)]
    all_data_tensor = torch.tensor(all_data_list, dtype=torch.float32, device=f'cuda:{rank}')
    
    mesh1 = DeviceMesh(device_type='cuda', mesh=[[0, 1], [2, 3]], mesh_dim_names=['dp', 'tp'])
    placements = [Replicate(), Shard(0)]
    # dtensor = distribute_tensor(all_data_tensor, mesh1, placements)
    dtensor = DTensor.from_local(all_data_tensor, mesh1, placements)
    print(f"Rank {rank} DTensor: {dtensor}")
    
    if rank == 0:   
        print("=================================================") 
    mesh2 = DeviceMesh(device_type='cuda', mesh=[[4, 5], [6, 7]], mesh_dim_names=['dp', 'tp'])
    dtensor2 = dtensor.redistribute(mesh2, placements)
    print(f"Rank {rank} Redistributed DTensor: {dtensor2}")
    
    if rank == 0:
        print("=================================================")
    mesh3 = DeviceMesh(device_type='cuda', mesh=[[0, 1, 2, 3]], mesh_dim_names=['dp', 'tp'])    
    dtensor3 = dtensor.redistribute(mesh3, placements=[Replicate(), Replicate()])
    print(f"Rank {rank} Redistributed DTensor to mesh3: {dtensor3}")
    
    dist.destroy_process_group()