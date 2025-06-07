import copy
from paddle.autograd import PyLayer
from paddle import Tensor
import paddle.distributed as dist
from paddle.distributed import ProcessMesh
from paddle.distributed.auto_parallel.api import dtensor_from_local, dtensor_to_local, unshard_dtensor
from paddle.distributed.auto_parallel.static.reshard_funcs.nd_mesh_reshard_func import get_1D_sub_process_mesh

def split_batch_with_sequence_parallel(tensor: Tensor, next_mesh: ProcessMesh):
    """
        when dp_degree increases, 
        we need to split the tensor along the batch dimension, 
        and gather the tensor along the sequence dimension when the sequence parallel is enabled.
    """
    # print("entering split_batch_with_sequence_parallel")
    rank = dist.get_rank()
    # print(f'rank {rank}, tensor is {tensor}')
    # print(f'next_mesh is {next_mesh}')
    if rank not in tensor.process_mesh.process_ids:
        return tensor
    
    dp_dim, tp_dim, seq_dim, batch_dim = 0, 1, 0, 1
    
    # Step1: restore the sequence dimension
    dtensor = dist.reshard(tensor, tensor.process_mesh, [dist.Shard(batch_dim), dist.Replicate()]) 
    local_tensor = dtensor_to_local(dtensor, dtensor.process_mesh, dtensor.placements).contiguous() # restore the sequence dimension, local_tensor shape is [seq_len, batch_size, hidden_size]
    
    # Step2: find the tp groups and split the batch dimension    
    origin_mesh = tensor.process_mesh
    origin_local_shape = tensor._local_shape # [seq_len, batch_size, hidden_size]
    
    origin_tp_mesh = get_1D_sub_process_mesh(origin_mesh, tp_dim)
    origin_tp_process_ids = origin_tp_mesh.process_ids
    split_group_num = origin_mesh.shape[tp_dim] // next_mesh.shape[tp_dim]
    split_group_size = len(origin_tp_process_ids) // split_group_num
    new_tp_groups = [origin_tp_process_ids[i:i + split_group_size] for i in range(0, len(origin_tp_process_ids), split_group_size)]
    idx = next(i for i, group in enumerate(new_tp_groups) if rank in group)
    batch_length = origin_local_shape[batch_dim] // split_group_num
    local_tensor = local_tensor[:, idx * batch_length : (idx + 1) * batch_length, :].contiguous()
    
    # Step3: split the sequence dimension
    tp_mesh = get_1D_sub_process_mesh(next_mesh, tp_dim)  
    tp_idx = next(i for i, process_id in enumerate(tp_mesh.process_ids) if process_id == rank)
    tp_length = local_tensor.shape[seq_dim] // len(tp_mesh.process_ids)
    local_tensor = local_tensor[tp_idx * tp_length:(tp_idx + 1) * tp_length, :, :].contiguous()
    
    # Step4: reconstruct the distributed tensor with new mesh
    local_tensor = local_tensor.contiguous()
    input = dtensor_from_local(local_tensor, next_mesh, [dist.Shard(batch_dim), dist.Shard(seq_dim)])
    return input

def gather_batch_with_sequence_parallel(tensor: Tensor, next_mesh: ProcessMesh):
    """
        when dp_degree decreases,
        we need to gather the tensor along the batch dimension,
        and split the tensor along the sequence dimension when the sequence parallel is enabled.
    """
    # print("entering gather_batch_with_sequence_parallel")
    rank = dist.get_rank()
    # print(f'rank {rank}, tensor is {tensor}')
    if rank not in tensor.process_mesh.process_ids:
        return tensor
    
    dp_dim, tp_dim, seq_dim, batch_dim = 0, 1, 0, 1
    
    # Step1: restore the sequence dimension
    dtensor = dist.reshard(tensor, tensor.process_mesh, [dist.Shard(batch_dim), dist.Replicate()]) # restore the sequence dimension
    local_tensor = dtensor_to_local(dtensor, dtensor.process_mesh, dtensor.placements).contiguous()

    # Step2: find the dp groups and gather the batch dimension
    origin_mesh = tensor.process_mesh
    origin_dp_mesh = get_1D_sub_process_mesh(origin_mesh, dp_dim) 
    origin_dp_process_ids = origin_dp_mesh.process_ids
    gather_group_size = origin_mesh.shape[dp_dim] // next_mesh.shape[dp_dim]
    # print(f'rank {rank}, origin_dp_process_ids: {origin_dp_process_ids}, gather_group_size: {gather_group_size}')
    new_dp_groups = [origin_dp_process_ids[i:i + gather_group_size] for i in range(0, len(origin_dp_process_ids), gather_group_size)] # rank in same dp_group need to merge batch
    idx = next(i for i, group in enumerate(new_dp_groups) if rank in group)
    # print(f'rank {rank}, idx: {idx}, new_dp_groups: {new_dp_groups}')

    # 其实就是一个gather
    gather_mesh = dist.ProcessMesh([[process_id] for process_id in new_dp_groups[idx]], dim_names=['dp', 'tp'])
    # print(f'rank {rank}, gather_mesh is {gather_mesh}, idx: {idx}, new_dp_groups: {new_dp_groups}')
    dtensor = dtensor_from_local(local_tensor, gather_mesh, [dist.Shard(batch_dim), dist.Replicate()])
    dtensor = dist.reshard(dtensor, dtensor.process_mesh, [dist.Replicate(), dist.Replicate()])  # restore the batch dimension
    # print(f'rank {rank}, dtensor after gather, shape is: {dtensor.shape}')
    local_tensor = dtensor_to_local(dtensor, dtensor.process_mesh, dtensor.placements).contiguous()
    
    # Step3: split the sequence dimension
    tp_mesh = get_1D_sub_process_mesh(next_mesh, tp_dim) 
    tp_idx = next(i for i, process_id in enumerate(tp_mesh.process_ids) if process_id == rank)
    tp_length = local_tensor.shape[seq_dim] // len(tp_mesh.process_ids)
    local_tensor = local_tensor[tp_idx * tp_length:(tp_idx + 1) * tp_length, :, :].contiguous()
    
    # Step4: reconstruct the distributed tensor with new mesh
    local_tensor = local_tensor.contiguous()
    input = dtensor_from_local(local_tensor, next_mesh, [dist.Shard(batch_dim), dist.Shard(seq_dim)]) 
    return input

class SpiltBatchFwdGatherBatchBwd(PyLayer):
    @staticmethod
    def forward(ctx, dtensor:Tensor, mesh:ProcessMesh):
        origin_mesh = dtensor.process_mesh
        ctx.origin_mesh_shape = origin_mesh.shape
        ctx.origin_mesh_process_ids = origin_mesh.process_ids
        # ctx.origin_mesh = origin_mesh
        input = split_batch_with_sequence_parallel(dtensor, mesh)
        return input
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # print('SpiltBatchFwdGatherBatchBwd backward')
        origin_mesh_shape = ctx.origin_mesh_shape
        origin_mesh_process_ids = ctx.origin_mesh_process_ids
        dp, tp = origin_mesh_shape
        process_list = [origin_mesh_process_ids[i * tp : (i + 1) * tp] for i in range(dp)]
        origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'tp'])
        # print('origin_mesh is ', origin_mesh)
        out = gather_batch_with_sequence_parallel(grad_output, origin_mesh)
        return out

class GatherBatchFwdSplitBatchBwd(PyLayer):
    @staticmethod
    def forward(ctx, dtensor:Tensor, mesh:ProcessMesh):
        origin_mesh = dtensor.process_mesh
        ctx.origin_mesh_shape = origin_mesh.shape
        ctx.origin_mesh_process_ids = origin_mesh.process_ids
        input = gather_batch_with_sequence_parallel(dtensor, mesh)
        return input
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        origin_mesh_shape = ctx.origin_mesh_shape
        origin_mesh_process_ids = ctx.origin_mesh_process_ids
        dp, tp = origin_mesh_shape
        process_list = [origin_mesh_process_ids[i * tp : (i + 1) * tp] for i in range(dp)]
        origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'tp'])
        # print('GatherBatchFwdSplitBatchBwd backward')
        # print('origin_mesh is ', origin_mesh)?
        out = split_batch_with_sequence_parallel(grad_output, origin_mesh)
        return out
    


# def split_batch_with_sequence_parallel_version1(tensor: Tensor, next_mesh: ProcessMesh):
#     """
#         when dp_degree increases, 
#         we need to split the tensor along the batch dimension, 
#         and gather the tensor along the sequence dimension when the sequence parallel is enabled.
#     """
#     # shape = [seq_len, batch_size, hidden_size]
#     local_tensor = dtensor_to_local(tensor, tensor.process_mesh, tensor.placements)
    
#     rank = dist.get_rank()
#     tp_dim = 1
#     origin_mesh = tensor.process_mesh
    
#     origin_tp_mesh = get_1D_sub_process_mesh(origin_mesh, tp_dim) # When origin_mesh is Mesh([[0,1],[2,3]], dim_names=['dp','tp']), origin_tp_mesh is [0, 1] if rank is 0 or 1
#     origin_tp_process_ids = origin_tp_mesh.process_ids
    
#     split_group_num = origin_mesh.shape[tp_dim] // next_mesh.shape[tp_dim]
#     split_group_size = len(origin_tp_process_ids) // split_group_num
#     new_tp_groups = [origin_tp_process_ids[i:i + split_group_size] for i in range(0, len(origin_tp_process_ids), split_group_size)]
#     batch_length = local_tensor.shape[1] // split_group_num
#     seq_length = local_tensor.shape[0] * split_group_num
    
#     dp_idx, tp_idx = next(((i, j) for i, group in enumerate(new_tp_groups) for j, process_id in enumerate(group) if rank in group and process_id == rank), (-1, -1))
#     assert dp_idx != -1 and tp_idx != -1, f"rank {rank} not found in new_tp_groups: {new_tp_groups}"
#     print(f"rank {rank} dp_idx: {dp_idx}, tp_idx: {tp_idx}, split_group_num: {split_group_num}, batch_length: {batch_length}, seq_length: {seq_length}")
    
#     for i in range(split_group_num):
#         part_local_tensor = local_tensor[:, i * batch_length:(i + 1) * batch_length, :]
#         print(f"rank {rank} part_local_tensor: {part_local_tensor}")
#         mesh = ProcessMesh([[0, 1, 2, 3]], dim_names=['dp', 'tp'])
#         part_global_tensor = dtensor_from_local(part_local_tensor, mesh, [dist.Replicate(), dist.Shard(0)]) # 这个地方应该是只需要在原先的tp组进行就好
#         print(f"rank {rank} part_global_tensor: {part_global_tensor}")
#         if i == dp_idx:
#             part_global_tensor = dist.reshard(part_global_tensor, origin_mesh, [dist.Replicate(), dist.Replicate()]) # restore the sequence dimension to replicate
#             split_tensor = dtensor_to_local(part_global_tensor, part_global_tensor.process_mesh, part_global_tensor.placements)
#             split_tensor = split_tensor[tp_idx * seq_length:(tp_idx + 1) * seq_length, :, :] # split the sequence dimension
#             # print(f"rank {rank} split_tensor: {split_tensor}")

#     print(f'rank {rank} split_tensor: {split_tensor}')
#     input = dtensor_from_local(split_tensor, next_mesh, [dist.Shard(1), dist.Shard(0)])
#     return input
    
# def split_batch_with_sequence_parallel_version2(tensor: Tensor, next_mesh: ProcessMesh):
#     """
#         when dp_degree increases, 
#         we need to split the tensor along the batch dimension, 
#         and gather the tensor along the sequence dimension when the sequence parallel is enabled.
#     """
#     rank = dist.get_rank()
#     tp_dim = 1
#     origin_mesh = tensor.process_mesh
#     placements = copy.deepcopy(tensor.placements) # Actually, placements is [Shard(dim=1), Shard(dim=0)]
#     origin_local_shape = tensor._local_shape
    
#     origin_tp_mesh = get_1D_sub_process_mesh(origin_mesh, tp_dim) # When origin_mesh is Mesh([[0,1],[2,3]], dim_names=['dp','tp']), origin_tp_mesh is [0, 1] if rank is 0 or 1
#     origin_tp_process_ids = origin_tp_mesh.process_ids
    
#     split_group_num = origin_mesh.shape[tp_dim] // next_mesh.shape[tp_dim]
#     split_group_size = len(origin_tp_process_ids) // split_group_num
#     new_tp_groups = [origin_tp_process_ids[i:i + split_group_size] for i in range(0, len(origin_tp_process_ids), split_group_size)]
    
#     batch_length = origin_local_shape[1] // split_group_num
#     seq_length = origin_local_shape[0] * split_group_num
    
#     dp_idx, tp_idx = next(((i, j) for i, group in enumerate(new_tp_groups) for j, process_id in enumerate(group) if rank in group and process_id == rank), (-1, -1))
#     assert dp_idx != -1 and tp_idx != -1, f"rank {rank} not found in new_tp_groups: {new_tp_groups}"
#     print(f"rank {rank} dp_idx: {dp_idx}, tp_idx: {tp_idx}, split_group_num: {split_group_num}, batch_length: {batch_length}, seq_length: {seq_length}")
    
#     dtensor = tensor[:, dp_idx * batch_length : (dp_idx + 1) * batch_length,:] # 这个地方会有问题 所以这种方式可以直接排除了
#     print(f'rank {rank} dtensor after slicing: {dtensor}')
#     dtensor = dist.reshard(dtensor, origin_mesh, [dist.Shard(1), dist.Replicate()])
#     print(f'rank {rank} dtensor after reshard: {dtensor}')
#     local_tensor = dtensor_to_local(dtensor, dtensor.process_mesh, dtensor.placements)
#     print(f'rank {rank} local_tensor after dtensor_to_local: {local_tensor}')
#     local_tensor = local_tensor[tp_idx * seq_length:(tp_idx + 1) * seq_length, :, :]  # split the sequence dimension
#     print(f'rank {rank} local_tensor after slicing: {local_tensor}')
#     input = dtensor_from_local(local_tensor, next_mesh, placements)
#     return input