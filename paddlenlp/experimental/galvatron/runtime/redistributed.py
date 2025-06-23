import copy
from paddle.autograd import PyLayer
from paddle import Tensor
import paddle.distributed as dist
from paddle.distributed import ProcessMesh
from paddle.distributed.auto_parallel.api import dtensor_from_local, dtensor_to_local, unshard_dtensor
from paddle.distributed.auto_parallel.static.reshard_funcs.nd_mesh_reshard_func import get_1D_sub_process_mesh
import paddle.nn as nn
import paddle

class DummyLayer(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
    
    def forward(self, hidden_states):
        return hidden_states

def split_batch_with_sequence_parallel(tensor: Tensor, next_mesh: ProcessMesh):
    """
        when dp_degree increases, 
        we need to split the tensor along the batch dimension, 
        and gather the tensor along the sequence dimension when the sequence parallel is enabled.
    """
    rank = dist.get_rank()
    
    if rank not in tensor.process_mesh.process_ids:
        return tensor
        # assert False, f'rank {rank} not in tensor.process_mesh.process_ids, tensor.process_mesh.process_ids: {tensor.process_mesh.process_ids}'
    
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
    rank = dist.get_rank()
    
    if rank not in tensor.process_mesh.process_ids:
        return tensor
        # assert False, f'rank {rank} not in tensor.process_mesh.process_ids, tensor.process_mesh.process_ids: {tensor.process_mesh.process_ids}'
    
    dp_dim, tp_dim, seq_dim, batch_dim = 0, 1, 0, 1
    
    # Step1: restore the sequence dimension
    dtensor = dist.reshard(tensor, tensor.process_mesh, [dist.Shard(batch_dim), dist.Replicate()]) # restore the sequence dimension
    local_tensor = dtensor_to_local(dtensor, dtensor.process_mesh, dtensor.placements).contiguous()

    # Step2: find the dp groups and gather the batch dimension
    origin_mesh = tensor.process_mesh
    origin_dp_mesh = get_1D_sub_process_mesh(origin_mesh, dp_dim) 
    origin_dp_process_ids = origin_dp_mesh.process_ids
    gather_group_size = origin_mesh.shape[dp_dim] // next_mesh.shape[dp_dim]
    new_dp_groups = [origin_dp_process_ids[i:i + gather_group_size] for i in range(0, len(origin_dp_process_ids), gather_group_size)] # rank in same dp_group need to merge batch
    idx = next(i for i, group in enumerate(new_dp_groups) if rank in group)

    # Actually, this operation is gather
    gather_mesh = dist.ProcessMesh([[process_id] for process_id in new_dp_groups[idx]], dim_names=['dp', 'tp'])
    dtensor = dtensor_from_local(local_tensor, gather_mesh, [dist.Shard(batch_dim), dist.Replicate()])
    dtensor = dist.reshard(dtensor, dtensor.process_mesh, [dist.Replicate(), dist.Replicate()])  # restore the batch dimension
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
        input = split_batch_with_sequence_parallel(dtensor, mesh)
        return input
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        origin_mesh_shape = ctx.origin_mesh_shape
        origin_mesh_process_ids = ctx.origin_mesh_process_ids
        dp, tp = origin_mesh_shape
        process_list = [origin_mesh_process_ids[i * tp : (i + 1) * tp] for i in range(dp)]
        origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'tp'])
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
        out = split_batch_with_sequence_parallel(grad_output, origin_mesh)
        return out
    
class DummyRedistributed(PyLayer):
    @staticmethod
    def forward(ctx, dtensor: Tensor, mesh: ProcessMesh):
        origin_mesh = dtensor.process_mesh
        origin_dtensor_shape = dtensor.shape
        dummy_tensor = paddle.randn((origin_dtensor_shape), dtype=dtensor.dtype)
        dummy_dtensor = dist.shard_tensor(dummy_tensor, mesh, [dist.Shard(1), dist.Shard(0)])
        # print(f"DummyRedistributed forward, dtensor shape: {dtensor.shape}, mesh: {mesh}, dummy_dtensor shape: {dummy_dtensor.shape}")
        print(f'DummyRedistributed forward, dummy_dtensor: {dummy_dtensor}')
        return dummy_dtensor
    
    def backward(ctx, grad_output: Tensor):
        return grad_output