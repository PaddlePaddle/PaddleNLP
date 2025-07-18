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
        assert False, f'rank {rank} should not in tensor.process_mesh.process_ids, but tensor.process_mesh.process_ids is {tensor.process_mesh.process_ids}'

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
    input.stop_gradient = False
    return input

def gather_batch_with_sequence_parallel(tensor: Tensor, next_mesh: ProcessMesh):
    """
        when dp_degree decreases,
        we need to gather the tensor along the batch dimension,
        and split the tensor along the sequence dimension when the sequence parallel is enabled.
    """
    rank = dist.get_rank()
    
    if rank not in tensor.process_mesh.process_ids:
        assert False, f'rank {rank} should in tensor.process_mesh.process_ids, but tensor.process_mesh.process_ids is {tensor.process_mesh.process_ids}'

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
    # print(f'[linguangming] gather new_dp_groups[idx] = {new_dp_groups[idx]}')
    gather_mesh = dist.ProcessMesh([[process_id] for process_id in new_dp_groups[idx]], dim_names=['dp', 'mp'])
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
    input.stop_gradient = False
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
        origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'mp'])
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
        origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'mp'])
        out = split_batch_with_sequence_parallel(grad_output, origin_mesh)
        return out

class RedistributedLayer(PyLayer):
    @staticmethod
    def forward(ctx, dtensor: Tensor, mesh: ProcessMesh):
        origin_mesh = dtensor.process_mesh
        ctx.origin_mesh_shape = origin_mesh.shape
        ctx.origin_mesh_process_ids = origin_mesh.process_ids
        
        if origin_mesh.shape[0] < mesh.shape[0]: # dp increase -> split batch
            input = split_batch_with_sequence_parallel(dtensor, mesh)
        else: # dp decrease -> gather batch
            input = gather_batch_with_sequence_parallel(dtensor, mesh)
        input.stop_gradient = False  
        return input
            
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # print(f'[linguangming] RedistributedLayer backward, grad_output dir is {dir(grad_output)}')
        
        origin_mesh_shape = ctx.origin_mesh_shape
        origin_mesh_process_ids = ctx.origin_mesh_process_ids
        dp, tp = origin_mesh_shape
        process_list = [origin_mesh_process_ids[i * tp : (i + 1) * tp] for i in range(dp)]
        origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'mp'])
        
        if origin_mesh.shape[0] < grad_output.process_mesh.shape[0]:  # dp decrease -> gather batch
            out = gather_batch_with_sequence_parallel(grad_output, origin_mesh)
        else:  # dp increase -> split batch
            out = split_batch_with_sequence_parallel(grad_output, origin_mesh)
        out.stop_gradient = False
        return out

class DummyRedistributedLayer(PyLayer):
    @staticmethod
    def forward(ctx, dtensor: Tensor, mesh: ProcessMesh):
        if not hasattr(ctx, 'dummy_dtensor_map'):
            ctx.dummy_dtensor_map = {}
        ctx.origin_mesh_shape = dtensor.process_mesh.shape
        ctx.origin_mesh_process_ids = dtensor.process_mesh.process_ids      
        ctx.next_mesh_shape = mesh.shape
        ctx.next_mesh_process_ids = mesh.process_ids
        
        key = str(sorted(mesh.process_ids)) + f'_dp{mesh.shape[0]}_tp{mesh.shape[1]}' + f'_tensor_shape_{dtensor.shape}'
        if key not in ctx.dummy_dtensor_map:
            print(f'[linguangming] DummyRedistributed key {key} init forward')
            origin_dtensor_shape = dtensor.shape
            dummy_tensor = paddle.randn((origin_dtensor_shape), dtype=dtensor.dtype)
            dummy_dtensor = dist.shard_tensor(dummy_tensor, mesh, [dist.Shard(1), dist.Shard(0)])
            dummy_dtensor.stop_gradient = False  # [NOTE] this is important
            ctx.dummy_dtensor_map[key] = dummy_dtensor
            
        return ctx.dummy_dtensor_map[key]
    
    def backward(ctx, grad_output: Tensor):
        # print(f'[linguangming] DummyRedistributedLayer backward, grad_output dir is {dir(grad_output)}')
        
        origin_mesh_shape = ctx.origin_mesh_shape
        origin_mesh_process_ids = ctx.origin_mesh_process_ids
        dp, tp = origin_mesh_shape[0], origin_mesh_shape[1]
        process_list = [origin_mesh_process_ids[i * tp : (i + 1) * tp] for i in range(dp)]
        origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'mp'])
        
        key = str(sorted(origin_mesh_process_ids)) + f'_dp{origin_mesh_shape[0]}_tp{origin_mesh_shape[1]}' + f'_tensor_shape_{grad_output.shape}'
        if key not in ctx.dummy_dtensor_map:
            print(f'[linguangming] DummyRedistributed key {key} init backward')
            dummy_tensor = paddle.randn((grad_output.shape), dtype=grad_output.dtype)
            dummy_dtensor = dist.shard_tensor(dummy_tensor, origin_mesh, [dist.Shard(1), dist.Shard(0)])
            dummy_dtensor.stop_gradient = False # [NOTE] this is important
            ctx.dummy_dtensor_map[key] = dummy_dtensor
            
        return ctx.dummy_dtensor_map[key]

def split_batch(dtensor: Tensor, next_mesh: ProcessMesh):
    """
        when dp_degree increases, 
        we need to split the tensor along the batch dimension, 
    """
    rank = dist.get_rank()
    
    if rank not in dtensor.process_mesh.process_ids:
        assert False, f'rank {rank} should not in tensor.process_mesh.process_ids, but tensor.process_mesh.process_ids is {dtensor.process_mesh.process_ids}'

    dp_dim, tp_dim, seq_dim, batch_dim = 0, 1, 1, 0
    
    # Step1: find the tp groups and split the batch dimension    
    local_tensor = dtensor_to_local(dtensor, dtensor.process_mesh, dtensor.placements).contiguous()
    origin_mesh = dtensor.process_mesh
    origin_local_shape = dtensor._local_shape # [seq_len, batch_size, hidden_size]
    
    origin_tp_mesh = get_1D_sub_process_mesh(origin_mesh, tp_dim)
    origin_tp_process_ids = origin_tp_mesh.process_ids
    split_group_num = origin_mesh.shape[tp_dim] // next_mesh.shape[tp_dim]
    split_group_size = len(origin_tp_process_ids) // split_group_num
    new_tp_groups = [origin_tp_process_ids[i:i + split_group_size] for i in range(0, len(origin_tp_process_ids), split_group_size)]
    idx = next(i for i, group in enumerate(new_tp_groups) if rank in group)
    batch_length = origin_local_shape[batch_dim] // split_group_num
    # local_tensor = local_tensor[:, idx * batch_length : (idx + 1) * batch_length, :].contiguous() #  注意此处需要修改
    local_tensor = local_tensor[idx * batch_length : (idx + 1) * batch_length, :, :].contiguous() #  注意此处需要修改

    # Step2: reconstruct the distributed tensor with new mesh
    local_tensor = local_tensor.contiguous()
    input = dtensor_from_local(local_tensor, next_mesh, [dist.Shard(batch_dim), dist.Replicate()])
    input.stop_gradient = False
    return input

def gather_batch(dtensor: Tensor, next_mesh: ProcessMesh):
    """
        when dp_degree decreases,
        we need to gather the tensor along the batch dimension,
    """
    rank = dist.get_rank()
    
    if rank not in dtensor.process_mesh.process_ids:
        assert False, f'rank {rank} should in tensor.process_mesh.process_ids, but tensor.process_mesh.process_ids is {dtensor.process_mesh.process_ids}'

    dp_dim, tp_dim, seq_dim, batch_dim = 0, 1, 1, 0

    # Step1: find the dp groups and gather the batch dimension
    local_tensor = dtensor_to_local(dtensor, dtensor.process_mesh, dtensor.placements).contiguous()
    origin_mesh = dtensor.process_mesh
    origin_dp_mesh = get_1D_sub_process_mesh(origin_mesh, dp_dim) 
    origin_dp_process_ids = origin_dp_mesh.process_ids
    gather_group_size = origin_mesh.shape[dp_dim] // next_mesh.shape[dp_dim]
    new_dp_groups = [origin_dp_process_ids[i:i + gather_group_size] for i in range(0, len(origin_dp_process_ids), gather_group_size)] # rank in same dp_group need to merge batch
    idx = next(i for i, group in enumerate(new_dp_groups) if rank in group)

    # Actually, this operation is gather
    # print(f'[linguangming] gather new_dp_groups[idx] = {new_dp_groups[idx]}')
    gather_mesh = dist.ProcessMesh([[process_id] for process_id in new_dp_groups[idx]], dim_names=['dp', 'mp'])
    dtensor = dtensor_from_local(local_tensor, gather_mesh, [dist.Shard(batch_dim), dist.Replicate()])
    dtensor = dist.reshard(dtensor, dtensor.process_mesh, [dist.Replicate(), dist.Replicate()])  # restore the batch dimension
    local_tensor = dtensor_to_local(dtensor, dtensor.process_mesh, dtensor.placements).contiguous()
    
    # Step4: reconstruct the distributed tensor with new mesh
    local_tensor = local_tensor.contiguous()
    input = dtensor_from_local(local_tensor, next_mesh, [dist.Shard(batch_dim), dist.Replicate()]) 
    input.stop_gradient = False
    return input
 
def get_dummy_dtensor(tensor:Tensor, next_mesh: ProcessMesh):
    if not hasattr(get_dummy_dtensor, 'dummy_dtensor_map'):
        get_dummy_dtensor.dummy_dtensor_map = {}
    
    key = str(sorted(next_mesh.process_ids)) + f'_dp{next_mesh.shape[0]}_tp{next_mesh.shape[1]}' + f'_tensor_shape_{tensor.shape}'
    if key not in get_dummy_dtensor.dummy_dtensor_map:
        print(f'[linguangming] get_dummy_dtensor key {key} init')
        if tensor.dtype == paddle.int64:
            dummy_tensor = paddle.randint(0, 100, (tensor.shape), dtype=tensor.dtype)
        else:
            dummy_tensor =  paddle.randn((tensor.shape), dtype=tensor.dtype)
        dummy_dtensor = dist.shard_tensor(dummy_tensor, next_mesh, [dist.Shard(0), dist.Replicate()])
        dummy_dtensor.stop_gradient = False # [NOTE] only when change label, call this function, so we can set stop_gradient to True # NOTE
        get_dummy_dtensor.dummy_dtensor_map[key] = dummy_dtensor
    return get_dummy_dtensor.dummy_dtensor_map[key]
 
class RedistributedLayerWithoutSequenceParallel(PyLayer):
    @staticmethod
    def forward(ctx, dtensor: Tensor, mesh: ProcessMesh):
        origin_mesh = dtensor.process_mesh
        ctx.origin_mesh_shape = origin_mesh.shape
        ctx.origin_mesh_process_ids = origin_mesh.process_ids
        
        if origin_mesh.shape[0] < mesh.shape[0]: # dp increase -> split batch
            input = split_batch(dtensor, mesh)
        else: # dp decrease -> gather batch
            input = gather_batch(dtensor, mesh)
        input.stop_gradient = False  
        return input
            
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # print(f'[linguangming] RedistributedLayer backward, grad_output dir is {dir(grad_output)}')
        origin_mesh_shape = ctx.origin_mesh_shape
        origin_mesh_process_ids = ctx.origin_mesh_process_ids
        dp, tp = origin_mesh_shape
        process_list = [origin_mesh_process_ids[i * tp : (i + 1) * tp] for i in range(dp)]
        origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'mp'])
        
        if origin_mesh.shape[0] < grad_output.process_mesh.shape[0]:  # dp decrease -> gather batch
            out = gather_batch(grad_output, origin_mesh)
        else:  # dp increase -> split batch
            out = split_batch(grad_output, origin_mesh)
        out.stop_gradient = False
        return out

class DummyRedistributedLayerWithoutSequenceParallel(PyLayer):
    @staticmethod
    def forward(ctx, dtensor: Tensor, mesh: ProcessMesh):
        if not hasattr(ctx, 'dummy_dtensor_map'):
            ctx.dummy_dtensor_map = {}
        ctx.origin_mesh_shape = dtensor.process_mesh.shape
        ctx.origin_mesh_process_ids = dtensor.process_mesh.process_ids      
        ctx.next_mesh_shape = mesh.shape
        ctx.next_mesh_process_ids = mesh.process_ids
        
        key = str(sorted(mesh.process_ids)) + f'_dp{mesh.shape[0]}_tp{mesh.shape[1]}' + f'_tensor_shape_{dtensor.shape}'
        if key not in ctx.dummy_dtensor_map:
            print(f'[linguangming] DummyRedistributedLayerWithoutSequenceParallel key {key} init forward')
            origin_dtensor_shape = dtensor.shape
            # dummy_tensor = paddle.randn((origin_dtensor_shape), dtype=dtensor.dtype)
            if dtensor.dtype == paddle.int64:
                dummy_tensor = paddle.randint(0, 100, (dtensor.shape), dtype=dtensor.dtype)
            else:
                dummy_tensor =  paddle.randn((dtensor.shape), dtype=dtensor.dtype)
            # dummy_dtensor = dist.shard_tensor(dummy_tensor, mesh, [dist.Shard(0), dist.Replicate()])
            dummy_dtensor = dist.shard_tensor(dummy_tensor, mesh, dtensor.placements)
            dummy_dtensor.stop_gradient = False  # [NOTE] this is important
            ctx.dummy_dtensor_map[key] = dummy_dtensor
            
        return ctx.dummy_dtensor_map[key]
    
    def backward(ctx, grad_output: Tensor):  
        if not hasattr(ctx, 'dummy_dtensor_map'):
            ctx.dummy_dtensor_map = {}
                  
        origin_mesh_shape = ctx.origin_mesh_shape
        origin_mesh_process_ids = ctx.origin_mesh_process_ids
        dp, tp = origin_mesh_shape[0], origin_mesh_shape[1]
        process_list = [origin_mesh_process_ids[i * tp : (i + 1) * tp] for i in range(dp)]
        origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'mp'])
        
        key = str(sorted(origin_mesh_process_ids)) + f'_dp{origin_mesh_shape[0]}_tp{origin_mesh_shape[1]}' + f'_tensor_shape_{grad_output.shape}'
        if key not in ctx.dummy_dtensor_map:
            print(f'[linguangming] DummyRedistributedLayerWithoutSequenceParallel key {key} init backward')
            # dummy_tensor = paddle.randn((grad_output.shape), dtype=grad_output.dtype)
            if grad_output.dtype == paddle.int64:
                dummy_tensor = paddle.randint(0, 100, (grad_output.shape), dtype=grad_output.dtype)
            else:
                dummy_tensor =  paddle.randn((grad_output.shape), dtype=grad_output.dtype)
            # dummy_dtensor = dist.shard_tensor(dummy_tensor, origin_mesh, [dist.Shard(0), dist.Replicate()])
            dummy_dtensor = dist.shard_tensor(dummy_tensor, origin_mesh, grad_output.placements)
            dummy_dtensor.stop_gradient = False # [NOTE] this is important
            ctx.dummy_dtensor_map[key] = dummy_dtensor
            
        return ctx.dummy_dtensor_map[key]

 
 
 
 
 
 
 
 
 
 
    
# class DummyRedistributed(PyLayer):
#     @staticmethod
#     def forward(ctx, dtensor: Tensor, mesh: ProcessMesh):  
#         if not hasattr(ctx, 'dummy_dtensor_map'):
#             ctx.dummy_dtensor_map = {}
#         ctx.origin_mesh_shape = dtensor.process_mesh.shape
#         ctx.origin_mesh_process_ids = dtensor.process_mesh.process_ids      
#         ctx.next_mesh_shape = mesh.shape
#         ctx.next_mesh_process_ids = mesh.process_ids
        
#         # if ctx.origin_mesh_shape[0] > ctx.next_mesh_shape[0]: # dp 下降 需要进行gather
#         #     print(f'[linguangming] DummyRedistributed forward, dp decrease, gather')
#         #     gather_batch_with_sequence_parallel_dummy(dtensor, mesh) # 查看一下是否会产生通信组 # [NOTE] (没有必要去补全通信组啊)
#         # else:
#         #     print(f'[linguangming] DummyRedistributed forward, dp increase, split')
#         #     split_batch_with_sequence_parallel_dummy(dtensor, mesh)
        
#         key = str(sorted(mesh.process_ids)) + f'_dp{mesh.shape[0]}_tp{mesh.shape[1]}'
#         if key not in ctx.dummy_dtensor_map:
#             print(f'[linguangming] DummyRedistributed forward, key: {key} init')
#             origin_dtensor_shape = dtensor.shape
#             dummy_tensor = paddle.randn((origin_dtensor_shape), dtype=dtensor.dtype)
#             dummy_dtensor = dist.shard_tensor(dummy_tensor, mesh, [dist.Shard(1), dist.Shard(0)])
#             ctx.dummy_dtensor_map[key] = dummy_dtensor
            
#         return ctx.dummy_dtensor_map[key]
    
#     def backward(ctx, grad_output: Tensor):
#         origin_mesh_shape = ctx.origin_mesh_shape
#         origin_mesh_process_ids = ctx.origin_mesh_process_ids
#         dp, tp = origin_mesh_shape[0], origin_mesh_shape[1]
#         process_list = [origin_mesh_process_ids[i * tp : (i + 1) * tp] for i in range(dp)]
#         origin_mesh = ProcessMesh(process_list, dim_names=['dp', 'tp'])
        
#         # if ctx.next_mesh_shape[0] > ctx.origin_mesh_shape[0]: # dp 增加 需要进行split
#         #     print(f'[linguangming] DummyRedistributed backward, dp increase, split')
#         #     gather_batch_with_sequence_parallel_dummy(grad_output, origin_mesh) # 查看一下是否会产生通信组
#         # else:
#         #     print(f'[linguangming] DummyRedistributed backward, dp decrease, gather')
#         #     split_batch_with_sequence_parallel_dummy(grad_output, origin_mesh)
        
#         key = str(sorted(origin_mesh_process_ids)) + f'_dp{origin_mesh_shape[0]}_tp{origin_mesh_shape[1]}'
#         if key not in ctx.dummy_dtensor_map:
#             print(f'[linguanming] DummyRedistributed backward, key: {key} init')
            
#             dummy_tensor = paddle.randn((grad_output.shape), dtype=grad_output.dtype)
#             dummy_dtensor = dist.shard_tensor(dummy_tensor, origin_mesh, [dist.Shard(1), dist.Shard(0)])
#             ctx.dummy_dtensor_map[key] = dummy_dtensor
            
#         return ctx.dummy_dtensor_map[key]
    

# def split_batch_with_sequence_parallel_dummy(tensor: Tensor, next_mesh: ProcessMesh):
#     rank = dist.get_rank()
#     assert rank not in next_mesh.process_ids, f'rank {rank} should not in next_mesh.process_ids, next_mesh.process_ids: {next_mesh.process_ids}'
#     # tensor_shape = tensor.shape
#     # dummy_tensor = paddle.randn(tensor_shape, dtype=tensor.dtype)
#     # dummy_dtensor = dist.shard_tensor(dummy_tensor, next_mesh, [dist.Shard(1), dist.Shard(0)])
#     # return dummy_dtensor
#     test_tensor = paddle.randn((32, 32), dtype=tensor.dtype)  # 这里的shape需要和tensor的shape一致
#     test_dtensor = dist.shard_tensor(test_tensor, next_mesh, [dist.Shard(1), dist.Shard(0)])
#     test_dtensor = dist.reshard(test_dtensor, next_mesh, [dist.Shard(1), dist.Replicate()])  # restore the batch dimension
#     return

# def gather_batch_with_sequence_parallel_dummy(tensor: Tensor, next_mesh: ProcessMesh):
#     print(f'[linguangming] enter gather_batch_with_sequence_parallel_dummy')
#     rank = dist.get_rank()
#     origin_rank = rank
#     rank = origin_rank % len(next_mesh.process_ids) + next_mesh.process_ids[0]
    
#     dp_dim, tp_dim, seq_dim, batch_dim = 0, 1, 0, 1

#     # Step2: find the dp groups and gather the batch dimension
#     origin_mesh = tensor.process_mesh
#     origin_dp_mesh = get_1D_sub_process_mesh(origin_mesh, dp_dim)  # [NOTE] 这个地方会出现bug  其实可以写一个函数就好了。。。
#     origin_dp_process_ids = origin_dp_mesh.process_ids
#     gather_group_size = origin_mesh.shape[dp_dim] // next_mesh.shape[dp_dim]
#     new_dp_groups = [origin_dp_process_ids[i:i + gather_group_size] for i in range(0, len(origin_dp_process_ids), gather_group_size)] # rank in same dp_group need to merge batch
#     print(f'[linguangming] rank {rank}, new_dp_groups: {new_dp_groups}')
#     idx = next(i for i, group in enumerate(new_dp_groups) if rank in group)

#     # return 
#     # Actually, this operation is gather
#     print(f'[linguangming] gather new_dp_groups[idx] = {new_dp_groups[idx]}')
#     gather_mesh = dist.ProcessMesh([[process_id] for process_id in new_dp_groups[idx]], dim_names=['dp', 'tp'])
#     test_tensor = paddle.randn((32, 32), dtype=tensor.dtype)  # 这里的shape需要和tensor的shape一致
#     test_dtensor = dist.shard_tensor(test_tensor, gather_mesh, [dist.Shard(batch_dim), dist.Replicate()])
#     test_dtensor = dist.reshard(test_dtensor, gather_mesh, [dist.Replicate(), dist.Replicate()])  # restore the batch dimension
#     # test_local_tensor = dtensor_to_local(test_dtensor, gather_mesh, test_dtensor.placements)
#     return 