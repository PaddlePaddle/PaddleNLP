from paddle.autograd import PyLayer
from paddle import Tensor
import paddle.distributed as dist
from paddle.distributed import ProcessMesh
from paddle.distributed.auto_parallel.api import dtensor_from_local, unshard_dtensor
import copy
import paddle
import paddle.nn as nn
from paddle.distributed.auto_parallel.static.reshard_funcs.nd_mesh_reshard_func import get_1D_sub_process_mesh
import random

def print_dtensor(dtensor):
    print(dtensor)
    local_tensor = dtensor._local_value()
    print(local_tensor)

def split_tensor(tensor:Tensor, next_mesh:ProcessMesh):
    """
        when dp_size increase, we need to split the tensor from one dp process into multiple dp processes.
    """
    local_tensor = tensor._local_value()
    
    rank = dist.get_rank()
    origin_mesh = tensor.process_mesh
    tp_dim = 1 if origin_mesh.dim_names[1] == 'tp' else 0
    placements = copy.deepcopy(tensor.placements) # directly use the placements of the tensor
    
    # adjust local tensor, because when tensor_parallelism is used, the placements of tp dim is [Partitial], rather than [Replicate]  
    tp_mesh = get_1D_sub_process_mesh(origin_mesh, tp_dim)
    tp_process_id_list = tp_mesh.process_ids
    split_group_num = origin_mesh.shape[tp_dim] // next_mesh.shape[tp_dim]
    split_grpup_size = len(tp_process_id_list) // split_group_num
    tp_groups = [tp_process_id_list[i:i + split_grpup_size] for i in range(0, len(tp_process_id_list), split_grpup_size)]
    
    dp_idx, tp_idx = next(((i, j) for i, group in enumerate(tp_groups) for j, process_id in enumerate(group) if rank in group and process_id == rank), (-1, -1))
    assert dp_idx != -1 and tp_idx != -1, f"rank {rank} not found in tp_groups: {tp_groups}"
    
    partial_mesh = ProcessMesh([[group[tp_idx] for group in tp_groups]], dim_names=['dp', 'tp'])
    tensor = dtensor_from_local(local_tensor, partial_mesh, placements=placements)
    tensor = dist.reshard(tensor, tensor.process_mesh, placements=[dist.Shard(0), dist.Replicate()] if origin_mesh.dim_names[0] == 'dp' else [dist.Replicate(), dist.Shard(0)]) # reshard to get the partial sum
    local_tensor = tensor._local_value()
    length = local_tensor.shape[0] // split_group_num
    local_tensor = local_tensor[dp_idx * length:(dp_idx + 1) * length]
    
    split_tensor = dtensor_from_local(local_tensor, next_mesh, placements)
    return split_tensor

def merge_tensor(tensor:Tensor, next_mesh:ProcessMesh):
    """
        when dp_size decrease, we need to merge the tensors from multiple dp processes into one.
    """
    local_tensor = tensor._local_value()
    
    origin_mesh = tensor.process_mesh
    dp_dim = 0 if origin_mesh.dim_names[0] == 'dp' else 1
    placements = copy.deepcopy(tensor.placements)  # directly use the placements of the tensor
    
    dp_mesh = get_1D_sub_process_mesh(origin_mesh, dp_dim)
    dp_process_id_list = dp_mesh.process_ids
    merge_size = origin_mesh.shape[dp_dim] // next_mesh.shape[dp_dim] 
    dp_process_id_list = [dp_process_id_list[i:i + merge_size] for i in range(0, len(dp_process_id_list), merge_size)]
    
    rank = dist.get_rank()
    idx = next(i for i, group in enumerate(dp_process_id_list) if rank in group)
    
    merge_mesh = ProcessMesh([[process_id] for process_id in dp_process_id_list[idx]], dim_names=['dp', 'tp']) if origin_mesh.dim_names[0] == 'dp' else ProcessMesh([dp_process_id_list[idx]], dim_names=['tp', 'dp'])
    merge_placements = [dist.Shard(0), dist.Replicate()] if origin_mesh.dim_names[0] == 'dp' else [dist.Replicate(), dist.Shard(0)]
    local_tensor /= merge_size  # divide by merge_size to get the average values
    tensor = dtensor_from_local(local_tensor, merge_mesh, placements=merge_placements)
    local_tensor = unshard_dtensor(tensor)
    merge_dtensor = dtensor_from_local(local_tensor, next_mesh, placements)

    return merge_dtensor

class SplitFwdMergeBwd(PyLayer):
    @staticmethod
    def forward(ctx, dtensor:Tensor, mesh:ProcessMesh):
        origin_mesh = dtensor.process_mesh
        ctx.origin_mesh = copy.deepcopy(origin_mesh)
        out = split_tensor(dtensor, mesh)
        return out
    
    @staticmethod
    def backward(ctx, out_grad):
        origin_mesh = ctx.origin_mesh
        out = merge_tensor(out_grad, origin_mesh)
        return out
    
class MergeFwdSplitBwd(PyLayer):
    @staticmethod
    def forward(ctx, dtensor:Tensor, mesh:ProcessMesh):
        origin_mesh = dtensor.process_mesh
        ctx.origin_mesh = copy.deepcopy(origin_mesh)
        out = merge_tensor(dtensor, mesh)
        return out
    
    @staticmethod
    def backward(ctx, out_grad):
        origin_mesh = ctx.origin_mesh
        out = split_tensor(out_grad, origin_mesh)
        return out

class LocalLayer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.w0 = nn.Linear(16, 16 * 3, bias_attr=False)
        self.w1 = nn.Linear(16 * 3, 16, bias_attr=False)
        with paddle.no_grad():
            paddle.seed(42)
            w0_init = paddle.uniform(self.w0.weight.shape, min=-0.5, max=0.5)
            w1_init = paddle.uniform(self.w1.weight.shape, min=-0.5, max=0.5)
            self.w0.weight.set_value(w0_init)
            self.w1.weight.set_value(w1_init)
    
    def forward(self, hidden_states):
        hidden_states = self.w0(hidden_states)
        hidden_states = self.w1(hidden_states)
        return hidden_states

class LocalModel(nn.Layer):
    def __init__(self, layer_num):
        super().__init__()
        self.layers = nn.LayerList([LocalLayer() for _ in range(layer_num)])
    
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class DistLayer(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        
        # two linear layers with weight initialization manually set to 0.1
        self.w0 = nn.Linear(16, 16 * 3, bias_attr=False)
        self.w1 = nn.Linear(16 * 3, 16, bias_attr=False)
        with paddle.no_grad():
            paddle.seed(42)
            w0_init = paddle.uniform(self.w0.weight.shape, min=-0.5, max=0.5)
            w1_init = paddle.uniform(self.w1.weight.shape, min=-0.5, max=0.5)
            self.w0.weight.set_value(w0_init)
            self.w1.weight.set_value(w1_init)
        
        # tensor parallelism
        self.w0.weight = dist.shard_tensor(self.w0.weight, self.mesh, [dist.Replicate(), dist.Shard(1)])
        self.w1.weight = dist.shard_tensor(self.w1.weight, self.mesh, [dist.Replicate(), dist.Shard(1)])
    
    def forward(self, hidden_states):
        hidden_states = self.w0(hidden_states)
        hidden_states = self.w1(hidden_states)
        return hidden_states

class SingleMeshModel(nn.Layer):
    def __init__(self, meshs, idx):
        super().__init__()
        self.meshs = meshs
        self.layers = nn.LayerList([DistLayer(meshs[idx]) for _ in meshs])
    
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class MultiMeshModel(nn.Layer):
    def __init__(self, meshs):
        super().__init__()
        self.meshs = meshs
        self.dp_dim = 0 if meshs[0].dim_names[0] == 'dp' else 1
        self.layers = nn.LayerList([DistLayer(mesh) for mesh in meshs])
    
    def forward(self, hidden_states):
        for i, layer in enumerate(self.layers):
            if i != 0:
                current_dp_size = self.meshs[i].shape[self.dp_dim]
                prev_dp_size = self.meshs[i - 1].shape[self.dp_dim]
                if current_dp_size < prev_dp_size: # dp2->dp1: merge
                    hidden_states = MergeFwdSplitBwd.apply(hidden_states, self.meshs[i])
                elif current_dp_size > prev_dp_size: # dp1->dp2: split
                    hidden_states = SplitFwdMergeBwd.apply(hidden_states, self.meshs[i])
                else:
                    pass
            hidden_states = layer(hidden_states)
        return hidden_states

def runtime_test():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"world_size: {world_size}, rank: {rank}")
    
    dp_size_list = []
    i = 1
    while i <= world_size:
        dp_size_list.append(i)
        i *= 2
    
    placements = [dist.Shard(0), dist.Replicate()]
    process_id_list = [[[dp_idx * world_size // dp_size + tp_idx for tp_idx in range(world_size // dp_size)] for dp_idx in range(dp_size)] for dp_size in dp_size_list]
    process_id_list = process_id_list + process_id_list[::-1]  # add reverse order for testing
    random.shuffle(process_id_list)  # randomize the order for testing
    # process_id_list = process_id_list[::-1] # test reverse order
    meshs = [ProcessMesh(process_ids, dim_names=['dp', 'tp']) for process_ids in process_id_list]
    print(process_id_list)
    
    all_data_list = [[i + 1 for _ in range(16)] for i in range(world_size)]
    all_data_tensor = paddle.to_tensor(all_data_list, dtype='float32')
    
    print('==========test local model==========')
    model = LocalModel(len(meshs))
    result = model(all_data_tensor)
    print(f'result: {result}')
    result.backward()
    
    print("==========test single mesh model==========")
    for idx in range(len(meshs)):
        print(f'\t use mesh[{idx}] to test single mesh model')
        model = SingleMeshModel(meshs, idx)
        dtensor = dist.shard_tensor(all_data_tensor, mesh=meshs[idx], placements=placements)
        result = model(dtensor)
        print(f'result: {result}')
        result.backward()
    
    print("==========test multi mesh model==========")
    model = MultiMeshModel(meshs)
    dtensor = dist.shard_tensor(all_data_tensor, mesh=meshs[0], placements=placements)
    result = model(dtensor)
    print(f'result: {result}')
    result.backward()

if __name__ == '__main__':
    runtime_test()


# def split_merge_tensor_test():
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
#     print(f"world_size: {world_size}, rank: {rank}")
    
#     dp_size_list = []
#     i = 1
#     while i <= world_size:
#         dp_size_list.append(i)
#         i *= 2
    
#     placements = [dist.Shard(0), dist.Replicate()]
#     process_id_list = [[[dp_idx * world_size // dp_size + tp_idx for tp_idx in range(world_size // dp_size)] for dp_idx in range(dp_size)] for dp_size in dp_size_list]
#     meshs = [ProcessMesh(process_ids, dim_names=['dp', 'tp']) for process_ids in process_id_list]
#     print(process_id_list)
    
#     all_data_list = [[i for _ in range(16)] for i in range(world_size)]
#     all_data_tensor = paddle.to_tensor(all_data_list, dtype='float32')
    
#     for i in range(len(meshs)):
#         for j in range(len(meshs)):
#             if i == j:
#                 continue
#             prev_mesh = meshs[i]
#             next_mesh = meshs[j]
#             print(f"==========prev_mesh: {prev_mesh}, next_mesh: {next_mesh}==========")
#             prev_dtensor = dist.shard_tensor(all_data_tensor, mesh=prev_mesh, placements=placements)
#             print("prev_dtensor:")
#             print_dtensor(prev_dtensor)
#             if prev_mesh.shape[0] > next_mesh.shape[0]:  # dp decrease (dp2->dp1: gatmergeher)
#                 next_dtensor = merge_tensor(prev_dtensor, next_mesh)
#                 print(f"after merge:dp{prev_mesh.shape[0]}->dp{next_mesh.shape[0]}")
#                 print_dtensor(next_dtensor)
#             elif prev_mesh.shape[0] < next_mesh.shape[0]: # dp increase (dp1->dp2: split)
#                 next_dtensor = split_tensor(prev_dtensor, next_mesh)
#                 print(f"after split:dp{prev_mesh.shape[0]}->dp{next_mesh.shape[0]}")
#                 print_dtensor(next_dtensor)
#             else:
#                 assert False
