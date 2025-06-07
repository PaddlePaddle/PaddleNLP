import paddle.nn as nn
import paddle.distributed as dist
from paddle import framework
from paddle import core

def print_dtensor(dtensor):
    print(dtensor)
    local_tensor = dtensor._local_value()
    print(local_tensor)

def test(dim):
    mesh = dist.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'tp'])
    
    embedding = nn.Embedding(32000, 4096)
    placement = [dist.Replicate(), dist.Shard(dim)]
    embedding.weight = dist.shard_tensor(embedding.weight, mesh, placement)
    print_dtensor(embedding.weight)

    print('After model initialization, current allocated memory')
    current_device = framework._current_expected_place_()
    max_memory_allocated = core.device_memory_stat_peak_value("Allocated", current_device.get_device_id()) / 2**20
    current_memory_allocated = core.device_memory_stat_current_value("Allocated", current_device.get_device_id()) / 2**20
    print(f"Max memory allocated: {max_memory_allocated} MB")
    print(f"Current memory allocated: {current_memory_allocated} MB")
    
if __name__ == '__main__':
    test(1)