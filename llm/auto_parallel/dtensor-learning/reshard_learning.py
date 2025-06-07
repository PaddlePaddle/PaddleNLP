import paddle
import paddle.distributed as dist

def print_dtensor(dtensor):
    print(dtensor)
    local_tensor = dtensor._local_value()
    print(local_tensor)

def test():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    all_data_list = [[i + 1 for _ in range(16)] for i in range(world_size)]
    all_data_tensor = paddle.to_tensor(all_data_list, dtype='float32')
    
    assert world_size == 8, "The world size should be 8 for this example."
    
    mesh1 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['dp', 'tp'])
    mesh2 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=['dp', 'tp'])
    mesh3 = dist.ProcessMesh([[4, 5, 6, 7]], dim_names=['dp', 'tp'])
    
    placements = [dist.Replicate(), dist.Shard(0)]
    dtensor = dist.shard_tensor(all_data_tensor, mesh1, placements)
    print_dtensor(dtensor)
    # dtensor = dist.reshard(dtensor, mesh2, placements)
    # print_dtensor(dtensor)
    dtensor = dist.reshard(dtensor, mesh3, placements)
    print_dtensor(dtensor)


if __name__ == '__main__':
    test()