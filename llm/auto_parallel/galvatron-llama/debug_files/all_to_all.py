import paddle
import paddle.distributed as dist
import numpy as np

def sep_reshard_layer(input, split_axis, concat_axis):
    # do alltoall operation to reshard input from [Shard(concat_axis)] to [Shard[split_axis]]
    sep_axis = input.process_mesh.dim_names.index("sep")

    input_placements = input.placements
    if not isinstance(input_placements[sep_axis], dist.Shard):
        raise ValueError(
            f"Input placements for 'sep' axis should be Shard({split_axis}), but got {input_placements[sep_axis]}"
        )

    if input_placements[sep_axis].get_dim() != concat_axis: # 这个的意思是说 原先的split轴需要是现在的concat轴
        raise ValueError(
            f"Input placements for 'sep' axis should be Shard({concat_axis}), but got {input_placements[sep_axis]}"
        )

    input_placements[sep_axis] = dist.Shard(split_axis)

    out = dist.reshard(input, input.process_mesh, input_placements)
    return out

def run_all_to_all_demo():
    # 初始化分布式环境
    dist.init_parallel_env()
    
    # 创建8卡的进程网格
    mesh = dist.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'sep'])
    
    # 每个卡上的初始数据
    batch_size = 8
    seq_length = 24
    hidden_size = 16
    matrix = np.zeros((batch_size, seq_length, hidden_size), dtype=np.int16)  # 改用int16节省空间
    for b in range(batch_size):
        for s in range(seq_length):
            for h in range(hidden_size):
                matrix[b, s, h] = (b + 1) * 100 + (s + 1) * 10 + (h + 1)
    tensor = paddle.to_tensor(matrix, dtype='bfloat16')  # 使用bfloat16类型
    
    dtensor = dist.shard_tensor(tensor, mesh, [dist.Shard(0), dist.Shard(1)])
    
    print(f'Rank {dist.get_rank()} 的初始数据: {dtensor._local_value()}')
    print(f'Rank {dist.get_rank()} 的初始数据形状: {dtensor._local_shape}')
    
    all_to_all_dtensor = sep_reshard_layer(dtensor, split_axis=2, concat_axis=1)
    
    print(f'Rank {dist.get_rank()} 的all_to_all操作后的数据形状: {all_to_all_dtensor._local_value()}')
    print(f'Rank {dist.get_rank()} 的all_to_all操作后的数据形状: {all_to_all_dtensor._local_shape}')
    
    
    
    # data_list = [[rank for _ in range(seq_length)] for rank in range(batch_size)]
    # data_tensor = paddle.to_tensor(data_list, dtype='float32')
    
    # data_dtensor = dist.shard_tensor(data_tensor, mesh, [dist.Shard(0), dist.Replicate()])

    # local_batch_size = 2
    # feature_size = 4
    # local_data = np.random.rand(local_batch_size, feature_size).astype('float32')
    
    # # 创建分布式张量，初始分片在batch维度(axis=0)
    # input_tensor = paddle.to_tensor(local_data)
    # dist_input = dist.shard_tensor(
    #     input_tensor,
    #     mesh,
    #     [dist.Shard(0)]  # 初始分片在batch维度
    # )
    
    # print("初始分片情况:")
    # print(f"每个卡上的数据形状: {local_data.shape}")
    # print(f"全局张量形状: {[local_batch_size * 8, feature_size]}")
    
    # # 使用sep_reshard_layer进行all_to_all操作
    # # 从Shard(0)转换为Shard(1)，即从batch维度分片转为feature维度分片
    # output = sep_reshard_layer(dist_input, split_axis=1, concat_axis=0)
    
    # # 同步等待所有卡完成
    # paddle.device.cuda.synchronize()
    
    # # 打印结果
    # print(f"Rank {dist.get_rank()} 转换后的本地数据形状: {output.shape}")
    # print(f"Rank {dist.get_rank()} 转换后的本地数据:\n{output.numpy()}")

if __name__ == '__main__':
    run_all_to_all_demo()