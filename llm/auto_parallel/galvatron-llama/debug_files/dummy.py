import paddle
import paddle.distributed as dist

if __name__ == '__main__':
    rank = dist.get_rank()
    mesh = dist.ProcessMesh([[0, 1, 2, 3]], dim_names=['dp', 'tp'])
    if rank in [4, 5, 6, 7]:
        tensor = paddle.randn((32, 32))
        dtensor = dist.shard_tensor(tensor, mesh, [dist.Shard(1), dist.Shard(0)])
        print(f'[rank {rank}] dtensor: {dtensor}')