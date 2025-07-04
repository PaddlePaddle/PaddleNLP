import paddle.distributed as dist
from paddle.distributed.auto_parallel.api import dtensor_from_local


if __name__ == '__main__':
    
    rank = dist.get_rank()
    
    
    pass