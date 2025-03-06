import paddle
import numpy as np
import unittest
from paddlenlp_ops import trt_reduce
import paddle.distributed as dist

from paddlenlp.trl import llm_utils

def test_custom_allreduce():
    dist.init_parallel_env()
    input_tensor = paddle.ones([1, 4096], "float16")
    input_tensor_copy = paddle.to_tensor(input_tensor)

    for i in range(5):
        dist.all_reduce(input_tensor_copy)
    print("nccl all reduce: ", input_tensor_copy)

    tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()
    for i in range(5):
        out = trt_reduce(input_tensor, tensor_parallel_rank, tensor_parallel_degree)
    print("custom all reduce: ", out)
    # np.testing.assert_allclose(input_tensor_copy.numpy(), out.numpy(), rtol=1e-3, err_msg="trt_reduce get different result") 

if __name__ == "__main__":
    test_custom_allreduce()