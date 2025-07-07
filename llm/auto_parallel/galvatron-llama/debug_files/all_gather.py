import paddle
import paddle.distributed as dist
from paddle.distributed import new_group

dist.init_parallel_env()
tensor_list = []
if dist.get_rank() == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
dist.all_gather(tensor_list, data)
print(tensor_list)
# [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [1, 2, 3]]] (2 GPUs)
if len(tensor_list) > 0:
    merged_tensor = paddle.concat(tensor_list, axis=0)  # 沿第0轴（行方向）拼接
    print("Merged Tensor:\n", merged_tensor)
else:
    print("No tensors to merge.")
    
if dist.get_rank() in [0, 1, 2, 3]:
    tensor_list = []
    pg = new_group(ranks=[0, 1, 2, 3])
    dist.all_gather(tensor_list, data, group=pg)
    print(tensor_list)
    if len(tensor_list) > 0:
        merged_tensor = paddle.concat(tensor_list, axis=0)
        print("Merged Tensor with new group:\n", merged_tensor)