from paddlenlp.experimental.galvatron.runtime.redistributed import split_batch_with_sequence_parallel, gather_batch_with_sequence_parallel
import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.api import dtensor_to_local, dtensor_from_local
import copy
import time

def print_dtensor(dtensor):
    print(dtensor)
    local_tensor = dtensor_to_local(dtensor, dtensor.process_mesh, dtensor.placements)
    print(local_tensor)    

if __name__ == '__main__':
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    base_data = paddle.arange(0, 16 * 4, dtype='float32').reshape([16, 4]) # seq_len=16, batch_size=4, hidden_size=1
    all_data_tensor = base_data.unsqueeze(-1)

    mesh1 = dist.ProcessMesh([[0, 1, 2, 3]], dim_names=['dp', 'tp'])
    mesh2 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['dp', 'tp'])
    mesh3 = dist.ProcessMesh([[0], [1], [2], [3]], dim_names=['dp', 'tp'])
    
    meshs = [mesh1, mesh2, mesh3]
    
    placements = [dist.Shard(1), dist.Shard(0)] # dp维度对batch切分，tp维度对seq切分
    dtensor = dist.shard_tensor(all_data_tensor, mesh1, placements)
    print("original dtensor is")
    print_dtensor(dtensor)
    
    print('from mesh1 to mesh3')
    next_tensor = split_batch_with_sequence_parallel(dtensor, mesh3)
    print_dtensor(next_tensor)
    
    print('from mesh3 to mesh2')
    next_tensor = gather_batch_with_sequence_parallel(next_tensor, mesh2)
    print_dtensor(next_tensor)
    
    # local_tensor = split_batch_with_sequence_parallel(dtensor, mesh2)
    # print(f'local tensor is {local_tensor}')
    # print(f'type(local_tensor) is {type(local_tensor)}')
    # # local_tensor = copy.deepcopy(local_tensor)
    # local_tensor = local_tensor.contiguous()
    # time.sleep(5)
    # next_tensor = dtensor_from_local(local_tensor, mesh2, [dist.Shard(1), dist.Shard(0)])
    # print_dtensor(next_tensor)
    
        
    # print('from mesh1 to mesh2')
    # next_tensor = split_batch_with_sequence_parallel(dtensor, mesh2)
    # print_dtensor(next_tensor)
    
    # if rank == 0:
    #     local_tensor = paddle.to_tensor(
    #         [
    #             [[0], [1]],
    #             [[4], [5]],
    #             [[8], [9]],
    #             [[12], [13]],
    #             [[16], [17]],
    #             [[20], [21]],
    #             [[24], [25]],
    #             [[28], [29]],
    #         ],
    #         dtype='float32'
    #     )
    # elif rank == 1:
    #     local_tensor = paddle.to_tensor(
    #         [
    #             [[32], [33]],
    #             [[36], [37]],
    #             [[40], [41]],
    #             [[44], [45]],
    #             [[48], [49]],
    #             [[52], [53]],
    #             [[56], [57]],
    #             [[60], [61]],
    #         ],
    #         dtype='float32'
    #     )
    # elif rank == 2:
    #     local_tensor = paddle.to_tensor(
    #         [
    #             [[2], [3]],
    #             [[6], [7]],
    #             [[10], [11]],
    #             [[14], [15]],
    #             [[18], [19]],
    #             [[22], [23]],
    #             [[26], [27]],
    #             [[30], [31]],
    #         ],
    #         dtype='float32'
    #     )
    # else:
    #     local_tensor = paddle.to_tensor(
    #         [
    #             [[34], [35]],
    #             [[38], [39]],
    #             [[42], [43]],
    #             [[46], [47]],
    #             [[50], [51]],
    #             [[54], [55]],
    #             [[58], [59]],
    #             [[62], [63]],
    #         ],
    #         dtype='float32'
    #     )
    
    # print("test==========================")
    # print(f'local tensor is{local_tensor}')
    # local_tensor = local_tensor.contiguous()
    # global_tensor = dtensor_from_local(local_tensor, mesh2, [dist.Shard(1), dist.Shard(0)])
    # print_dtensor(global_tensor)
    #     # next_tensor = gather_batch_with_sequence_parallel(next_tensor, mesh2)
        # print_dtensor(next_tensor)
        
    # if rank == 0:
    #     local_tensor = paddle.to_tensor(
    #         [
    #             [
    #                 [0], [1]
    #             ],
    #             [
    #                 [4], [5]
    #             ],
    #             [
    #                 [8], [9]
    #             ],
    #             [
    #                 [12], [13]
    #             ],
    #          ]
    #     )
    # elif rank == 1:
    #     local_tensor = paddle.to_tensor(
    #         [
    #             [
    #                 [16], [17]
    #             ],
    #             [
    #                 [20], [21]
    #             ],
    #             [
    #                 [24], [25]
    #             ],
    #             [
    #                 [28], [29]
    #             ],
    #          ]
    #     )
    # elif rank == 2:
    #     local_tensor = paddle.to_tensor(
    #         [
    #             [
    #                 [32], [33]
    #             ],
    #             [
    #                 [36], [37]
    #             ],
    #             [
    #                 [40], [41]
    #             ],
    #             [
    #                 [44], [45]
    #             ]
    #          ]
    #     )
    # elif rank == 3:
    #     local_tensor = paddle.to_tensor(
    #         [
    #             [
    #                 [48], [49]
    #             ],
    #             [
    #                 [52], [53]
    #             ],
    #             [
    #                 [56], [57]
    #             ],
    #             [
    #                 [60], [61]
    #             ]
    #          ]
    #     )
    # placements = [dist.Replicate(), dist.Shard(0)]  # dp维度对batch切分，tp维度对seq切分
    # dtensor = dtensor_from_local(local_tensor, mesh1, placements)
    # print(dtensor)

"""
    global tensor:
    0   1   2   3
    4   5   6   7
    8   9   10  11
    12  13  14  15
    16  17  18  19
    20  21  22  23
    24  25  26  27
    28  29  30  31
    32  33  34  35
    36  37  38  39
    40  41  42  43
    44  45  46  47
    48  49  50  51
    52  53  54  55
    56  57  58  59
    60  61  62  64
    
    tp4dp1 
    rank0:
    0   1   2   3
    4   5   6   7
    8   9   10  11
    12  13  14  15
    
    tp2dp2
    rank0:
    0   1 
    4   5 
    8   9 
    12  13
    16  17
    20  21 
    24  25 
    28  29 

"""