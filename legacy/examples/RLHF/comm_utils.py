# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import paddle
import paddle.distributed as dist

from paddlenlp.trainer.plugins.unified_checkpoint import flatten_list
from paddlenlp.trainer.trainer import Trainer, logger
from paddlenlp.trainer.utils.helper import nested_broadcast_tensor_with_empty
from paddlenlp.utils.distributed import distributed_gather

global_dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device().split(":")[1])


def offload_tensor_to_cpu(tensors):
    if isinstance(tensors, dict):
        for _, v in tensors.items():
            offload_tensor_to_cpu(v)
    elif isinstance(tensors, paddle.Tensor):
        if tensors.place.is_gpu_place():
            cpu_tensor = tensors._copy_to(paddle.CUDAPinnedPlace(), False)
            tensors.value().get_tensor()._share_data_with(cpu_tensor.value().get_tensor())
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def reload_tensor_to_gpu(tensors):
    if isinstance(tensors, dict):
        for _, v in tensors.items():
            reload_tensor_to_gpu(v)
    elif isinstance(tensors, paddle.Tensor):
        if tensors._is_initialized() and not tensors.place.is_gpu_place():
            gpu_tensor = tensors._copy_to(paddle.CUDAPlace(global_dev_id), False)
            tensors.value().get_tensor()._share_data_with(gpu_tensor.value().get_tensor())
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def cleanup_tensor_space(tensors):
    if isinstance(tensors, dict):
        for _, v in tensors.items():
            cleanup_tensor_space(v)
    elif isinstance(tensors, paddle.Tensor):
        tensors._clear_data()
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def data_group_split(tensors, group):
    if group is None:
        return tensors
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(data_group_split(t, group) for t in tensors)
    elif isinstance(tensors, dict):
        new_dict = {}
        for k, v in tensors.items():
            new_dict[k] = data_group_split(v, group)
        return new_dict
    elif isinstance(tensors, paddle.Tensor):
        return tensors.split(group.nranks)[group.rank]
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def data_group_merge(tensors, group):
    if group is None:
        return tensors

    if isinstance(tensors, (list, tuple)):
        return type(tensors)(data_group_merge(t, group) for t in tensors)
    elif isinstance(tensors, dict):
        new_dict = {}
        for k, v in tensors.items():
            new_dict[k] = data_group_merge(v, group)
        return new_dict
    elif isinstance(tensors, paddle.Tensor):
        tensor_list = []
        all_gather_nd(tensor_list, tensors, group=group, padded=True)
        return paddle.concat(tensor_list)
    else:
        logger.warning(f"Can't parse for type {type(tensors)}")
        return tensors


def group_rank_guard(group, rank=0):
    def decorator(func):
        def wrapper_func(*args, **kwargs):
            if group.rank == rank:
                ret = func(*args, **kwargs)
                dist.barrier()
            else:
                ret = None
                dist.barrier()
            ret = nested_broadcast_tensor_with_empty(ret, group=group)
            return ret

        return wrapper_func

    return decorator


def repad_rl_batches(batches, input_lengths):
    if batches.get("position_ids", None) is not None:
        v = batches["position_ids"]
        for x in range(v.shape[0]):
            v[x, input_lengths[x] :] = 1
        batches["position_ids"] = v
    for key in list(batches.keys()):
        if batches[key].shape[0] != input_lengths.shape[0]:
            batches[key] = batches[key].mean()

    return batches


# https://stackoverflow.com/questions/12594148/skipping-execution-of-with-block
class SkipWithBlock(Exception):
    pass


class SkipContextManager:
    def __init__(self, skip):
        self.skip = skip

    def __enter__(self):
        if self.skip:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        raise SkipWithBlock()

    def __exit__(self, type, value, traceback):
        if type is None:
            return  # No exception
        if issubclass(type, SkipWithBlock):
            return True  # Suppress special SkipWithBlock exception


def all_gather_nd(tensor_list, tensor, group=None, padded=False):
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        (Tensor): output list of tensors that can be of different sizes
    """
    if len(tensor.shape) == 0:
        tensor = tensor.reshape([1])
        dist.all_gather(tensor_list, tensor, group=group)
        return tensor_list

    world_size = group.nranks
    local_size = paddle.to_tensor(tensor.shape, place=tensor.place)
    all_sizes = [paddle.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size, group=group)

    # max_length = max(size[0] for size in all_sizes)

    # length_diff = max_length.item() - local_size[0].item()
    # if length_diff:
    #     pad_size = (length_diff, *tensor.size()[1:])
    #     padding = paddle.zeros(pad_size, place=tensor.place(), dtype=tensor.dtype)
    #     tensor = padle.concat((tensor, padding))

    max_length = max(size[-1] for size in all_sizes)

    length_diff = max_length.item() - local_size[-1].item()
    if length_diff:
        pad_size = (*tensor.shape[:-1], length_diff)
        padding = paddle.zeros(pad_size, dtype=tensor.dtype)
        tensor = paddle.concat([tensor, padding], axis=-1)

    all_tensors_padded = []
    dist.all_gather(all_tensors_padded, tensor, group=group)
    # all_tensors = []
    if padded:
        tensor_list.extend(all_tensors_padded)
        return all_tensors_padded

    for tensor_, size in zip(all_tensors_padded, all_sizes):
        tensor_list.append(tensor_[..., : size[-1]])
    return tensor_list


def export_evaluate_model(self: Trainer, train_model, eval_model, **kwargs):
    if eval_model is None:
        return None

    with_offload = kwargs.pop("with_offload", False)
    train_tp_size = max(train_model.config.tensor_parallel_degree, 1)
    eval_tp_size = max(eval_model.config.tensor_parallel_degree, 1)
    eval_tp_rank = max(eval_model.config.tensor_parallel_rank, 0)

    hcg = dist.fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    sd_group = hcg.get_sharding_parallel_group()
    dp_group = hcg.get_data_parallel_group()

    global_rank = paddle.distributed.get_rank()

    train_state_dict = train_model.state_dict()
    eval_state_dict = eval_model.state_dict()

    if dp_group.rank <= 0 and sd_group.rank <= 0:
        train_pp_size = pp_group.nranks
        if eval_tp_size > 1 and train_tp_size != eval_tp_size:
            raise ValueError("Only support for the same tensor_parallel_degree for train and eval model for now.")

        # 单卡情况
        # tp->single
        # tp+pp -> single
        if eval_tp_size == 1:
            if train_pp_size == 1 and train_tp_size > 1:
                # tp ->single
                logger.error("using tp to single eval model.")
                # state = train_model.merge_tensor_parallel()
                tp_actions = train_model.get_tensor_parallel_convert_actions(
                    train_model.config,
                    loaded_state_dict_keys=eval_state_dict.keys(),
                    is_split=False,
                    ignore_error=False,
                )

                is_dst = global_rank == 0
                for key in eval_state_dict.keys():
                    tensor = train_state_dict[key]
                    if key in tp_actions:
                        ret = distributed_gather(tensor, dst=0, group=tp_group, offload=False)
                        action = tp_actions.pop(key)
                        tensor = action(ret) if is_dst else None
                    else:
                        tensor = tensor._copy_to(paddle.CPUPlace(), False) if is_dst else None

                    if tensor is not None:
                        eval_state_dict[key].set_value(tensor)

                    if not eval_state_dict[key]._is_initialized():
                        v = eval_state_dict[key]
                        t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                        v.get_tensor()._share_data_with(t.get_tensor())

                    if with_offload:
                        offload_tensor_to_cpu(train_state_dict[key])
            else:
                # single to single
                # tp+pp -> single
                raise ValueError("Not support yet.")

        def create_send_recv_table(train_keys, eval_keys):
            recv_table = []
            send_table = []
            if pp_group.rank == 0:
                for key in eval_keys:
                    recv_table.append((key, global_rank))

            for key in train_keys:
                send_table.append((key, global_rank))

            all_recv, all_send = [], []
            paddle.distributed.all_gather_object(all_recv, [recv_table], group=pp_group)
            paddle.distributed.all_gather_object(all_send, [send_table], group=pp_group)
            all_recv = flatten_list(all_recv)
            all_send = flatten_list(all_send)

            send_dict = {}
            for k, v in all_send:
                send_dict[k] = v

            table = []
            for k, v in all_recv:
                # key, send, recv
                table.append([k, send_dict.pop(k), v])
            assert len(send_dict) == 0, f"Some key can't be recv {send_dict.keys()}"
            return table

            # pp0tp0 -> pp0tp0
            # pp0tp1 -> pp0tp1
            # pp1tp0 -> pp0tp0
            # pp1tp1 -> pp0tp1

        # tp情况
        # tp+pp->tp
        self.timers and self.timers("export-merge-pp").start()
        if eval_tp_size > 1 and train_pp_size > 1:
            table = create_send_recv_table(train_state_dict.keys(), eval_state_dict.keys())

            for key, src_rank, dst_rank in table:
                # Init tensor for model is cleaned
                if not eval_state_dict[key]._is_initialized():
                    v = eval_state_dict[key]
                    t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                    v.get_tensor()._share_data_with(t.get_tensor())

                if src_rank == dst_rank and global_rank == src_rank:
                    eval_state_dict[key].copy_(train_state_dict[key], True)
                else:
                    if global_rank == src_rank:
                        dist.stream.send(train_state_dict[key], dst=dst_rank)

                    if global_rank == dst_rank:
                        dist.stream.recv(eval_state_dict[key], src=src_rank)

                # Offload train model if need
                if global_rank == src_rank and with_offload:
                    offload_tensor_to_cpu(train_state_dict[key])

        self.timers and self.timers("export-merge-pp").stop()
        self.timers and self.timers("export-broadcast-pp").start()
        if pp_group.nranks > 1:
            paddle.distributed.parallel.sync_params_buffers(
                eval_model, comm_group=pp_group, src_rank=pp_group.ranks[0], fuse_params=False
            )
        self.timers and self.timers("export-broadcast-pp").stop()
    else:
        # 其他 DP rank 的state dict, 适配 offload 和初始化
        self.timers and self.timers("export-offload-and-init").start()
        if with_offload:
            for key in list(train_state_dict.keys()):
                offload_tensor_to_cpu(train_state_dict[key])
        for k, v in eval_state_dict.items():
            if not v._is_initialized():
                t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                v.get_tensor()._share_data_with(t.get_tensor())
        self.timers and self.timers("export-offload-and-init").stop()

    paddle.distributed.barrier()
    self.timers and self.timers("export-broadcast-sd-dp").start()
    if eval_tp_size == 1:
        for _, tensor in eval_state_dict.items():
            paddle.distributed.broadcast(tensor, src=0, group=None, sync_op=True)
    else:
        if sd_group.nranks > 1:
            if dp_group.rank <= 0:
                paddle.distributed.parallel.sync_params_buffers(
                    eval_model, comm_group=sd_group, src_rank=sd_group.ranks[0], fuse_params=False
                )
        if dp_group.nranks > 1:
            paddle.distributed.parallel.sync_params_buffers(
                eval_model, comm_group=dp_group, src_rank=dp_group.ranks[0], fuse_params=False
            )
    self.timers and self.timers("export-broadcast-sd-dp").stop()
    # paddle.save(eval_state_dict, f"./tmp/eval_{sd_group.rank}_tp_{eval_tp_rank}_pp_{pp_group.rank}.pdparams")
    # paddle.save(train_state_dict, f"./tmp/train_{sd_group.rank}_tp_{tp_group.rank}_pp_{pp_group.rank}.pdparams")
    # paddle.distributed.barrier()
    # exit(-1)

    old_dp_workers = self.args.world_size // (max(sd_group.nranks, 1) * max(dp_group.nranks, 1))
    group_nums = self.args.logical_process_index // old_dp_workers * eval_tp_size + eval_tp_rank

    if not hasattr(self, "_policy_model_eval_group") or self._policy_model_eval_group is None:
        self._policy_model_eval_group = create_data_trans_group(global_rank, group_nums)

    return None


def create_data_trans_group(global_rank, group_nums):
    all_split_table = []
    paddle.distributed.all_gather_object(all_split_table, [(global_rank, group_nums)])
    all_split_table = flatten_list(all_split_table)
    split_dict = {}
    for k, v in all_split_table:
        split_dict[k] = v

    split_ranks = {}
    for k, v in all_split_table:
        if v in split_ranks:
            split_ranks[v].append(k)
        else:
            split_ranks[v] = [k]

    group = None
    for k, ranks in split_ranks.items():
        gp = paddle.distributed.new_group(ranks=ranks)
        if global_rank in ranks:
            group = gp

    return group


Trainer.export_evaluate_model = export_evaluate_model
