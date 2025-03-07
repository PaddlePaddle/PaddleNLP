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

import os
import sys
from enum import Enum, auto

import paddle
import paddle.distributed as dist
from paddle import nn

from paddlenlp.trainer import strtobool
from paddlenlp.trainer.trainer import Trainer, logger
from paddlenlp.utils.distributed import distributed_gather
from paddlenlp.utils.nested import flatten_list, nested_broadcast_tensor_with_empty

global_dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device().split(":")[1])


class ActorStages(Enum):
    """
    Enum class, the stages of the actor training process.
    """

    MODEL_ENABLE_DISABLE = auto()
    RL_STEP = auto()
    PTX_STEP = auto()


class CriticStages(Enum):
    """
    Enum class, the stages of the critic training process.
    """

    MODEL_ENABLE_DISABLE = auto()
    CRITIC_TRAINING_STEP = auto()


class RolloutStages(Enum):
    """
    Enum class, the stages of the rollout process.
    """

    ACTOR_MODEL_ENABLE_DISABLE = auto()
    GENERATE = auto()
    ROLLOUT_LOGPROB = auto()
    REWARD_MODEL_ENABLE_DISABLE = auto()
    ROLLOUT_REWARD_VALUE = auto()


def get_timer_label(stage: Enum) -> str:
    """
    获取计时器标签。

    Args:
        stage (Enum): RolloutStages/CriticStages/RolloutStages.

    Returns:
        str: 打印Timer时的前缀。格式为 "[prefix] stage number.description"。
            - prefix: 阶段前缀，如"actor-step"、"critic-step"等。
            - stage number: 从1开始编号。
            - description: 阶段描述，小写形式。
    """
    step_prefix = {
        ActorStages.MODEL_ENABLE_DISABLE: "actor-step",
        ActorStages.RL_STEP: "actor-step",
        ActorStages.PTX_STEP: "actor-step",
        CriticStages.MODEL_ENABLE_DISABLE: "critic-step",
        CriticStages.CRITIC_TRAINING_STEP: "critic-step",
        RolloutStages.ACTOR_MODEL_ENABLE_DISABLE: "rollout",
        RolloutStages.GENERATE: "rollout",
        RolloutStages.ROLLOUT_LOGPROB: "rollout",
        RolloutStages.REWARD_MODEL_ENABLE_DISABLE: "rollout",
        RolloutStages.ROLLOUT_REWARD_VALUE: "rollout",
    }
    # stage
    prefix = step_prefix.get(stage, "unknown")
    # index
    stage_number = list(stage.__class__).index(stage) + 1
    # description
    description = stage.name.lower()  # .replace('_', ' ')
    # all
    return f"[{prefix}] {stage_number}.{description}"


@paddle.no_grad()
def _move_param(src, device=None, blocking=True):
    """
    将参数从源设备移动到目标设备，并返回目标设备上的参数。如果目标设备未指定，则使用当前设备。

    Args:
        src (Tensor): 需要移动的参数张量。
        device (Optional[Union[str, paddle.Device]], optional): 目标设备，默认为None，表示使用当前设备。可以是字符串或paddle.Device对象。默认为None。
        blocking (bool, optional): 是否阻塞等待操作完成，默认为True。

    Returns:
        Tensor: 在目标设备上的参数张量。
    """
    if isinstance(device, str):
        device = paddle.device._convert_to_place(device)
    dst = src._copy_to(device, blocking)
    dst_tensor = dst.value().get_tensor()
    src_tensor = src.value().get_tensor()
    src_tensor._clear()
    src_tensor._share_data_with(dst_tensor)


def offload_tensor_to_cpu(tensors):
    """
    将给定的张量迁移到CPU上。如果使用了CUDA管理内存，则该函数无效。

    Args:
        tensors (tuple, list): tuple或list，包含两个元素，第一个元素是模型或优化器，第二个元素是字符串，表示是否为模型或优化器。

    Returns:
        None, 无返回值，直接修改原有张量。

    Raises:
        None, 没有引发任何异常。
    """
    if strtobool(os.getenv("FLAGS_use_cuda_managed_memory", "False")):
        logger.warning("FLAGS_use_cuda_managed_memory has been set to True, " "offloading strategy is ineffective.")
        return

    pin_device = paddle.CUDAPinnedPlace()

    def clear_main_grad(model):
        for param in model.parameters():
            if hasattr(param, "main_grad") and param.main_grad is not None:
                param.main_grad._clear_data()
                param.main_grad = None

    # optimizer
    if "optimizer" in tensors[1]:
        optimizer = tensors[0]
        # offload moment1
        for key, value in optimizer._accumulators[optimizer._moment1_acc_str].items():
            if value._is_initialized() and not isinstance(value.place, paddle.CUDAPinnedPlace):
                optimizer._accumulators[optimizer._moment1_acc_str][key] = value.pin_memory()

        # offload moment2
        for key, value in optimizer._accumulators[optimizer._moment2_acc_str].items():
            if value._is_initialized() and not isinstance(value.place, paddle.CUDAPinnedPlace):
                optimizer._accumulators[optimizer._moment2_acc_str][key] = value.pin_memory()

        # offload master_weight
        for key, value in optimizer._master_weights.items():
            if value._is_initialized() and not isinstance(value.place, paddle.CUDAPinnedPlace):
                optimizer._master_weights[key] = value.pin_memory()
    # model
    elif "model" in tensors[1]:
        model = tensors[0]
        clear_main_grad(model)
        for name, src in model.named_parameters():
            if src._is_initialized() and not isinstance(src.place, paddle.CUDAPinnedPlace):
                _move_param(src, pin_device)

    elif "tensor" in tensors[1]:
        src = tensors[0]
        if src._is_initialized() and not isinstance(src.place, paddle.CUDAPinnedPlace):
            _move_param(src, pin_device)
    else:
        logger.debug(f"Can't parse for type {tensors[1]}")


def reload_tensor_to_gpu(tensors):
    """
    将给定的张量从CPU转移到GPU中，并返回新的张量。如果没有设置环境变量FLAGS_use_cuda_managed_memory为True，则此函数无效。

    Args:
        tensors (List[Tuple[Any, str]]): 包含两个元素的列表，第一个元素是需要转移到GPU的张量，第二个元素是字符串，用于指示张量类型（"optimizer"或"model"）。

    Returns:
        List[Tuple[Any, str]]: 与输入相同的列表，但所有张量已经被转移到GPU中。

    Raises:
        None.
    """
    if strtobool(os.getenv("FLAGS_use_cuda_managed_memory", "False")):
        logger.warning("FLAGS_use_cuda_managed_memory has been set to True, " "offloading strategy is ineffective.")
        return

    # optimizer
    if "optimizer" in tensors[1]:
        optimizer = tensors[0]
        # offload moment1
        for key, value in optimizer._accumulators[optimizer._moment1_acc_str].items():
            if value._is_initialized() and not isinstance(value.place, paddle.CUDAPlace):
                optimizer._accumulators[optimizer._moment1_acc_str][key] = value.cuda()

        # offload moment2
        for key, value in optimizer._accumulators[optimizer._moment2_acc_str].items():
            if value._is_initialized() and not isinstance(value.place, paddle.CUDAPlace):
                optimizer._accumulators[optimizer._moment2_acc_str][key] = value.cuda()

        # offload master_weight
        for key, value in optimizer._master_weights.items():
            if value._is_initialized() and not isinstance(value.place, paddle.CUDAPlace):
                optimizer._master_weights[key] = value.cuda()
    # model
    elif "model" in tensors[1]:
        model = tensors[0]
        device = paddle.device.get_device()
        for name, src in model.named_parameters():
            if src._is_initialized() and not isinstance(src.place, paddle.CUDAPlace):
                _move_param(src, device)
    else:
        logger.debug(f"Can't parse for type {tensors[1]}")


def cleanup_tensor_space(tensors):
    """
    释放张量所占的空间，包括内存和磁盘空间。如果输入是字典类型，则递归处理其中的值；如果是paddle.Tensor类型，则清除数据；否则返回原始对象。

    Args:
        tensors (Union[dict, paddle.Tensor]): 需要释放空间的张量或字典，其中字典的值为张量。

    Returns:
        Union[dict, paddle.Tensor]: 如果输入是字典，则返回一个新的字典，其中值已经被释放空间；如果输入是paddle.Tensor，则返回一个清除了数据的paddle.Tensor。否则返回原始对象。
    """
    if isinstance(tensors, dict):
        for _, v in tensors.items():
            cleanup_tensor_space(v)
    elif isinstance(tensors, paddle.Tensor):
        tensors._clear_data()
    else:
        logger.debug(f"Can't parse for type {type(tensors)}")
        return tensors


def data_group_split(tensors, group):
    """
    将数据按照给定的分组进行切分，如果没有给定分组则直接返回原始数据。
    支持列表、元组、字典和paddle.Tensor类型的数据。

    Args:
        tensors (Union[List[Any], Tuple[Any], Dict[str, Any], paddle.Tensor]): 待切分的数据，可以是任意类型。
        group (Optional[distributed.Group]): 指定要切分的分组，如果为None则直接返回原始数据。默认为None。

    Returns:
        Union[List[Any], Tuple[Any], Dict[str, Any], paddle.Tensor]: 切分后的数据，与输入数据类型一致。
        如果输入数据为字典，则返回的新字典中的值也会被切分。
    """
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
        logger.debug(f"Can't parse for type {type(tensors)}")
        return tensors


def data_group_merge(tensors, group):
    """
    将数据组合成一个新的列表或字典，如果不是None则在指定的分组中进行all_gather_nd操作。

    Args:
        tensors (Union[List[Any], Tuple[Any], Dict[str, Any], paddle.Tensor]): 需要组合的数据，可以是列表、元组、字典或张量。
            如果是张量，则会在指定的分组中进行all_gather_nd操作，并返回一个张量。
        group (Optional[int]): 指定的分组，如果为None，则直接返回原始数据。默认为None。

    Returns:
        Union[List[Any], Tuple[Any], Dict[str, Any], paddle.Tensor]: 返回一个新的列表或字典，或者一个张量，取决于传入的数据类型。
        如果是张量，则是在指定的分组中进行all_gather_nd操作后的结果。

    Raises:
        None
    """
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
        logger.debug(f"Can't parse for type {type(tensors)}")
        return tensors


def group_rank_guard(group, rank=0):
    """
    用于控制某个进程组中的某个进程是否参与函数调用，并在所有进程完成后进行通信。
    如果该进程组中的某个进程不是指定的rank，则不会调用该函数。

    Args:
        group (distributed.ProcessGroup): 进程组对象。
        rank (int, optional, default=0): 需要参与函数调用的进程的rank，默认为0。
            rank为-1时表示所有进程都参与。

    Returns:
        function: 返回一个装饰器，该装饰器接受一个函数作为参数，返回一个包装后的函数。
                  被装饰的函数将在指定的rank的进程中被调用，其他进程不会被调用。
                  在所有进程完成后，将进行通信，并广播结果到所有进程。
    """

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
    """
    对输入的批次进行重新填充，使得每个批次的长度都是最大长度。
    如果批次中包含了位置ID，则在未被访问到的部分填充为1。

    Args:
        batches (dict): 包含输入数据和其他信息的字典，格式为{"input_ids": Tensor, "attention_mask": Tensor, ...}。
            其中Tensor的形状应该是（batch_size, sequence_length）。
        input_lengths (Tensor): 一个长度为batch_size的张量，表示每个批次的实际长度。
            形状为（batch_size,）。

    Returns:
        dict: 返回一个更新后的字典，包含了重新填充后的输入数据和其他信息。
            如果原始批次中没有包含位置ID，那么这个字段将不会出现在返回值中。

    Raises:
        None
    """
    if batches.get("position_ids", None) is not None:
        v = batches["position_ids"]
        for x in range(v.shape[0]):
            v[x, input_lengths[x] :] = 1
        batches["position_ids"] = v
    for key in list(batches.keys()):
        if batches[key].shape[0] != input_lengths.shape[0]:
            batches[key] = batches[key].mean()

    return batches


def remove_input_padding(input_ids, pad_id):
    """
    从输入ID中移除填充，返回一个列表，每个元素是一个不包含pad_id的paddle.Tensor。

    Args:
        input_ids (List[paddle.Tensor]): 包含输入ID的列表，每个元素是一个1维的paddle.Tensor，dtype为int64。
        pad_id (int): 需要移除的填充ID。

    Returns:
        List[paddle.Tensor]: 包含不包含pad_id的输入ID的列表，每个元素是一个1维的paddle.Tensor，dtype为int64。
    """
    result = []
    for ids in input_ids:
        ids_list = ids.tolist()
        filtered_ids = [id for id in ids_list if id != pad_id]
        result.append(paddle.to_tensor(filtered_ids, dtype="int64"))
    return result


def concat_input_response_and_padding(input_ids_wo_padding, response, pad_id):
    """
    将输入和响应进行拼接，并添加适当的填充。

    Args:
        input_ids_wo_padding (List[Tensor]): 不包含填充的输入ID列表，形状为（batch_size，seq_len）。
        response (Tensor): 响应矩阵，形状为（num_return_index，batch_size，seq_len）。
        pad_id (int): 用于填充的ID。

    Returns:
        Tensor: 返回一个形状为（num_return_index，batch_size，max_seq_len）的Tensor，其中max_seq_len是所有输入和响应的最大长度。
        每个元素都是由input_ids_wo_padding和response的对应元素拼接而成的。如果拼接后的长度小于max_seq_len，则会在末尾追加pad_id。
    """
    concat_results = []
    max_seq_len = 0
    for num_return_index in range(response.shape[0]):
        batch_concat_input_response = []
        for batch_index in range(response.shape[1]):
            one_input = input_ids_wo_padding[batch_index]
            one_response = response[num_return_index][batch_index]
            one_concat_input_response = paddle.concat((one_input, one_response))
            max_seq_len = max(max_seq_len, one_concat_input_response.shape[0])
            batch_concat_input_response.append(one_concat_input_response)
        concat_results.append(batch_concat_input_response)

    padding_results = []
    for num_return_index in range(response.shape[0]):
        batch_padding_result = []
        for batch_index in range(response.shape[1]):
            difference = max_seq_len - concat_results[num_return_index][batch_index].shape[0]
            one_padding_result = concat_results[num_return_index][batch_index].tolist() + difference * [pad_id]
            batch_padding_result.append(paddle.to_tensor(one_padding_result, dtype="int64"))
        padding_results.append(batch_padding_result)

    return paddle.to_tensor(padding_results, dtype="int64")


# https://stackoverflow.com/questions/12594148/skipping-execution-of-with-block
class SkipWithBlock(Exception):
    pass


class SkipContextManager:
    def __init__(self, skip):
        """
            Initializes the class with the given skip value.

        Args:
            skip (int): The number of rows to skip in the input data.

        Returns:
            None.
        """
        self.skip = skip

    def __enter__(self):
        """
            在进入上下文管理器时调用，返回自身。
        如果需要执行一些初始化操作，可以重写此方法。

        Returns:
            TraceContextManager: 当前实例对象自身。
        """
        if self.skip:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        """
        跟踪函数执行，并在遇到指定的代码块时抛出SkipWithBlock异常。
            当前实现只支持单个代码块，不支持多个。

        Args:
            frame (types.FrameType): 当前执行的frame对象。
            event (str): 事件类型，包括'call', 'return', 'exception_raised', 'yield'.
            arg (Any): 可选参数，用于传递给event_handler函数。

        Raises:
            SkipWithBlock: 当遇到指定的代码块时抛出此异常，表示需要跳过后续的测试执行。
        """
        raise SkipWithBlock

    def __exit__(self, type, value, traceback):
        """
            如果退出时没有异常，则返回True。如果退出时是SkipWithBlock的子类，则返回True以抑制该异常。否则返回False。
        如果没有异常，则返回True。如果退出时是SkipWithBlock的子类，则返回True以抑制该异常。否则返回False。

        Args:
            type (Optional[Type[BaseException]]): 可选，异常类型，如果为None，则表示没有异常。默认为None。
            value (Optional[BaseException]): 可选，异常对象，如果type不为None，则必须提供value参数。默认为None。
            traceback (Optional[traceback]): 可选，追踪信息，如果type不为None，则必须提供traceback参数。默认为None。

        Returns:
            bool: 如果没有异常或者异常是SkipWithBlock的子类，则返回True；否则返回False。
        """
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
    tensor_dim = tensor.dim()
    if tensor_dim == 0:
        tensor = tensor.reshape([1])
        dist.all_gather(tensor_list, tensor, group=group)
        return tensor_list

    world_size = group.nranks
    local_size = paddle.to_tensor(tensor.shape, place=tensor.place)
    all_sizes = [paddle.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size, group=group)

    max_length = max(size[-1] for size in all_sizes)

    length_diff = max_length.item() - local_size[-1].item()
    if length_diff:
        if tensor_dim == 2:
            pad_size = (*tensor.shape[:-1], length_diff)
            padding = paddle.zeros(pad_size, dtype=tensor.dtype)
            tensor = paddle.concat([tensor, padding], axis=-1)
        elif tensor_dim == 4:
            # Note(gongenlei): support attention mask
            tensor = nn.Pad2D([0, length_diff, 0, length_diff], mode="constant", value=0.0)(tensor)

    all_tensors_padded = []
    tensor = tensor.contiguous()
    dist.all_gather(all_tensors_padded, tensor, group=group)
    # all_tensors = []
    if padded:
        tensor_list.extend(all_tensors_padded)
        return all_tensors_padded

    for tensor_, size in zip(all_tensors_padded, all_sizes):
        if tensor_dim == 2:
            tensor_list.append(tensor_[..., : size[-1]])
        elif tensor_dim == 4:
            tensor_list.append(tensor_[..., : size[-1], : size[-1]])
    return tensor_list


def export_evaluate_model(self: Trainer, train_model, eval_model, **kwargs):
    """
    导出评估模型。

    Args:
        self (Trainer, required):
            Trainer 对象的引用。

        train_model (nn.Layer, required):
            Train 模型，需要在训练过程中使用。

        eval_model (Optional[nn.Layer], optional):
            评估模型，如果没有提供，则返回 None。默认为 None。

        with_offload (bool, optional):
            是否将训练模型的张量转换到 CPU 上，默认为 False。

        kwargs (Dict, optional):
            可选参数字典，包括：
            - with_offload (bool, optional):
                是否将训练模型的张量转换到 CPU 上，默认为 False。

    Returns:
        Optional[None]:
            如果 eval_model 不存在，则返回 None；否则返回 None。

    Raises:
        ValueError:
            当 eval_model 的 tensor_parallel_degree 与 train_model 的 tensor_parallel_degree 不相同时，会引发此错误。
    """
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
                        offload_tensor_to_cpu((train_state_dict[key], "tensor"))
            else:
                # single to single
                # tp+pp -> single
                raise ValueError("Not support yet.")

        def create_send_recv_table(train_keys, eval_keys, is_value_trainer):
            recv_table = []
            send_table = []
            if pp_group.rank == 0:
                for key in eval_keys:
                    if (not eval_model.config.weight_sharing) and is_value_trainer:
                        if "output_linear.out_linear" in key:
                            logger.debug(f"Skip: {key}")
                            continue
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
        # self.timers and self.timers("export-merge-pp").start()
        if eval_tp_size > 1 and train_pp_size > 1:
            table = create_send_recv_table(
                train_state_dict.keys(),
                eval_state_dict.keys(),
                self.trainer_type == "value",
            )

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
                    offload_tensor_to_cpu((train_state_dict[key], "tensor"))

        # self.timers and self.timers("export-merge-pp").stop()
        # self.timers and self.timers("export-broadcast-pp").start()
        if pp_group.nranks > 1:
            paddle.distributed.parallel.sync_params_buffers(
                eval_model,
                comm_group=pp_group,
                src_rank=pp_group.ranks[0],
                fuse_params=False,
            )
        # self.timers and self.timers("export-broadcast-pp").stop()
    else:
        # 其他 DP rank 的state dict, 适配 offload 和初始化
        # self.timers and self.timers("export-offload-and-init").start()
        if with_offload:
            for key in list(train_state_dict.keys()):
                offload_tensor_to_cpu((train_state_dict[key], "tensor"))
        for k, v in eval_state_dict.items():
            if not v._is_initialized():
                t = paddle._C_ops.full_like(v, 0, v.dtype, paddle.CUDAPlace(global_dev_id))
                v.get_tensor()._share_data_with(t.get_tensor())
        # self.timers and self.timers("export-offload-and-init").stop()

    paddle.distributed.barrier()
    # self.timers and self.timers("export-broadcast-sd-dp").start()
    if eval_tp_size == 1:
        for _, tensor in eval_state_dict.items():
            paddle.distributed.broadcast(tensor, src=0, group=None, sync_op=True)
    else:
        if sd_group.nranks > 1:
            if dp_group.rank <= 0:
                paddle.distributed.parallel.sync_params_buffers(
                    eval_model,
                    comm_group=sd_group,
                    src_rank=sd_group.ranks[0],
                    fuse_params=False,
                )
        if dp_group.nranks > 1:
            paddle.distributed.parallel.sync_params_buffers(
                eval_model,
                comm_group=dp_group,
                src_rank=dp_group.ranks[0],
                fuse_params=False,
            )
    # self.timers and self.timers("export-broadcast-sd-dp").stop()

    old_dp_workers = self.args.world_size // (max(sd_group.nranks, 1) * max(dp_group.nranks, 1))
    group_nums = self.args.logical_process_index // old_dp_workers * eval_tp_size + eval_tp_rank

    if not hasattr(self, "_policy_model_eval_group") or self._policy_model_eval_group is None:
        self._policy_model_eval_group = create_data_trans_group(global_rank, group_nums)

    return None


def create_data_trans_group(global_rank, group_nums):
    """
    创建一个数据传输组，该组将根据给定的全局排名和组数分割。
    该函数使用了paddle.distributed.all_gather_object进行通信，并返回一个新的分布式组对象。

    Args:
        global_rank (int): 当前全局排名。
        group_nums (List[int]): 需要分割的组数列表。

    Returns:
        paddle.distributed.Group: 返回一个新的分布式组对象，包含所有参与分割的全局排名。如果当前全局排名在任何一个组中，则返回该组。如果当前全局排名不在任何一个组中，则返回None。
    """
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


def new_timer_log(self, names, normalizer=1.0, reset=True):
    """Log a group of timers."""

    def format_dict(data):
        """Format the timer log."""
        result = {}
        order = []
        for key, value in data.items():
            category, detail = key.split(" ", maxsplit=1)
            if category not in result:
                result[category] = []
                order.append(category)
            result[category].append(f"{detail}: {round(value, 2)}")

        output = ""
        for category in order:
            if category in result:
                output += f"\n{category}"
                for value in result[category]:
                    output += f"\n  {value}"
        return output

    assert normalizer > 0.0
    string = "time (ms)"
    names = sorted(names)
    time_dict = {}
    for name in names:
        time_dict[name] = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
    if len(time_dict) == 0:
        return "skipped"
    string += format_dict(time_dict)
    return string


Trainer.export_evaluate_model = export_evaluate_model
