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
from enum import Enum, auto

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet

from ...trainer.trainer import Trainer, logger
from ...utils.nested import flatten_list, nested_broadcast_tensor_with_empty
from .reshard_utils import init_reshard_mappings, init_rollout_env, reshard_to_rollout

global_dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device().split(":")[1])


class ActorStages(Enum):
    """
    Enum class, the stages of the actor training process.
    """

    MODEL_ENABLE_DISABLE = auto()
    RL_STEP = auto()
    MICRO_STEPS = auto()
    OPTIMIZE_STEP = auto()


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
    ROLLOUT_OLD_LOGPROB = auto()
    ROLLOUT_REF_LOGPROB = auto()
    REWARD_MODEL_ENABLE_DISABLE = auto()
    ROLLOUT_REWARD_VALUE = auto()
    ROLLOUT_ADVANTAGE = auto()


def get_timer_label(stage: Enum) -> str:
    """
    Get the timer label.

    Args:
        stage (Enum): RolloutStages/CriticStages/RolloutStages.

    Returns:
        str: The prefix when printing the Timer. Format is "[prefix] stage number.description".
            - prefix: Stage prefix, e.g., "actor-step", "critic-step".
            - stage number: Numbered from 1.
            - description: Stage description in lowercase.
    """
    step_prefix = {
        ActorStages.MODEL_ENABLE_DISABLE: "actor-step",
        ActorStages.RL_STEP: "actor-step",
        ActorStages.MICRO_STEPS: "actor-step",
        ActorStages.OPTIMIZE_STEP: "actor-step",
        CriticStages.MODEL_ENABLE_DISABLE: "critic-step",
        CriticStages.CRITIC_TRAINING_STEP: "critic-step",
        RolloutStages.ACTOR_MODEL_ENABLE_DISABLE: "rollout",
        RolloutStages.GENERATE: "rollout",
        RolloutStages.ROLLOUT_LOGPROB: "rollout",
        RolloutStages.ROLLOUT_OLD_LOGPROB: "rollout",
        RolloutStages.ROLLOUT_REF_LOGPROB: "rollout",
        RolloutStages.ROLLOUT_ADVANTAGE: "rollout",
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


def cleanup_tensor_space(tensors):
    """
    Release the space occupied by tensors, including memory and disk space.
    If the input is a dictionary, recursively process its values;
    if it is a paddle.Tensor, clear the data; otherwise, return the original object.

    Args:
        tensors (Union[dict, paddle.Tensor]): Tensors or dictionary to release space, where the values of the dictionary are tensors.

    Returns:
        Union[dict, paddle.Tensor]: If the input is a dictionary, return a new dictionary with values having their space released;
        if the input is a paddle.Tensor, return a paddle.Tensor with data cleared. Otherwise, return the original object.
    """
    if isinstance(tensors, dict):
        for _, v in tensors.items():
            cleanup_tensor_space(v)
    elif isinstance(tensors, paddle.Tensor):
        tensors._clear_data()
    else:
        logger.debug(f"[cleanup_tensor_space]Can't parse for type {type(tensors)}")
        return tensors


def data_group_split(tensors, group):
    """
    Split data according to the given group. If no group is given, return the original data.
    Supports list, tuple, dictionary, and paddle.Tensor types of data.

    Args:
        tensors (Union[List[Any], Tuple[Any], Dict[str, Any], paddle.Tensor]): Data to be split, can be any type.
        group (Optional[distributed.Group]): The group to split by, if None, return the original data. Default is None.

    Returns:
        Union[List[Any], Tuple[Any], Dict[str, Any], paddle.Tensor]: Split data, consistent with the input data type.
        If the input data is a dictionary, the values in the returned new dictionary will also be split.
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
        logger.debug(f"[data_group_split]Can't parse for type {type(tensors)}")
        return tensors


def data_group_merge(tensors, group):
    """
    Combine data into a new list or dictionary, or perform all_gather_nd operation in the specified group if not None.

    Args:
        tensors (Union[List[Any], Tuple[Any], Dict[str, Any], paddle.Tensor]): Data to be combined, can be list, tuple, dictionary, or tensor.
            If it is a tensor, an all_gather_nd operation will be performed in the specified group, and a tensor will be returned.
        group (Optional[int]): The specified group, if None, return the original data. Default is None.

    Returns:
        Union[List[Any], Tuple[Any], Dict[str, Any], paddle.Tensor]: Return a new list or dictionary, or a tensor, depending on the input data type.
        If it is a tensor, it is the result of the all_gather_nd operation in the specified group.

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
    elif isinstance(tensors, np.ndarray):
        tensor_list = []
        all_gather_nd(tensor_list, tensors, group=group, padded=True)
        return np.concatenate(tensor_list)
    else:
        logger.debug(f"[data_group_merge]Can't parse for type {type(tensors)}")
        return tensors


def group_rank_guard(group, rank=0):
    """
    Control whether a process in a process group participates in a function call and communicate after all processes are done.
    If a process in the process group is not the specified rank, the function will not be called.

    Args:
        group (distributed.ProcessGroup): Process group object.
        rank (int, optional, default=0): The rank of the process that needs to participate in the function call, default is 0.
            When rank is -1, all processes participate.

    Returns:
        function: Returns a decorator that accepts a function as an argument and returns a wrapped function.
                  The decorated function will be called in the specified rank process, and other processes will not be called.
                  After all processes are done, communication will be performed, and the results will be broadcast to all processes.
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
    Repad the input batches so that the length of each batch is the maximum length.
    If the batch contains position IDs, fill the unaccessed parts with 1.

    Args:
        batches (dict): A dictionary containing input data and other information, formatted as {"input_ids": Tensor, "attention_mask": Tensor, ...}.
            The shape of the Tensor should be (batch_size, sequence_length).
        input_lengths (Tensor): A tensor of length batch_size, indicating the actual length of each batch.
            Shape is (batch_size,).

    Returns:
        dict: Returns an updated dictionary containing the repadded input data and other information.
            If the original batch does not contain position IDs, this field will not appear in the return value.

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
    Remove padding from input IDs and return a list, where each element is a paddle.Tensor without pad_id.

    Args:
        input_ids (List[paddle.Tensor]): A list containing input IDs, each element is a 1D paddle.Tensor with dtype int64.
        pad_id (int): The padding ID to be removed.

    Returns:
        List[paddle.Tensor]: A list containing input IDs without pad_id, each element is a 1D paddle.Tensor with dtype int64.
    """
    result = []
    for ids in input_ids:
        ids_list = ids.tolist()
        filtered_ids = [id for id in ids_list if id != pad_id]
        result.append(paddle.to_tensor(filtered_ids, dtype="int64"))
    return result


def concat_input_response_and_padding(input_ids_wo_padding, response, pad_id):
    """
    Concatenate input and response with appropriate padding.

    Args:
        input_ids_wo_padding (List[Tensor]): List of input IDs without padding, shape (batch_size, seq_len).
        response (Tensor): Response matrix, shape (num_return_index, batch_size, seq_len).
        pad_id (int): ID used for padding.

    Returns:
        Tensor: Returns a Tensor of shape (num_return_index, batch_size, max_seq_len), where max_seq_len is the maximum length of all inputs and responses.
        Each element is concatenated from input_ids_wo_padding and the corresponding element of response.
        If the concatenated length is less than max_seq_len, pad_id will be appended at the end.
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
        Called when entering the context manager, returns self.
        If initialization operations are needed, this method can be overridden.

        Returns:
            SkipContextManager: The current instance of the object.
        """
        if self.skip:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        """
        Traces function execution and raises a SkipWithBlock exception when encountering the specified code block.
        Current implementation only supports a single code block, not multiple.

        Args:
            frame (types.FrameType): The current executing frame object.
            event (str): The event type, including 'call', 'return', 'exception_raised', 'yield'.
            arg (Any): Optional argument passed to the event_handler function.

        Raises:
            SkipWithBlock: Raised when encountering the specified code block, indicating that subsequent test execution should be skipped.
        """
        raise SkipWithBlock

    def __exit__(self, type, value, traceback):
        """
        If no exception is present when exiting, returns True. If the exception is a subclass of SkipWithBlock, returns True to suppress the exception. Otherwise, returns False.

        Args:
            type (Optional[Type[BaseException]]): Optional, the exception type. If None, indicates no exception. Default is None.
            value (Optional[BaseException]): Optional, the exception object. If type is not None, value must be provided. Default is None.
            traceback (Optional[traceback]): Optional, traceback information. If type is not None, traceback must be provided. Default is None.

        Returns:
            bool: Returns True if no exception is present or the exception is a subclass of SkipWithBlock; otherwise, returns False.
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
    if isinstance(tensor, paddle.Tensor):
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
            if tensor_dim == 1:
                tensor = paddle.concat([tensor, paddle.zeros([length_diff], dtype=tensor.dtype)])
            elif tensor_dim == 2:
                pad_size = (*tensor.shape[:-1], length_diff)
                padding = paddle.zeros(pad_size, dtype=tensor.dtype)
                tensor = paddle.concat([tensor, padding], axis=-1)
            elif tensor_dim == 4:
                # Note(gongenlei): support attention mask(not used)
                tensor = nn.Pad2D([0, length_diff, 0, length_diff], mode="constant", value=0.0)(tensor)

        all_tensors_padded = []
        tensor = tensor.contiguous()
        dist.all_gather(all_tensors_padded, tensor, group=group)
        # all_tensors = []
        if padded:
            tensor_list.extend(all_tensors_padded)
            return all_tensors_padded

        for tensor_, size in zip(all_tensors_padded, all_sizes):
            if tensor_dim == 1:
                tensor_list.append(tensor_[: size[-1]])
            elif tensor_dim == 2:
                tensor_list.append(tensor_[..., : size[-1]])
            elif tensor_dim == 4:
                tensor_list.append(tensor_[..., : size[-1], : size[-1]])
        return tensor_list
    elif isinstance(tensor, np.ndarray):
        dist.all_gather_object(tensor_list, tensor, group=group)
    else:
        logger.debug(f"[all_gather_nd]Can't parse for type {type(tensor)}")


def export_evaluate_model(self: Trainer, train_model, eval_model, **kwargs):
    """
    Export the evaluation model.

    Args:
        self (Trainer, required):
            Reference to the Trainer object.

        train_model (nn.Layer, required):
            The training model to be used during training.

        eval_model (Optional[nn.Layer], optional):
            The evaluation model. If not provided, returns None. Default is None.

        with_offload (bool, optional):
            Whether to offload the tensors of the training model to CPU. Default is False.

        kwargs (Dict, optional):
            A dictionary of optional parameters, including:
            - with_offload (bool, optional):
                Whether to offload the tensors of the training model to CPU. Default is False.

    Returns:
        Optional[None]:
            Returns None if eval_model does not exist; otherwise, returns None.

    Raises:
        ValueError:
            Raised when the tensor_parallel_degree of eval_model is different from that of train_model.
    """
    if eval_model is None:
        return None

    hcg = fleet.get_hybrid_communicate_group()
    pp_group = hcg.get_pipe_parallel_group()
    tp_group = hcg.get_model_parallel_group()
    sd_group = hcg.get_sharding_parallel_group()
    dp_group = hcg.get_data_parallel_group()
    pp_rank = hcg.get_stage_id()

    if not hasattr(self, "global_meta_dict") or self.global_meta_dict is None:
        self.global_meta_dict = init_reshard_mappings(train_model, self.args, pp_rank, pp_group)

    with init_rollout_env(self.args.rollout_tensor_parallel_degree):
        hcg = fleet.get_hybrid_communicate_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
        tensor_parallel_rank = hcg.get_model_parallel_rank()
        eval_tp_size = max(tensor_parallel_degree, 1)
        eval_tp_rank = max(tensor_parallel_rank, 0)
        reshard_to_rollout(
            train_model, eval_model, self.global_meta_dict, pp_rank, pp_group, hcg.get_model_parallel_group(), tp_group
        )

    old_dp_workers = self.args.world_size // (max(sd_group.nranks, 1) * max(dp_group.nranks, 1))
    group_nums = self.args.logical_process_index // old_dp_workers * eval_tp_size + eval_tp_rank

    if not hasattr(self, "_policy_model_eval_group") or self._policy_model_eval_group is None:
        self._policy_model_eval_group = create_data_trans_group(paddle.distributed.get_rank(), group_nums)

    return None


def create_data_trans_group(global_rank, group_nums):
    """
    Create a data transfer group that is partitioned based on the given global rank and number of groups.
    This function uses paddle.distributed.all_gather_object for communication and returns a new distributed group object.

    Args:
        global_rank (int): The current global rank.
        group_nums (List[int]): A list of group numbers to partition.

    Returns:
        paddle.distributed.Group: Returns a new distributed group object containing all global ranks participating in the partition.
            If the current global rank is in any of the groups, it returns that group. If the current global rank is not in any of the groups, it returns None.
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


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=None) / mask.sum(axis=None)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * paddle.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def gather_and_pad(tensor, dp_group, sd_group, pad_index=0.0, pad=True):
    """Gather tensor from all devices."""
    if not isinstance(tensor, list):
        tensor = [tensor]
    dtype = tensor[0].dtype

    def pad_tensor(tensor_list):
        max_size = max([i.shape[-1] for i in tensor_list])
        data_num = sum([i.shape[0] for i in tensor_list])
        new_tensor = paddle.full((data_num, max_size), pad_index, dtype=dtype)
        offset = 0
        for idx, i in enumerate(tensor_list):
            new_tensor[offset : offset + i.shape[0], : i.shape[-1]] = i
            offset += i.shape[0]
        return new_tensor

    if dp_group.nranks == 1 and sd_group.nranks == 1:
        if not pad:
            return paddle.concat(tensor, axis=0)
        else:
            return pad_tensor(tensor)

    def map_func(weight):
        if isinstance(weight, paddle.Tensor):
            weight = weight.numpy()
        return weight

    tensor = [map_func(i) for i in tensor]

    sd_gathered_tensor = []
    if sd_group.nranks > 1:
        dist.all_gather_object(sd_gathered_tensor, tensor, group=sd_group)

    dp_gathered_tensor = []
    if dp_group.nranks > 1:
        if len(sd_gathered_tensor) > 0:
            tensor = sd_gathered_tensor
        dist.all_gather_object(dp_gathered_tensor, tensor, group=dp_group)

    if len(dp_gathered_tensor) > 0:
        gathered_tensor = dp_gathered_tensor
    else:
        gathered_tensor = sd_gathered_tensor

    gathered_tensor = [paddle.to_tensor(i, dtype=dtype) for i in flatten_list(gathered_tensor)]

    if not pad:
        return paddle.concat(gathered_tensor, axis=0)
    else:
        return pad_tensor(gathered_tensor)
