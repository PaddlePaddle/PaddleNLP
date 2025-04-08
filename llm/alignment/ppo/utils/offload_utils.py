# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.trainer.trainer import logger


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
        logger.warning("FLAGS_use_cuda_managed_memory has been set to True, offloading strategy is ineffective.")
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
        logger.warning("FLAGS_use_cuda_managed_memory has been set to True, offloading strategy is ineffective.")
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


class OffloadController:
    def __init__(self, objs):
        self.objs = objs

    def __enter__(self):
        for obj in self.objs:
            if hasattr(obj[0], "enable"):
                obj[0].enable()
            else:
                if obj[1] != "":
                    reload_tensor_to_gpu(obj)
        # offload_tensor_to_cpu/reload_tensor_to_gpu use non-blocking copy
        # maybe overlap with compute later
        if len(self.objs) > 0:
            paddle.device.synchronize()

    def __exit__(self, *args):
        for obj in self.objs:
            if hasattr(obj[0], "disable"):
                obj[0].disable()
            else:
                if obj[1] != "":
                    offload_tensor_to_cpu(obj)
        # offload_tensor_to_cpu/reload_tensor_to_gpu use non-blocking copy
        # maybe overlap with compute later
        if len(self.objs) > 0:
            paddle.device.synchronize()


def reload_and_offload_scope(trainer, *args):
    offload_map = {
        trainer.actor_model: "train_model",
        trainer.reference_model: "freeze_model",
        **({trainer.reward_model: "freeze_model"} if not trainer.args.use_rm_server else {}),
        trainer.actor_trainer.optimizer: "optimizer",
    }

    if trainer.args.rl_algorithm == "ppo":
        offload_map.update(
            {
                trainer.reward_critic_model: "train_model",
                trainer.critic_trainer.optimizer: "optimizer",
            }
        )

    if getattr(trainer.actor_trainer, "_inner_eval_model", None) is not None:
        offload_map.update({trainer.actor_trainer._inner_eval_model: "freeze_model"})

    if trainer.args.rl_algorithm == "ppo" and getattr(trainer.critic_trainer, "_inner_eval_model", None) is not None:
        offload_map.update({trainer.critic_trainer._inner_eval_model: "freeze_model"})

    objs = [(arg, offload_map.get(arg, "")) for arg in args if offload_map.get(arg, "") in trainer.args.offload_level]
    return OffloadController(objs)
