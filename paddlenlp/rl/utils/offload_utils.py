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

from ...trainer.argparser import strtobool
from ...trainer.trainer import logger


@paddle.no_grad()
def _move_param(src, device=None, blocking=True):
    """
    Move parameters from the source device to the target device and return the parameters on the target device.
    If the target device is not specified, the current device is used.

    Args:
        src (Tensor): The tensor of parameters to be moved.
        device (Optional[Union[str, paddle.Device]], optional): The target device. Can be a string or paddle.Device object.
            Defaults to None, which means using the current device.
        blocking (bool, optional): Whether to block until the operation is complete. Defaults to True.

    Returns:
        Tensor: The tensor of parameters on the target device.
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
    Migrate the given tensors to CPU. This function has no effect if CUDA managed memory is used.

    Args:
        tensors (tuple, list): A tuple or list containing two elements. The first element is the model or optimizer,
            and the second element is a string indicating whether it is a model or optimizer.

    Returns:
        None: No return value, modifies the original tensors directly.

    Raises:
        None: Does not raise any exceptions.
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
        logger.debug(f"[offload_tensor_to_cpu]Can't parse for type {tensors[1]}")


def reload_tensor_to_gpu(tensors):
    """
    Transfer the given tensors from CPU to GPU and return new tensors. This function has no effect if the environment variable
    FLAGS_use_cuda_managed_memory is not set to True.

    Args:
        tensors (List[Tuple[Any, str]]): A list containing tuples. Each tuple has two elements: the tensor to be transferred
            to GPU and a string indicating the tensor type ("optimizer" or "model").

    Returns:
        List[Tuple[Any, str]]: The same list as the input, but all tensors have been transferred to GPU.

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
        logger.debug(f"[reload_tensor_to_gpu]Can't parse for type {tensors[1]}")


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

    objs = [(arg, offload_map.get(arg, "")) for arg in args if offload_map.get(arg, "") in trainer.args.offload_level]
    if trainer.actor_model not in [i for i, _ in objs]:
        if getattr(trainer.actor_trainer, "_inner_eval_model", None) is not None:
            # NOTE(gongenlei): for export_evaluate_model
            objs.append((trainer.actor_model, offload_map.get(trainer.actor_model, "")))
    if trainer.args.rl_algorithm == "ppo":
        if trainer.reward_critic_model not in [i for i, _ in objs]:
            if getattr(trainer.critic_trainer, "_inner_eval_model", None) is not None:
                # NOTE(gongenlei): for export_evaluate_model
                objs.append((trainer.reward_critic_model, offload_map.get(trainer.reward_critic_model, "")))
    return OffloadController(objs)
