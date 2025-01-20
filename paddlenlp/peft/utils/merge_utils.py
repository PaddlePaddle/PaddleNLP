# Copyright 2024-present the HuggingFace Inc. team.
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

import warnings
from typing import List, Literal

import paddle


def reshape_weight_task_tensors(task_tensors, weights):
    """
    Reshapes `weights` to match the shape of `task_tensors` by unsqeezing in the remaining dimenions.

    Args:
        task_tensors (`paddle.Tensor`): The tensors that will be used to reshape `weights`.
        weights (`paddle.Tensor`): The tensor to be reshaped.

    Returns:
        `paddle.Tensor`: The reshaped tensor.
    """
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    weights = weights.view(new_shape)
    return weights


def magnitude_based_pruning(tensor: paddle.Tensor, density: float) -> paddle.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction
    `density`.

    Args:
        tensor (`paddle.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `paddle.Tensor`: The tensor with the pruned weights.
    """
    mask = paddle.zeros_like(tensor).reshape(-1)
    k = int(density * tensor.numel())
    top_k = paddle.topk(tensor.abs().reshape(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.reshape(tensor.shape)


def random_pruning(tensor: paddle.Tensor, density: float, rescale: bool) -> paddle.Tensor:
    """
    Prune random values based on the specified fraction `density`.

    Args:
        tensor (`paddle.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `paddle.Tensor`: The pruned tensor.
    """
    mask = paddle.bernoulli(paddle.full_like(input=tensor, fill_value=density))
    pruned_tensor = tensor * mask
    if rescale:
        paddle.div(input=pruned_tensor, other=density)
    return pruned_tensor


def prune(
    tensor: paddle.Tensor, density: float, method: Literal["magnitude", "random"], rescale: bool = False
) -> paddle.Tensor:
    """
    Prune the values of task tensors based on the `method`.

    Args:
        tensor (`paddle.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        method (`str`):The method to use to prune. Should be one of ["magnitude", "random"].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `paddle.Tensor`: The pruned tensor.
    """
    if density >= 1:
        warnings.warn(f"The density {density} is greater than or equal to 1, no pruning will be performed.")
        return tensor
    elif density < 0:
        raise ValueError(f"Density should be >= 0, got {density}")
    if method == "magnitude":
        return magnitude_based_pruning(tensor, density)
    elif method == "random":
        return random_pruning(tensor, density, rescale=rescale)
    else:
        raise ValueError(f"Unknown method {method}")


def calculate_majority_sign_mask(
    tensor: paddle.Tensor, method: Literal["total", "frequency"] = "total"
) -> paddle.Tensor:
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
        tensor (`paddle.Tensor`):The tensor to get the mask from.
        method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].

    Returns:
        `paddle.Tensor`: The majority sign mask.
    """

    sign = tensor.sign()
    if method == "total":
        sign_magnitude = tensor.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = paddle.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors: paddle.Tensor, majority_sign_mask: paddle.Tensor) -> paddle.Tensor:
    """
    Merge the task tensors using disjoint merge.

    Args:
        task_tensors (`paddle.Tensor`):The task tensors to merge.
        majority_sign_mask (`paddle.Tensor`):The mask of the majority sign across the task tensors.

    Returns:
        `paddle.Tensor`: The merged tensor.
    """
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / paddle.clamp(num_params_preserved, min=1.0)


def task_arithmetic(task_tensors: List[paddle.Tensor], weights: paddle.Tensor) -> paddle.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[paddle.Tensor]`):The task tensors to merge.
        weights (`paddle.Tensor`):The weights of the task tensors.

    Returns:
        `paddle.Tensor`: The merged tensor.
    """
    task_tensors = paddle.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def magnitude_prune(task_tensors: List[paddle.Tensor], weights: paddle.Tensor, density: float) -> paddle.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[paddle.Tensor]`):The task tensors to merge.
        weights (`paddle.Tensor`):The weights of the task tensors.
        density (`float`): The fraction of values to preserve. Should be in [0,1].

    Returns:
        `paddle.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
    task_tensors = paddle.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def ties(
    task_tensors: List[paddle.Tensor],
    weights: paddle.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> paddle.Tensor:
    """
    Merge the task tensors using `ties`.

    Args:
        task_tensors(`List[paddle.Tensor]`):The task tensors to merge.
        weights (`paddle.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `paddle.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
    task_tensors = paddle.stack(task_tensors, dim=0)
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors


def dare_linear(task_tensors: List[paddle.Tensor], weights: paddle.Tensor, density: float) -> paddle.Tensor:
    """
    Merge the task tensors using `dare linear`.

    Args:
        task_tensors(`List[paddle.Tensor]`):The task tensors to merge.
        weights (`paddle.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `paddle.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="random", rescale=True) for tensor in task_tensors]
    task_tensors = paddle.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def dare_ties(
    task_tensors: List[paddle.Tensor],
    weights: paddle.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> paddle.Tensor:
    """
    Merge the task tensors using `dare ties`.

    Args:
        task_tensors(`List[paddle.Tensor]`):The task tensors to merge.
        weights (`paddle.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `paddle.Tensor`: The merged tensor.
    """
    # sparsify
    task_tensors = [prune(tensor, density, method="random", rescale=True) for tensor in task_tensors]
    task_tensors = paddle.stack(task_tensors, dim=0)
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors
