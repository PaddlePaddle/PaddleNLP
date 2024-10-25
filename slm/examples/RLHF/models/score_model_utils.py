# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
"""Utilities for score models."""

from __future__ import annotations

import importlib
import io
import json
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import paddle
import paddle.distributed as dist
import paddle.nn as nn

from paddlenlp.transformers import PretrainedConfig
from paddlenlp.transformers.auto.modeling import (
    MAPPING_NAMES,
    _BaseAutoModelClass,
    get_init_configurations,
    get_name_mapping,
    is_standard_config,
)
from paddlenlp.transformers.model_outputs import ModelOutput


class AutoModelForScore(_BaseAutoModelClass):
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    # "ForScore" might be more consistent with other tasks, while this suffix is
    # consistent with Beaver now.
    _task_suffix = "ModelForScore"
    _name_mapping: OrderedDict[str, Any] = get_name_mapping(_task_suffix)
    _score_module_name: str = "models.score_model"

    @classmethod
    def _get_model_class_from_config(cls, pretrained_model_name_or_path, config_file_path, config=None):
        if config is None:
            with io.open(config_file_path, encoding="utf-8") as f:
                config = json.load(f)

        # Get class name corresponds to this configuration
        if is_standard_config(config):
            architectures = config["architectures"]
            init_class = architectures.pop() if len(architectures) > 0 else None
        else:
            init_class = config.pop("init_class", None)

        # Get class name corresponds to this task, since we might init Score
        # model with CausalLM model.
        init_class = init_class[:-5] if init_class is not None and init_class.endswith("Model") else init_class
        model_name = None
        if init_class:
            for model_flag, name in MAPPING_NAMES.items():
                if model_flag in init_class:
                    model_name = model_flag + "Model"
                    break
        if model_name is None:
            raise AttributeError(
                f"Unable to parse 'architectures' or 'init_class' from {config_file_path}. Also unable to infer model class from 'pretrained_model_name_or_path'"
            )
        init_class = cls._name_mapping[model_name + "_Import_Class"]
        # module_name = cls._name_mapping[init_class]
        module_name = cls._score_module_name

        import_class = importlib.import_module(module_name)
        model_class = getattr(import_class, init_class)
        return model_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


@dataclass
class ScoreModelOutput(ModelOutput):
    """
    Output of the score model.

    Args:
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim, sequence_length)`):
            Prediction scores of the score model.
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim)`):
            Prediction scores of the end of the sequence.
    """

    scores: paddle.Tensor | None = None  # size = (B, L, D)
    end_scores: paddle.Tensor | None = None  # size = (B, D)


class ScoreModelMixin:
    """Base class for score models."""

    score_head: nn.Linear
    normalizer: Normalizer
    do_normalize: bool = False
    normalize_function: NormalizeFunction = "affine"
    _initialized: bool = False

    def init_score_head(self, config: PretrainedConfig, hidden_size: int, **kwargs: Any) -> None:
        """Initialize the score head."""
        if self._initialized:
            return

        config.score_dim = kwargs.pop("score_dim", getattr(config, "score_dim", 1))
        config.bias = kwargs.pop("bias", getattr(config, "bias", False))

        config.score_type = kwargs.pop("score_type", getattr(config, "score_type", "reward"))
        if config.score_type == "reward":
            self.normalize_function = "affine"
        elif config.score_type == "cost":
            self.normalize_function = "scale"
        elif config.score_type == "critic":
            self.normalize_function = "identity"
        else:
            raise ValueError(
                f"Invalid score type: {config.score_type}. Expected one of 'reward', 'cost', or 'critic'.",
            )

        config.do_normalize = kwargs.pop(
            "do_normalize",
            getattr(config, "do_normalize", False),
        )
        self.do_normalize = config.do_normalize

        config.normalizer_type = kwargs.pop(
            "normalizer_type",
            getattr(config, "normalizer_type", None),
        )
        if config.normalizer_type not in {"RunningMeanStd", "ExponentialMovingAverage", None}:
            raise ValueError(
                f"Invalid norm type: {config.normalizer_type}."
                "Expected one of 'RunningMeadStd', 'ExponentialMovingAverage', or None.",
            )
        if config.normalizer_type == "ExponentialMovingAverage":
            config.momentum = kwargs.pop("momentum", getattr(config, "momentum", None))
        momentum = getattr(config, "momentum", None)

        self.score_head = nn.Linear(hidden_size, config.score_dim, bias_attr=config.bias)
        self.normalizer = Normalizer.instantiate(
            normalizer_type=config.normalizer_type,
            normalize_function=self.normalize_function,
            shape=(config.score_dim,),
            momentum=momentum,
        )

        mean = getattr(config, "mean", None)
        var = getattr(config, "var", None)
        self.normalizer.set_mean_var(mean, var)

        self._initialized = True

    def get_score(
        self,
        hidden_state: paddle.Tensor,  # size = (B, L, E)
        attention_mask: paddle.Tensor | None = None,  # size = (B, L)
        position_ids: paddle.Tensor | None = None,  # size = (B, L)
        return_dict: bool | None = None,
    ) -> ScoreModelOutput:
        """Forward pass of the score model."""
        scores = self.score_head(hidden_state)  # size = (B, L, D)

        if position_ids is not None:
            first_pos = paddle.arange(hidden_state.shape[0]).unsqueeze(-1)
            # Take left padding into account, which has 0s in left and max_len
            # in right.
            left_pad_mask = position_ids == 0
            # position_ids = paddle.where(
            #     left_pad_mask, position_ids, position_ids + left_pad_mask.sum(-1, keepdim=True) - 1
            # )
            # the above limits right padding must not be 0s, the following suits
            # to both left and right padding with 0s
            left_pad_num = (
                paddle.where(left_pad_mask, position_ids.shape[-1] + 100, position_ids).argmin(axis=-1, keepdim=True)
                - 1
            )
            position_ids = left_pad_num + position_ids
            second_pos = paddle.max(position_ids, axis=-1, keepdim=True)
            end_pos = paddle.stack([first_pos, second_pos], axis=-1).squeeze(1)
            end_score = scores.gather_nd(end_pos)
        else:
            # attention_mask passed from pipeline pre-stage is shaped (bs, 1, seq_len, seq_len)
            assert attention_mask is not None and len(attention_mask.shape) == 2
            end_score = []
            end_pos = []
            for i in range(hidden_state.shape[0]):
                end_index = attention_mask[i].nonzero()[-1].item()
                end_pos.append((i, end_index))
                end_score.append(scores[i, end_index])  # size = (D,)
            end_score = paddle.stack(end_score, axis=0)  # size = (B, D)

        if self.training and self.do_normalize:

            if dist.is_initialized():
                gathered_end_score_list = []
                try:
                    # gather among data parallel group
                    hcg = dist.fleet.get_hybrid_communicate_group()
                    group = hcg.get_sharding_parallel_group()
                    dist.all_gather(gathered_end_score_list, end_score, group)
                except:
                    dist.all_gather(gathered_end_score_list, end_score)
                gathered_end_score = paddle.concat(gathered_end_score_list, axis=0)
                self.normalizer.update(gathered_end_score)
            else:
                self.normalizer.update(end_score)
            self.config.mean = self.normalizer.mean.tolist()
            self.config.var = self.normalizer.var.tolist()

        if self.do_normalize:
            scores = self.normalizer.normalize(scores)
            end_score = self.normalizer.normalize(end_score)

        if not return_dict:
            return scores, end_score

        return ScoreModelOutput(
            scores=scores,  # size = (B, L, D)
            end_scores=end_score,  # size = (B, D)
        )

    def set_normalize(self, mode: bool = True) -> None:
        if self.do_normalize == mode:
            return

        self.do_normalize = self.config.do_normalize = mode


NormalizeFunction = Literal["affine", "scale", "translate", "identity"]
NormalizerType = Literal["RunningMeanStd", "ExponentialMovingAverage"]


class Normalizer(nn.Layer):
    """Normalize input to have zero mean and unit variance."""

    mean: paddle.Tensor
    var: paddle.Tensor
    count: paddle.Tensor
    normalize_function: NormalizeFunction

    def __init__(
        self,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: str | None = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        if normalize_function not in {"affine", "scale", "translate", "identity"}:
            raise ValueError(
                f"Invalid normalization function type: {normalize_function}. ",
                'Expected one of "affine", "scale", "translate", "identity".',
            )
        self.normalize_function = normalize_function
        self.register_buffer("mean", paddle.zeros(shape, dtype=paddle.get_default_dtype()))  # align torch zeros/ones
        self.register_buffer("var", paddle.ones(shape, dtype=paddle.get_default_dtype()))  # align torch zeros/ones
        self.register_buffer("count", paddle.zeros(1, dtype=paddle.int64))

    @abstractmethod
    def update(self, data: paddle.Tensor) -> None:
        """Update mean and variance."""
        raise NotImplementedError

    @property
    def std(self) -> paddle.Tensor:
        """Return standard deviation."""
        return self.var.sqrt()

    def set_mean_var(
        self,
        mean: paddle.Tensor | list[float] | tuple[float, ...] | None,
        var: paddle.Tensor | list[float] | tuple[float, ...] | None,
    ) -> None:
        """Set mean and variance."""
        mean = paddle.to_tensor(mean, dtype=self.mean.dtype, place=self.mean.place) if mean is not None else self.mean
        var = paddle.to_tensor(var, dtype=self.var.dtype, place=self.var.place) if var is not None else self.var

        assert mean.shape == self.mean.shape
        assert var.shape == self.var.shape

        self.mean = mean
        self.var = var

    def forward(
        self,
        data: paddle.Tensor,
        epsilon=1e-8,
    ) -> paddle.Tensor:
        """Update and normalize input."""
        if self.training:
            self.update(data)
        return self.normalize(data, epsilon=epsilon)

    def normalize(
        self,
        data: paddle.Tensor,
        epsilon=1e-8,
    ) -> paddle.Tensor:
        """Normalize input."""
        if self.normalize_function == "affine":
            return (data - self.mean.detach()) / (self.std.detach() + epsilon)
        if self.normalize_function == "scale":
            return data / (self.std.detach() + epsilon)
        if self.normalize_function == "translate":
            return data - self.mean.detach()
        if self.normalize_function == "identity":
            return data
        raise ValueError(
            f"Invalid normalization function type: {self.normalize_function}. ",
            'Expected one of "affine", "scale", "translate", "identity".',
        )

    @classmethod
    def instantiate(
        cls,
        normalizer_type: NormalizerType | None,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: str | None = None,
        **kwargs: Any,
    ) -> Normalizer:
        """Get a normalizer."""
        if normalizer_type == "RunningMeanStd":
            return RunningMeanStd(
                normalize_function,
                shape=shape,
                device=device,
            )
        if normalizer_type == "ExponentialMovingAverage":
            return ExponentialMovingAverage(
                normalize_function,
                shape=shape,
                device=device,
                **kwargs,
            )
        if normalizer_type is None:
            return IdentityNormalizer(
                normalize_function,
                shape=shape,
                device=device,
            )
        raise ValueError(
            f"Invalid normalization function type: {normalizer_type}. "
            'Expected one of "RunningMeanStd", "ExponentialMovingAverage".',
        )


class RunningMeanStd(Normalizer):
    """Running mean and standard deviation."""

    def update(self, data: paddle.Tensor) -> None:
        """Update mean and variance."""
        batch_mean = data.mean(0)
        batch_var = data.var(0)
        batch_count = data.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = (  # pylint: disable=invalid-name
            m_a + m_b + paddle.square(delta) * (self.count * batch_count / total_count)
        )
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class ExponentialMovingAverage(Normalizer):
    """Exponential moving average."""

    def __init__(
        self,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: str | None = None,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(normalize_function, shape=shape, device=device)
        self.momentum = momentum

    def update(self, data: paddle.Tensor) -> None:
        """Update mean and variance."""
        batch_mean = data.mean(0)
        batch_var = data.var(0)
        batch_count = data.shape[0]

        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1.0 - self.momentum) * batch_var
        self.count += batch_count  # pylint: disable=no-member


class IdentityNormalizer(Normalizer):
    """Identity normalizer."""

    def update(self, data: paddle.Tensor) -> None:
        """Update mean and variance."""
        self.count += data.shape[0]  # pylint: disable=no-member
