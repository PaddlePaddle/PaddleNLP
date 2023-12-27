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
# ==============================================================================
"""Auto-models for score models."""

from __future__ import annotations

from abc import abstractmethod

# from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import paddle
import paddle.distributed as dist
import paddle.nn as nn

from paddlenlp.transformers import PretrainedConfig
from paddlenlp.transformers.model_outputs import ModelOutput

# from paddle.distributed import fleet


# from paddlenlp.transformers.sequence_parallel_utils import AllGatherOp, all_gather

# from transformers.models.auto.auto_factory import (
#     _BaseAutoModelClass,
#     _LazyAutoMapping,
#     auto_class_update,
#     getattribute_from_module,
# )
# from transformers.models.auto.configuration_auto import (
#     CONFIG_MAPPING_NAMES,
#     model_type_to_module_name,
# )

# class _LazyAutoMappingInSafeRLHF(_LazyAutoMapping):
#     def _load_attr_from_module(self, model_type: str, attr: str) -> Any:
#         module_name = model_type_to_module_name(model_type)
#         if module_name not in self._modules:
#             self._modules[module_name] = importlib.import_module(
#                 f'.{module_name}',
#                 'safe_rlhf.models.score_model',
#             )
#         return getattribute_from_module(self._modules[module_name], attr)

# MODEL_FOR_SCROE_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
#     [
#         # Score model mapping
#         ('llama', 'LlamaModelForScore'),
#         ('bloom', 'BloomModelForScore'),
#         ('open_llama', 'OpenLlamaForScore'),
#         ('opt', 'OPTForScore'),
#         ('gpt_neo', 'GPTNeoForScore'),
#         ('gptj', 'GPTJForScore'),
#         ('gpt2', 'GPT2ForScore'),
#         ('gpt_neox', 'GPTNeoXForScore'),
#     ], )
# MODEL_FOR_SCORE_MAPPING: OrderedDict[str, Any] = _LazyAutoMappingInSafeRLHF(
#     CONFIG_MAPPING_NAMES,
#     MODEL_FOR_SCROE_MAPPING_NAMES,
# )

# @functools.partial(auto_class_update, head_doc='score model')
# class AutoModelForScore(_BaseAutoModelClass):
#     _model_mapping: OrderedDict[str, Any] = MODEL_FOR_SCORE_MAPPING


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
        attention_mask: paddle.Tensor,  # size = (B, L)
        return_dict: bool | None = None,
    ) -> ScoreModelOutput:
        """Forward pass of the score model."""
        scores = self.score_head(hidden_state)  # size = (B, L, D)

        end_score = []
        for i in range(hidden_state.shape[0]):
            end_index = attention_mask[i].nonzero()[-1].item()
            end_score.append(scores[i, end_index])  # size = (D,)
        end_score = paddle.stack(end_score, axis=0)  # size = (B, D)

        if self.training:

            if dist.is_initialized():
                # TODO(guosheng): maybe only need nodes in data parallel group
                # when support hybird dist parallel.
                gathered_end_score_list = [paddle.zeros_like(end_score) for _ in range(dist.get_world_size())]
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


"""Normalizer for score models."""
from typing import Literal

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
