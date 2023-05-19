# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import NEG_INF, BaseOutput
from .attention import BasicTransformerBlock
from .embeddings import TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin


@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    Args:
        predicted_image_embedding (`paddle.Tensor` of shape `(batch_size, embedding_dim)`):
            The predicted CLIP image embedding conditioned on the CLIP text embedding input.
    """

    predicted_image_embedding: paddle.Tensor


class PriorTransformer(ModelMixin, ConfigMixin):
    """
    The prior transformer from unCLIP is used to predict CLIP image embeddings from CLIP text embeddings. Note that the
    transformer predicts the image embeddings through a denoising diffusion process.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    For more details, see the original paper: https://arxiv.org/abs/2204.06125

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_layers (`int`, *optional*, defaults to 20): The number of layers of Transformer blocks to use.
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the CLIP embeddings. Note that CLIP
            image embeddings and text embeddings are both the same dimension.
        num_embeddings (`int`, *optional*, defaults to 77): The max number of clip embeddings allowed. I.e. the
            length of the prompt after it has been tokenized.
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected hidden_states. The actual length of the used hidden_states is `num_embeddings +
            additional_embeddings`.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.

    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 20,
        embedding_dim: int = 768,
        num_embeddings=77,
        additional_embeddings=4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings

        self.time_proj = Timesteps(inner_dim, True, 0)
        self.time_embedding = TimestepEmbedding(inner_dim, inner_dim)

        self.proj_in = nn.Linear(embedding_dim, inner_dim)

        self.embedding_proj = nn.Linear(embedding_dim, inner_dim)
        self.encoder_hidden_states_proj = nn.Linear(embedding_dim, inner_dim)

        self.positional_embedding = self.create_parameter(
            (1, num_embeddings + additional_embeddings, inner_dim),
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.prd_embedding = self.create_parameter(
            (1, 1, inner_dim), dtype=paddle.get_default_dtype(), default_initializer=nn.initializer.Constant(0.0)
        )
        self.transformer_blocks = nn.LayerList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                    attention_bias=True,
                )
                for d in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim)
        self.proj_to_clip_embeddings = nn.Linear(inner_dim, embedding_dim)

        causal_attention_mask = paddle.triu(
            paddle.full([num_embeddings + additional_embeddings, num_embeddings + additional_embeddings], NEG_INF), 1
        )
        causal_attention_mask = causal_attention_mask.unsqueeze(0)
        self.register_buffer("causal_attention_mask", causal_attention_mask, persistable=False)

        self.clip_mean = self.create_parameter(
            (1, embedding_dim), dtype=paddle.get_default_dtype(), default_initializer=nn.initializer.Constant(0.0)
        )
        self.clip_std = self.create_parameter(
            (1, embedding_dim), dtype=paddle.get_default_dtype(), default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(
        self,
        hidden_states,
        timestep: Union[paddle.Tensor, float, int],
        proj_embedding: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`paddle.Tensor` of shape `(batch_size, embedding_dim)`):
                x_t, the currently predicted image embeddings.
            timestep (`paddle.Tensor`):
                Current denoising step.
            proj_embedding (`paddle.Tensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`paddle.Tensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`paddle.Tensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.prior_transformer.PriorTransformerOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.prior_transformer.PriorTransformerOutput`] or `tuple`:
            [`~models.prior_transformer.PriorTransformerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        hidden_states = hidden_states.cast(self.dtype)
        batch_size = hidden_states.shape[0]

        timesteps = timestep
        if not paddle.is_tensor(timesteps):
            timesteps = paddle.to_tensor([timesteps], dtype=paddle.int64)
        elif paddle.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None]

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * paddle.ones((batch_size,), dtype=timesteps.dtype)

        timesteps_projected = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        timesteps_projected = timesteps_projected.cast(hidden_states.dtype)
        time_embeddings = self.time_embedding(timesteps_projected)

        proj_embeddings = self.embedding_proj(proj_embedding)
        encoder_hidden_states = self.encoder_hidden_states_proj(encoder_hidden_states)
        hidden_states = self.proj_in(hidden_states)
        prd_embedding = self.prd_embedding.cast(hidden_states.dtype).expand([batch_size, -1, -1])
        positional_embeddings = self.positional_embedding.cast(hidden_states.dtype)

        hidden_states = paddle.concat(
            [
                encoder_hidden_states,
                proj_embeddings[:, None, :],
                time_embeddings[:, None, :],
                hidden_states[:, None, :],
                prd_embedding,
            ],
            axis=1,
        )

        hidden_states = hidden_states + positional_embeddings

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.cast(hidden_states.dtype)) * NEG_INF
            attention_mask = F.pad(
                attention_mask.unsqueeze(0), (0, self.additional_embeddings), value=0.0, data_format="NCL"
            ).squeeze(0)
            attention_mask = (attention_mask[:, None, :] + self.causal_attention_mask).cast(hidden_states.dtype)
            attention_mask = attention_mask.repeat_interleave(self.config.num_attention_heads, axis=0)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -1]
        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
