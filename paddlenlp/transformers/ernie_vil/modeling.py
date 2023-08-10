# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team.
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
from functools import partial
from typing import Any, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F

from ...utils.initializer import normal_
from .. import PretrainedModel
from ..clip.modeling import CLIPVisionTransformer as ErnieViLVisionTransformer
from ..clip.modeling import clip_loss
from ..ernie.modeling import ErnieModel
from ..model_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
)
from .configuration import ErnieViLConfig, ErnieViLTextConfig, ErnieViLVisionConfig

__all__ = [
    "ErnieViLModel",
    "ErnieViLTextModel",
    "ErnieViLVisionModel",
    "ErnieViLPretrainedModel",
]

ERNIE_VIL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # vit model
    "PaddlePaddle/ernie_vil-2.0-base-zh",
    "PaddlePaddle/disco_diffusion_ernie_vil-2.0-base-zh",
]


def quick_gelu(x):
    return x * F.sigmoid(1.702 * x)


F.quick_gelu = quick_gelu


@dataclass
class ErnieViLOutput(ModelOutput):
    """
    Args:
        loss: (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image: (`paddle.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text: (`paddle.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds: (`paddle.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`ErnieModel`].
        image_embeds: (`paddle.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`ErnieViLVisionTransformer`].
        text_model_output: (:class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`ErnieModel`].
        vision_model_output: (:class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPooling`):
            The output of the [`VisionTransformer`].
    """

    loss: Optional[paddle.Tensor] = None
    logits_per_image: paddle.Tensor = None
    logits_per_text: paddle.Tensor = None
    text_embeds: paddle.Tensor = None
    image_embeds: paddle.Tensor = None
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class ErnieViLPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ErnieViL models. It provides ErnieViL related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    config_class = ErnieViLConfig
    base_model_prefix = "ernie_vil"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, nn.TransformerEncoder):
            module.enable_recompute = value

    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def _init_weights(self, layer):
        """Initialize the weights"""
        if isinstance(layer, ErnieViLVisionTransformer):
            # find nn.LayerNorm
            for sub_layer in layer.sublayers():
                if isinstance(sub_layer, nn.LayerNorm):
                    sub_layer._epsilon = layer.config.layer_norm_eps

        elif isinstance(layer, ErnieModel):
            # find nn.LayerNorm
            for sub_layer in layer.sublayers():
                if isinstance(sub_layer, nn.LayerNorm):
                    sub_layer._epsilon = layer.config.layer_norm_eps
                elif isinstance(layer, (nn.Linear, nn.Embedding)):
                    normal_(layer.weight, mean=0.0, std=layer.config.initializer_range)


class ErnieViLModel(ErnieViLPretrainedModel):
    r"""
    The bare ErnieViL Model outputting logits_per_image and logits_per_text.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`ErnieViLConfig`):
            An instance of ErnieViLConfig used to construct ErnieViLModel.
    """
    config_class = ErnieViLConfig

    def __init__(self, config: ErnieViLConfig):
        super().__init__(config)

        if not isinstance(config.text_config, ErnieViLTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type ErnieViLTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, ErnieViLVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type ErnieViLVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.text_model = ErnieModel(text_config)

        self.vision_model = ErnieViLVisionTransformer(vision_config)

        self.temperature = self.create_parameter(
            shape=(1,),
            default_initializer=nn.initializer.Constant(config.logit_scale_init_value),
            dtype=paddle.get_default_dtype(),
        )

    def get_image_features(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> paddle.Tensor:
        r"""
        Args:
            pixel_values (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
                [`ErnieViLFeatureExtractor`]. See [`ErnieViLFeatureExtractor.__call__`] for details.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.

        Returns:
            image_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`ErnieViLVisionModel`].

        Examples:
            .. code-block::

                import requests
                from PIL import Image
                from paddlenlp.transformers import ErnieViLProcessor, ErnieViLModel

                model = ErnieViLModel.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
                processor = ErnieViLProcessor.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")

                url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                image = Image.open(requests.get(url, stream=True).raw)
                inputs = processor(images=image, return_tensors="pd")
                image_features = model.get_image_features(**inputs)

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_features = vision_outputs[1]
        return image_features

    def get_text_features(
        self,
        input_ids,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        task_type_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
                Indices can be obtained using [`ErnieViLTokenizer`].
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
            token_type_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
                Its data type should be `int64`. Defaults to `None`, which means we don't add segment embeddings.
            task_type_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of tasks of each input sequence tokens in the task embeddings (ErnieModel). Selected in
                the range ``[0, task_type_vocab_size - 1]``. Defaults to `None`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPoolingAndCrossAttentions`] instead of a plain tuple.

        Returns:
            text_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            the pooled output of [`ErnieModel`].

        Example:
            .. code-block::

                from paddlenlp.transformers import ErnieViLModel, ErnieViLTokenizer

                model = ErnieViLModel.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
                tokenizer = ErnieViLTokenizer.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")

                inputs = tokenizer(["一只猫的照片", "一条狗的照片"], padding=True, return_tensors="pd")
                text_features = model.get_text_features(**inputs)

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            task_type_ids=task_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_features = text_outputs[1]
        return text_features

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        task_type_ids: Optional[paddle.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ErnieViLOutput]:
        r"""
        The ErnieViLModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
                Its data type should be `int64` and it has a shape of [text_batch_size, sequence_length].
            pixel_values (Tensor):
                Pixel values. Padding will be ignored by default should you provide it.
                Its data type should be `float32` and it has a shape of [image_batch_size, num_channels, height, width].
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings (ErnieModel). Selected in
                the range ``[0, max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            task_type_ids (Tensor, optional):
                Indices of tasks of each input sequence tokens in the task embeddings (ErnieModel). Selected in
                the range ``[0, task_type_vocab_size - 1]``.
                Shape as `(batch_size, sequence_length)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention (ErnieModel) to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`ErnieViLOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`ErnieViLOutput` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`ErnieViLOutput`.

        Example:
            .. code-block::

                import requests
                import paddle.nn.functional as F
                from PIL import Image
                from paddlenlp.transformers import ErnieViLModel, ErnieViLProcessor

                processor = ErnieViLProcessor.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
                model = ErnieViLModel.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
                model.eval()

                url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                image = Image.open(requests.get(url, stream=True).raw)

                inputs = processor(text=["一只猫的照片", "一条狗的照片"],
                                images=image,
                                padding=True,
                                return_tensors="pd")

                outputs = model(**inputs)

                logits_per_image = outputs[0]
                probs = F.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]
        # normalized features
        image_embeds = F.normalize(image_embeds)
        text_embeds = F.normalize(text_embeds)
        if paddle.distributed.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            gathered_image_features = [paddle.zeros_like(image_embeds) for _ in range(world_size)]
            gathered_text_features = [paddle.zeros_like(text_embeds) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_embeds)
            dist.all_gather(gathered_text_features, text_embeds)
            # Add current text_embeds image_embeds into the batch for gradient update
            image_embeds = paddle.concat(
                [image_embeds] + gathered_image_features[:rank] + gathered_image_features[rank + 1 :]
            )
            text_embeds = paddle.concat(
                [text_embeds] + gathered_text_features[:rank] + gathered_text_features[rank + 1 :]
            )
        # cosine similarity as logits
        logit_scale = self.temperature.exp()

        logits_per_text = paddle.matmul(text_embeds * logit_scale, image_embeds, transpose_y=True)
        logits_per_image = logits_per_text.t()

        # clip temperature
        self.temperature.clip(-100.0, 100.0)

        loss = None

        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return ErnieViLOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class ErnieViLTextModel(ErnieViLPretrainedModel):
    r"""
    The text model from ErnieViL without any head or projection on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`ErnieViLTextConfig`):
            An instance of ErnieViLTextConfig used to construct ErnieViLTextModel.
    """

    config_class = ErnieViLTextConfig

    def __init__(self, config: ErnieViLTextConfig):
        super().__init__(config)
        self.text_model = ErnieModel(config)

    def get_input_embeddings(self) -> nn.Layer:
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_model.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        task_type_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
                Indices can be obtained using [`ErnieViLTokenizer`].
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
            token_type_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
                Its data type should be `int64`. Defaults to `None`, which means we don't add segment embeddings.
            task_type_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of tasks of each input sequence tokens in the task embeddings (ErnieModel). Selected in
                the range ``[0, task_type_vocab_size - 1]``. Defaults to `None`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPoolingAndCrossAttentions`] instead of a plain tuple.

        Returns:
            An instance of :class:`BaseModelOutputWithPoolingAndCrossAttentions` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`BaseModelOutputWithPoolingAndCrossAttentions`.

        Examples:

        ```python
        >>> from paddlenlp.transformers import ErnieViLTokenizer, ErnieViLTextModel

        >>> model = ErnieViLTextModel.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
        >>> tokenizer = ErnieViLTokenizer.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")

        >>> inputs = tokenizer(["一只猫的照片", "一条狗的照片"], padding=True, return_tensors="pd")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class ErnieViLVisionModel(ErnieViLPretrainedModel):
    r"""
    The vision model from ErnieViL without any head or projection on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`ErnieViLVisionConfig`):
            An instance of ErnieViLVisionConfig used to construct ErnieViLVisionModel.
    """

    config_class = ErnieViLVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: ErnieViLVisionConfig):
        super().__init__(config)

        self.vision_model = ErnieViLVisionTransformer(config)

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.conv1

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Args:
            pixel_values (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
                [`ErnieViLFeatureExtractor`]. See [`ErnieViLFeatureExtractor.__call__`] for details.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.

        Returns:
            An instance of :class:`BaseModelOutputWithPooling` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`BaseModelOutputWithPooling`.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import ErnieViLProcessor, ErnieViLVisionModel

        >>> model = ErnieViLVisionModel.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
        >>> processor = ErnieViLProcessor.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pd")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
