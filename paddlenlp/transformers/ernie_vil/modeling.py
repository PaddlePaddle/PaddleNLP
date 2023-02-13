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
from ...utils.log import logger
from .. import PretrainedModel
from ..clip.modeling import CLIPVisionTransformer as ErnieViLVisionTransformer
from ..clip.modeling import clip_loss
from ..configuration_utils import PretrainedConfig
from ..ernie.modeling import ErnieModel
from ..guided_diffusion_utils import (
    DiscoDiffusionMixin,
    create_gaussian_diffusion,
    create_secondary_model,
    create_unet_model,
)
from ..model_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
)
from ..utils import resolve_cache_dir
from .configuration import ErnieViLConfig, ErnieViLTextConfig, ErnieViLVisionConfig

__all__ = [
    "ErnieViLModel",
    "ErnieViLTextModel",
    "ErnieViLVisionModel",
    "ErnieViLPretrainedModel",
    "ErnieViLForImageGeneration",
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

    def init_weights(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.apply(self._init_weights)

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

    @classmethod
    def from_pretrained_v2(cls, pretrained_model_name_or_path, from_hf_hub: bool = False, *args, **kwargs):
        load_state_as_np = kwargs.pop("load_state_as_np", False)
        config = kwargs.pop("config", None)
        force_download = kwargs.pop("force_download", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", None)
        dtype = kwargs.pop("dtype", None)
        cache_dir = kwargs.pop("cache_dir", None)

        cache_dir = resolve_cache_dir(pretrained_model_name_or_path, from_hf_hub, cache_dir)

        model_kwargs = kwargs
        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                from_hf_hub=from_hf_hub,
                **kwargs,
            )
        # Attention! we donot save this config.json
        # config.save_pretrained(cache_dir)

        # 2. init the model
        init_args = config["init_args"] or ()
        model = cls(config, *init_args, **model_kwargs)

        # 3. resolve model_weight file
        model_weight_file = cls._resolve_model_file_path(
            pretrained_model_name_or_path, cache_dir=cache_dir, from_hf_hub=from_hf_hub
        )

        # 4. loading the state dict
        model_state_dict = paddle.load(model_weight_file, return_numpy=load_state_as_np)

        loaded_state_dict_keys = list(model_state_dict.keys())
        # TODO(wj-Mcat): load shard checkpoint weight file, refer to: https://github.com/huggingface/transformers/pull/16343
        model, missing_keys, unexpected_keys, mismatched_keys = cls._load_pretrained_model(
            model=model,
            state_dict=model_state_dict,
            loaded_keys=loaded_state_dict_keys,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            dtype=dtype,
        )

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )
        if paddle.in_dynamic_mode():
            return model

        return model, model_state_dict


class ErnieViLModel(ErnieViLPretrainedModel):
    r"""
    The bare ErnieViL Model outputting logits_per_image and logits_per_text.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
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

        self.init_weights()

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

        if dist.get_world_size() > 1:
            feature_list_img = []
            feature_list_txt = []
            dist.all_gather(feature_list_img, image_embeds)
            dist.all_gather(feature_list_txt, text_embeds)
            image_embeds = paddle.concat(x=feature_list_img, axis=0)
            text_embeds = paddle.concat(x=feature_list_txt, axis=0)

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
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`ErnieViLTextConfig`):
            An instance of ErnieViLTextConfig used to construct ErnieViLTextModel.
    """

    config_class = ErnieViLTextConfig

    def __init__(self, config: ErnieViLTextConfig):
        super().__init__(config)
        self.text_model = ErnieModel(config)

        self.init_weights()

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
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
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

        self.init_weights()

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


class ErnieViLForImageGeneration(ErnieViLPretrainedModel, DiscoDiffusionMixin):
    r"""
    ErnieViL Model with diffusion model on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`ErnieViLConfig`):
            An instance of ErnieViLConfig used to construct ErnieViLForImageGeneration.
    """

    config_class = ErnieViLConfig

    def __init__(self, config: ErnieViLConfig):
        super().__init__(config)
        self.ernie_vil = ErnieViLModel(config)
        self.unet_model = create_unet_model(
            image_size=512,
            num_channels=256,
            num_res_blocks=2,
            channel_mult="",
            learn_sigma=True,
            class_cond=False,
            attention_resolutions="32, 16, 8",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            resblock_updown=True,
            use_new_attention_order=False,
        )
        self.secondary_model = create_secondary_model()

        # eval mode and stop all param's gradient
        self.eval()
        for param in self.parameters():
            param.stop_gradient = True

    def generate(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        task_type_ids=None,
        init_image=None,
        output_dir="disco_diffusion_ernie_vil-2.0-base-zh/",
        width_height=[1280, 768],
        skip_steps=0,
        steps=250,
        cut_ic_pow=1,
        init_scale=1000,
        clip_guidance_scale=5000,
        tv_scale=0,
        range_scale=0,
        sat_scale=0,
        cutn_batches=4,
        perlin_init=False,
        perlin_mode="mixed",
        seed=None,
        eta=0.8,
        clamp_grad=True,
        clamp_max=0.05,
        cut_overview="[12]*400+[4]*600",
        cut_innercut="[4]*400+[12]*600",
        cut_icgray_p="[0.2]*400+[0]*600",
        save_rate=10,
        n_batches=1,
        batch_name="",
        use_secondary_model=True,
        randomize_class=True,
        clip_denoised=False,
    ):
        r"""
        The ErnieViLForImageGeneration generate method.

        Args:
            input_ids (Tensor):
                See :class:`ErnieViLModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieViLModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieViLModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieViLModel`.
            task_type_ids (Tensor, optional):
                See :class:`ErnieViLModel`.
            init_image (Path, optional):
                Recall that in the image sequence above, the first image shown is just noise.  If an init_image
                is provided, diffusion will replace the noise with the init_image as its starting state.  To use
                an init_image, upload the image to the Colab instance or your Google Drive, and enter the full
                image path here. If using an init_image, you may need to increase skip_steps to ~ 50% of total
                steps to retain the character of the init. See skip_steps above for further discussion.
                Default to `None`.
            output_dir (Path, optional):
                Output directory.
                Default to `disco_diffusion_ernie_vil-2.0-base-zh/`.
            width_height (List[int, int], optional):
                Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge
                length should be set to a multiple of 64px, and a minimum of 512px on the default ErnieViL model setting.
                If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your
                image to make it so.
                Default to `[1280, 768]`.
            skip_steps (int, optional):
                Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively
                gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high,
                so images change dramatically in early steps.As DD moves along the curve, noise levels (and thus the
                amount an image changes per step) declines, and image coherence from one step to the next increases.
                The first few steps of denoising are often so dramatic that some steps (maybe 10-15% of total) can be
                skipped without affecting the final image. You can experiment with this as a way to cut render times.
                If you skip too many steps, however, the remaining noise may not be high enough to generate new content,
                and thus may not have time left to finish an image satisfactorily.Also, depending on your other settings,
                you may need to skip steps to prevent ErnieViL from overshooting your goal, resulting in blown out colors
                (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that
                the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate
                other problems.Lastly, if using an init_image, you will need to skip ~50% of the diffusion steps to retain
                the shapes in the original init image. However, if you're using an init_image, you can also adjust
                skip_steps up or down for creative reasons.  With low skip_steps you can get a result "inspired by"
                the init_image which will retain the colors and rough layout and shapes but look quite different.
                With high skip_steps you can preserve most of the init_image contents and just do fine tuning of the texture.
                Default to `0`.
            steps:
                When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration)
                involves the AI looking at subsets of the image called 'cuts' and calculating the 'direction' the image
                should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser,
                and moves to the next step.Increasing steps will provide more opportunities for the AI to adjust the image,
                and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps
                comes at the expense of longer render times.  Also, while increasing steps should generally increase image
                quality, there is a diminishing return on additional steps beyond 250 - 500 steps.  However, some intricate
                images can take 1000, 2000, or more steps.  It is really up to the user.  Just know that the render time is
                directly related to the number of steps, and many other parameters have a major impact on image quality, without
                costing additional time.
            cut_ic_pow (int, optional):
                This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and
                therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small
                inner cuts, you may lose overall image coherency and/or it may cause an undesirable 'mosaic' effect.
                Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping
                with some details.
                Default to `1`.
            init_scale (int, optional):
                This controls how strongly ErnieViL will try to match the init_image provided.  This is balanced against the
                clip_guidance_scale (CGS) above.  Too much init scale, and the image won't change much during diffusion.
                Too much CGS and the init image will be lost.
                Default to `1000`.
            clip_guidance_scale (int, optional):
                CGS is one of the most important parameters you will use. It tells DD how strongly you want ErnieViL to move
                toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot
                the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust
                CGS. Note that this parameter generally scales with image dimensions. In other words, if you increase your
                total dimensions by 50% (e.g. a change from 512 x 512 to 512 x 768), then to maintain the same effect on the
                image, you'd want to increase clip_guidance_scale from 5000 to 7500. Of the basic settings, clip_guidance_scale,
                steps and skip_steps are the most important contributors to image quality, so learn them well.
                Default to `5000`.
            tv_scale (int, optional):
                Total variance denoising. Optional, set to zero to turn off. Controls smoothness of final output. If used,
                tv_scale will try to smooth out your final image to reduce overall noise. If your image is too 'crunchy',
                increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.
                See https://en.wikipedia.org/wiki/Total_variation_denoising
                Default to `0`.
            range_scale (int, optional):
                Optional, set to zero to turn off.  Used for adjustment of color contrast.  Lower range_scale will increase
                contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images.
                Higher range_scale will reduce contrast, for more muted images.
                Default to `0`.
            sat_scale (int, optional):
                Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation.
                If your image is too saturated, increase sat_scale to reduce the saturation.
                Default to `0`.
            cutn_batches (int, optional):
                Each iteration, the AI cuts the image into smaller pieces known as cuts, and compares each cut to the prompt
                to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has
                more chances to fine-tune the image precision in each timestep.  Additional cuts are memory intensive, however,
                and if DD tries to evaluate too many cuts at once, it can run out of memory.  You can use cutn_batches to increase
                cuts per timestep without increasing memory usage. At the default settings, DD is scheduled to do 16 cuts per
                timestep.  If cutn_batches is set to 1, there will indeed only be 16 cuts total per timestep. However, if
                cutn_batches is increased to 4, DD will do 64 cuts total in each timestep, divided into 4 sequential batches
                of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts,
                but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take ~4 times as long to
                render each image.So, (scheduled cuts) x (cutn_batches) = (total cuts per timestep). Increasing cutn_batches will
                increase render times, however, as the work is being done sequentially.  DD's default cut schedule is a good place
                to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below.
                Default to `4`.
            perlin_init (bool, optional):
                Normally, DD will use an image filled with random noise as a starting point for the diffusion curve.
                If perlin_init is selected, DD will instead use a Perlin noise model as an initial state.  Perlin has very
                interesting characteristics, distinct from random noise, so it's worth experimenting with this for your projects.
                Beyond perlin, you can, of course, generate your own noise images (such as with GIMP, etc) and use them as an
                init_image (without skipping steps). Choosing perlin_init does not affect the actual diffusion process, just the
                starting point for the diffusion. Please note that selecting a perlin_init will replace and override any init_image
                you may have specified. Further, because the 2D, 3D and video animation systems all rely on the init_image system,
                if you enable Perlin while using animation modes, the perlin_init will jump in front of any previous image or video
                input, and DD will NOT give you the expected sequence of coherent images. All of that said, using Perlin and
                animation modes together do make a very colorful rainbow effect, which can be used creatively.
                Default to `False`.
            perlin_mode (str, optional):
                sets type of Perlin noise: colored, gray, or a mix of both, giving you additional options for noise types. Experiment
                to see what these do in your projects.
                Default to `mixed`.
            seed (int, optional):
                Deep in the diffusion code, there is a random number seed which is used as the basis for determining the initial
                state of the diffusion.  By default, this is random, but you can also specify your own seed. This is useful if you like a
                particular result and would like to run more iterations that will be similar. After each run, the actual seed value used will be
                reported in the parameters report, and can be reused if desired by entering seed # here.  If a specific numerical seed is used
                repeatedly, the resulting images will be quite similar but not identical.
                Default to `None`.
            eta (float, optional):
                Eta (greek letter η) is a diffusion model variable that mixes in a random amount of scaled noise into each timestep.
                0 is no noise, 1.0 is more noise. As with most DD parameters, you can go below zero for eta, but it may give you
                unpredictable results. The steps parameter has a close relationship with the eta parameter. If you set eta to 0,
                then you can get decent output with only 50-75 steps. Setting eta to 1.0 favors higher step counts, ideally around
                250 and up. eta has a subtle, unpredictable effect on image, so you'll need to experiment to see how this affects your projects.
                Default to `0.8`.
            clamp_grad (bool, optional):
                As I understand it, clamp_grad is an internal limiter that stops DD from producing extreme results. Try your images with and without
                clamp_grad. If the image changes drastically with clamp_grad turned off, it probably means your clip_guidance_scale is too high and
                should be reduced.
                Default to `True`.
            clamp_max (float, optional):
                Sets the value of the clamp_grad limitation. Default is 0.05, providing for smoother, more muted coloration in images, but setting
                higher values (0.15-0.3) can provide interesting contrast and vibrancy.
                Default to `0.05`.
            cut_overview (str, optional):
                The schedule of overview cuts.
                Default to `'[12]*400+[4]*600'`.
            cut_innercut (str, optional):
                The schedule of inner cuts.
                Default to `'[4]*400+[12]*600'`.
            cut_icgray_p (str, optional):
                This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts
                themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall
                image coherency and/or it may cause an undesirable 'mosaic' effect.   Low cut_ic_pow values will allow the inner cuts to be
                larger, helping image coherency while still helping with some details.
                Default to `'[0.2]*400+[0]*600'`.
            save_rate (int, optional):
                During a diffusion run, you can monitor the progress of each image being created with this variable.  If display_rate is set
                to 50, DD will show you the in-progress image every 50 timesteps. Setting this to a lower value, like 5 or 10, is a good way
                to get an early peek at where your image is heading. If you don't like the progression, just interrupt execution, change some
                settings, and re-run.  If you are planning a long, unmonitored batch, it's better to set display_rate equal to steps, because
                displaying interim images does slow Colab down slightly.
                Default to `10`.
            n_batches (int, optional):
                This variable sets the number of still images you want DD to create.  If you are using an animation mode (see below for details)
                DD will ignore n_batches and create a single set of animated frames based on the animation settings.
                Default to `1`.
            batch_name (str, optional):
                The name of the batch, the batch id will be named as "progress-[batch_name]-seed-[range(n_batches)]-[save_rate]". To avoid your
                artworks be overridden by other users, please use a unique name.
                Default to `''`.
            use_secondary_model (bool, optional):
                Whether or not use secondary model.
                Default to `True`.
            randomize_class (bool, optional):
                Random class.
                Default to `True`.
            clip_denoised (bool, optional):
                Clip denoised.
                Default to `False`.

        Returns:
            List[PIL.Image]: Returns n_batches of final image.
            Its data type should be PIL.Image.

        Example:
            .. code-block::

            from paddlenlp.transformers import ErnieViLForImageGeneration, ErnieViLTokenizer

            # Initialize the model and tokenizer
            model_name_or_path = "PaddlePaddle/disco_diffusion_ernie_vil-2.0-base-zh"
            model = ErnieViLForImageGeneration.from_pretrained(model_name_or_path)
            tokenizer = ErnieViLTokenizer.from_pretrained(model_name_or_path)
            model.eval()

            # Prepare the model inputs.
            text_prompt = "小桥流水人家。"
            style = None
            artist = None
            text_prompt = model.preprocess_text_prompt(text_prompt, style=style, artist=artist)
            tokenized_inputs = tokenizer(text_prompt, return_tensors="pd", padding="max_length", max_length=tokenizer.model_max_length)

            images = model.generate(**tokenized_inputs)
            # return List[PIL.Image]
            images[0].save("figure.png")

        """
        self.diffusion = create_gaussian_diffusion(
            steps=steps,
            learn_sigma=True,
            sigma_small=False,
            noise_schedule="linear",
            predict_xstart=False,
            rescale_timesteps=True,
        )
        target_text_embeds = self.ernie_vil.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
        )

        images_list = super().disco_diffusion_generate(
            target_text_embeds=target_text_embeds,
            clamp_grad=clamp_grad,
            clamp_max=clamp_max,
            clip_denoised=clip_denoised,
            clip_guidance_scale=clip_guidance_scale,
            cut_ic_pow=cut_ic_pow,
            cut_icgray_p=cut_icgray_p,
            cut_innercut=cut_innercut,
            cut_overview=cut_overview,
            cutn_batches=cutn_batches,
            save_rate=save_rate,
            eta=eta,
            init_image=init_image,
            init_scale=init_scale,
            n_batches=n_batches,
            output_dir=output_dir,
            perlin_init=perlin_init,
            perlin_mode=perlin_mode,
            randomize_class=randomize_class,
            range_scale=range_scale,
            sat_scale=sat_scale,
            seed=seed,
            skip_steps=skip_steps,
            tv_scale=tv_scale,
            use_secondary_model=use_secondary_model,
            width_height=width_height,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            batch_name=batch_name,
        )

        return images_list

    def preprocess_text_prompt(self, text_prompt, style=None, artist=None):
        text_prompt = text_prompt.rstrip(",.，。")
        if style is not None:
            text_prompt += "，{}".format(style)
        if artist is not None:
            text_prompt += "，由{}所作".format(artist)
        return text_prompt
