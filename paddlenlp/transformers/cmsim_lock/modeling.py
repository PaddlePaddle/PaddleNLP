# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from ...utils.initializer import normal_, ones_, zeros_
from ...utils.log import logger
from .. import PretrainedModel
from ..clip.modeling import CLIPVisionTransformer as CMSIMLockVisionTransformer
from ..clip.modeling import ModifiedResNet, clip_loss
from ..configuration_utils import PretrainedConfig
from ..ernie.modeling import ErnieModel
from ..model_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
)
from ..utils import resolve_cache_dir
from .configuration import CMSIMLockConfig, CMSIMLockTextConfig, CMSIMLockVisionConfig

__all__ = [
    "CMSIMLockModel",
    "CMSIMLockTextModel",
    "CMSIMLockVisionModelWithProjection",
    "CMSIMLockPretrainedModel",
]

CMSIM_LOCK_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # vit model
    "PaddlePaddle/cmsim-lock-vit-base-patch32",  # ViT-B/32
    "PaddlePaddle/cmsim-lock-vit-base-patch16",  # ViT-B/16
    # resnet model
    "PaddlePaddle/cmsim-lock-rn50",  # RN50
]


def quick_gelu(x):
    return x * F.sigmoid(1.702 * x)


F.quick_gelu = quick_gelu


@dataclass
class CMSIMLockVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`paddle.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[paddle.Tensor] = None
    last_hidden_state: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class CMSIMLockOutput(ModelOutput):
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
            The image embeddings obtained by applying the projection layer to the pooled output of [`CMSIMLockVisionTransformer`].
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


class CLSMeanPooler(nn.Layer):
    def __init__(self, config: CMSIMLockTextConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.projection_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        hidden_len = attention_mask.sum(-1).unsqueeze(-1)
        cls_token = hidden_states[:, 0]
        mp_fused_hidden = hidden_states[:, 1:].sum(1) / (hidden_len - 1)
        final_hidden = (cls_token + mp_fused_hidden) / 2
        pooled_output = self.dense(final_hidden)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieTextModel(ErnieModel):
    r"""
    The bare ERNIE Text Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`CMSIMLockTextConfig`):
            An instance of CMSIMLockTextConfig used to construct ErnieTextModel
    """

    def __init__(self, config: CMSIMLockTextConfig):
        super().__init__(config)
        self.pooler = CLSMeanPooler(config)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        task_type_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `[batch_size, num_tokens]` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                We use whole-word-mask in ERNIE, so the whole word will have the same value. For example, "使用" as a word,
                "使" and "用" will have the same value.
                Defaults to `None`, which means nothing needed to be prevented attention to.
             inputs_embeds (Tensor, optional):
                If you want to control how to convert `inputs_ids` indices into associated vectors, you can
                pass an embedded representation directly instead of passing `inputs_ids`.
            past_key_values (tuple(tuple(Tensor)), optional):
                The length of tuple equals to the number of layers, and each inner
                tuple haves 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`)
                which contains precomputed key and value hidden states of the attention blocks.
                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, optional):
                If set to `True`, `past_key_values` key value states are returned.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.cmsim_lock.modeling import ErnieTextModel
                from paddlenlp.transformers import CMSIMLockTokenizer

                tokenizer = ErnieTokenizer.from_pretrained(PaddlePaddle/cmsim-lock-vit-base-patch32)
                model = ErnieTextModel.from_pretrained(PaddlePaddle/cmsim-lock-vit-base-patch32)
                model.eval()

                inputs = tokenizer("欢迎使用paddlepaddle和paddlenlp!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
                print(outputs.last_hidden_state.shape)
                print(outputs.pooler_output.shape)
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        # init the default bool value
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False
        use_cache = use_cache if use_cache is not None else False
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if attention_mask is None:
            attention_mask_2d = (input_ids != self.pad_token_id).astype(self.pooler.dense.weight.dtype)
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
            )

            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = paddle.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = paddle.concat([past_mask, attention_mask], axis=-1)

        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask_2d = attention_mask.astype(self.pooler.dense.weight.dtype)
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4

        attention_mask.stop_gradient = True

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
            cache=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = encoder_outputs * attention_mask_2d.unsqueeze(-1)
            pooled_output = self.pooler(sequence_output, attention_mask_2d)
            return (sequence_output, pooled_output)
        else:
            sequence_output = encoder_outputs[0] * attention_mask_2d.unsqueeze(-1)
            pooled_output = self.pooler(sequence_output, attention_mask_2d)
            if not return_dict:
                return (sequence_output, pooled_output) + encoder_outputs[1:]
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )


class CMSIMLockPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained CMSIMLock models. It provides CMSIMLock related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    config_class = CMSIMLockConfig
    base_model_prefix = "cmsim_lock"
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
        if isinstance(layer, CMSIMLockVisionTransformer):
            factor = self.config.initializer_factor
            # find nn.LayerNorm
            for sub_layer in layer.sublayers():
                if isinstance(sub_layer, nn.LayerNorm):
                    sub_layer._epsilon = layer.config.layer_norm_eps

            vision_embed_dim = layer.config.hidden_size
            vision_layers = layer.config.num_hidden_layers
            initializer_range = layer.config.initializer_range

            # vision embedding
            normal_(layer.class_embedding, std=vision_embed_dim**-0.5 * factor)
            normal_(layer.conv1.weight, std=initializer_range * factor)
            normal_(layer.positional_embedding.weight, std=initializer_range * factor)

            # init Attention + MLP
            for sub_layer in layer.sublayers():
                if isinstance(sub_layer, nn.TransformerEncoderLayer):
                    # self_attn
                    in_proj_std = (sub_layer.self_attn.embed_dim**-0.5) * ((2 * vision_layers) ** -0.5) * factor
                    out_proj_std = (sub_layer.self_attn.embed_dim**-0.5) * factor
                    normal_(sub_layer.self_attn.q_proj.weight, std=in_proj_std)
                    normal_(sub_layer.self_attn.k_proj.weight, std=in_proj_std)
                    normal_(sub_layer.self_attn.v_proj.weight, std=in_proj_std)
                    normal_(sub_layer.self_attn.out_proj.weight, std=out_proj_std)
                    # ffn
                    in_proj_std = (sub_layer._config["d_model"] ** -0.5) * ((2 * vision_layers) ** -0.5) * factor
                    fc_std = (2 * sub_layer._config["d_model"]) ** -0.5 * factor
                    normal_(sub_layer.linear1.weight, std=fc_std)
                    normal_(sub_layer.linear2.weight, std=in_proj_std)

        elif isinstance(layer, ModifiedResNet):
            if layer.attnpool is not None:
                std = layer.output_dim**-0.5
                normal_(layer.attnpool.q_proj.weight, std=std)
                normal_(layer.attnpool.k_proj.weight, std=std)
                normal_(layer.attnpool.v_proj.weight, std=std)
                normal_(layer.attnpool.c_proj.weight, std=std)

            for resnet_block in [layer.layer1, layer.layer2, layer.layer3, layer.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        zeros_(param)

        elif isinstance(layer, ErnieTextModel):
            # find nn.LayerNorm
            for sub_layer in layer.sublayers():
                if isinstance(sub_layer, nn.LayerNorm):
                    sub_layer._epsilon = layer.config.layer_norm_eps
                elif isinstance(layer, (nn.Linear, nn.Embedding)):
                    normal_(layer.weight, mean=0.0, std=layer.config.initializer_range)

        elif isinstance(layer, CMSIMLockModel):
            if hasattr(layer, "vision_projection"):
                normal_(layer.vision_projection, std=layer.vision_embed_dim**-0.5 * self.config.initializer_factor)
        elif isinstance(layer, CMSIMLockVisionModelWithProjection):
            if hasattr(layer, "vision_projection"):
                normal_(layer.vision_projection, std=self.config.hidden_size**-0.5 * self.config.initializer_factor)

        if isinstance(layer, nn.LayerNorm):
            zeros_(layer.bias)
            ones_(layer.weight)

        if isinstance(layer, nn.Linear) and layer.bias is not None:
            zeros_(layer.bias)

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


class CMSIMLockModel(CMSIMLockPretrainedModel):
    r"""
    The bare CMSIMLock Model outputting logits_per_image and logits_per_text.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`CMSIMLockConfig`):
            An instance of CMSIMLockConfig used to construct CMSIMLockModel.
    """
    config_class = CMSIMLockConfig

    def __init__(self, config: CMSIMLockConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CMSIMLockTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CMSIMLockTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CMSIMLockVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CMSIMLockVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = ErnieTextModel(text_config)
        # CMSIMLock donot use text_projection

        # support resnet vision model
        if isinstance(vision_config.num_hidden_layers, (tuple, list)):
            if vision_config.num_attention_heads is None:
                vision_heads = vision_config.hidden_size * 32 // 64
            else:
                vision_heads = vision_config.num_attention_heads
            self.vision_model = ModifiedResNet(
                layers=vision_config.num_hidden_layers,
                output_dim=self.projection_dim,
                heads=vision_heads,
                input_resolution=vision_config.image_size,
                width=vision_config.hidden_size,
            )
        else:
            self.vision_model = CMSIMLockVisionTransformer(vision_config)
            self.vision_projection = paddle.create_parameter(
                (self.vision_embed_dim, self.projection_dim), paddle.get_default_dtype()
            )

        self.logit_scale = config.logit_scale_init_value
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
                [`CMSIMLockFeatureExtractor`]. See [`CMSIMLockFeatureExtractor.__call__`] for details.
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
            applying the projection layer to the pooled output of [`CMSIMLockVisionModel`].

        Examples:
            .. code-block::

                import requests
                from PIL import Image
                from paddlenlp.transformers import CMSIMLockProcessor, CMSIMLockModel

                model = CMSIMLockModel.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")
                processor = CMSIMLockProcessor.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")

                url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                image = Image.open(requests.get(url, stream=True).raw)
                inputs = processor(images=image, return_tensors="pd")
                image_features = model.get_image_features(**inputs)

        """
        if isinstance(self.vision_model, ModifiedResNet):
            return self.vision_model(pixel_values)
        else:
            # Use CMSIMLock model's config for some fields (if specified) instead of those of vision & text components.
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

            pooled_output = vision_outputs[1]  # pooled_output
            image_features = paddle.matmul(pooled_output, self.vision_projection)

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
                Indices can be obtained using [`CMSIMLockTokenizer`].
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
                Indices of tasks of each input sequence tokens in the task embeddings (ErnieTextModel). Selected in
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
            the pooled output of [`ErnieTextModel`].

        Example:
            .. code-block::

                from paddlenlp.transformers import CMSIMLockModel, CMSIMLockTokenizer

                model = CMSIMLockModel.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")
                tokenizer = CMSIMLockTokenizer.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")

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
    ) -> Union[Tuple, CMSIMLockOutput]:
        r"""
        The CMSIMLockModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
                Its data type should be `int64` and it has a shape of [text_batch_size, sequence_length].
            pixel_values (Tensor):
                Pixel values. Padding will be ignored by default should you provide it.
                Its data type should be `float32` and it has a shape of [image_batch_size, num_channels, height, width].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention (ErnieModel) to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
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
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`CMSIMLockOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`CMSIMLockOutput` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`CMSIMLockOutput`.

        Example:
            .. code-block::

                import requests
                import paddle.nn.functional as F
                from PIL import Image
                from paddlenlp.transformers import CMSIMLockModel, CMSIMLockProcessor

                processor = CMSIMLockProcessor.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")
                model = CMSIMLockModel.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")
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

        if isinstance(self.vision_model, ModifiedResNet):
            vision_outputs = None
            image_embeds = self.vision_model(pixel_values)
        else:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeds = vision_outputs[1]
            image_embeds = paddle.matmul(image_embeds, self.vision_projection)

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
        text_embeds = text_outputs[1]
        # CMSIMLock donot use text_projection

        # normalized features
        image_embeds = F.normalize(image_embeds, axis=-1)
        text_embeds = F.normalize(text_embeds, axis=-1)

        if dist.get_world_size() > 1:
            feature_list_img = []
            feature_list_txt = []
            dist.all_gather(feature_list_img, image_embeds)
            dist.all_gather(feature_list_txt, text_embeds)
            image_embeds = paddle.concat(feature_list_img, axis=0)
            text_embeds = paddle.concat(feature_list_txt, axis=0)

        # cosine similarity as logits
        logits_per_text = paddle.matmul(text_embeds / self.logit_scale, image_embeds, transpose_y=True)
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CMSIMLockOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class CMSIMLockTextModel(CMSIMLockPretrainedModel):
    r"""
    The text model from CMSIMLock without any head or projection on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`CMSIMLockTextConfig`):
            An instance of CMSIMLockTextConfig used to construct CMSIMLockTextModel.
    """

    config_class = CMSIMLockTextConfig

    def __init__(self, config: CMSIMLockTextConfig):
        super().__init__(config)
        self.text_model = ErnieTextModel(config)

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
                Indices can be obtained using [`CMSIMLockTokenizer`].
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
        >>> from paddlenlp.transformers import CMSIMLockTokenizer, CMSIMLockTextModel

        >>> model = CMSIMLockTextModel.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")
        >>> model.eval()
        >>> tokenizer = CMSIMLockTokenizer.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")

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


class CMSIMLockVisionModelWithProjection(CMSIMLockPretrainedModel):
    r"""
    CMSIMLock Vision Model with a projection layer on top (a linear layer on top of the pooled output).

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`CMSIMLockVisionConfig`):
            An instance of CMSIMLockVisionConfig used to construct CMSIMLockVisionModelWithProjection.
    """
    config_class = CMSIMLockVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CMSIMLockVisionConfig):
        super().__init__(config)

        # support resnet vision model
        if isinstance(config.num_hidden_layers, (tuple, list)):
            if config.num_attention_heads is None:
                vision_heads = config.hidden_size * 32 // 64
            else:
                vision_heads = config.num_attention_heads
            self.vision_model = ModifiedResNet(
                layers=config.num_hidden_layers,
                output_dim=config.projection_dim,
                heads=vision_heads,
                input_resolution=config.image_size,
                width=config.hidden_size,
            )
        else:
            self.vision_model = CMSIMLockVisionTransformer(config)
            self.vision_projection = paddle.create_parameter(
                (config.hidden_size, config.projection_dim), paddle.get_default_dtype()
            )

        self.init_weights()

    def get_input_embeddings(self) -> nn.Layer:
        if isinstance(self.vision_model, CMSIMLockVisionTransformer):
            return self.vision_model.conv1
        else:
            return None

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CMSIMLockVisionModelOutput]:
        r"""
        Args:
            pixel_values (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
                [`CMSIMLockFeatureExtractor`]. See [`CMSIMLockFeatureExtractor.__call__`] for details.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`CMSIMLockVisionModelOutput`] instead of a plain tuple.

        Returns:
            An instance of :class:`CMSIMLockVisionModelOutput` if `return_dict=True`. Otherwise it returns a tuple of tensors
            corresponding to ordered and not None (depending on the input arguments) fields of :class:`CMSIMLockVisionModelOutput`.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import CMSIMLockProcessor, CMSIMLockVisionModelWithProjection

        >>> model = CMSIMLockVisionModelWithProjection.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")
        >>> model.eval()
        >>> processor = CMSIMLockProcessor.from_pretrained("PaddlePaddle/cmsim-lock-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pd")

        >>> outputs = model(**inputs)
        >>> image_embeds = outputs.image_embeds
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(self.vision_model, ModifiedResNet):
            image_embeds = self.vision_model(pixel_values)
            if not return_dict:
                return (image_embeds,)
            else:
                return CMSIMLockVisionModelOutput(image_embeds=image_embeds)
        else:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = vision_outputs[1]  # pooled_output

            image_embeds = paddle.matmul(pooled_output, self.vision_projection)

            if not return_dict:
                outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
                return tuple(output for output in outputs if output is not None)

            return CMSIMLockVisionModelOutput(
                image_embeds=image_embeds,
                last_hidden_state=vision_outputs.last_hidden_state,
                hidden_states=vision_outputs.hidden_states,
                attentions=vision_outputs.attentions,
            )
