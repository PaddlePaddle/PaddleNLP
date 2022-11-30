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

from typing import Any, Tuple, Optional
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
from dataclasses import dataclass

from .. import PretrainedModel, register_base_model
from ..model_outputs import BaseModelOutputWithPoolingAndCrossAttentions, ModelOutput
from ..ernie.modeling import ErnieModel
from ..clip.modeling import VisionTransformer, clip_loss
from ..guided_diffusion_utils import (
    DiscoDiffusionMixin,
    create_gaussian_diffusion,
    create_unet_model,
    create_secondary_model,
)

__all__ = [
    "ErnieViLModel",
    "ErnieViLPretrainedModel",
    "ErnieViLForImageGeneration",
]


# set attr
def quick_gelu(x):
    return x * F.sigmoid(1.702 * x)


F.quick_gelu = quick_gelu


@dataclass
class ErnieViLOutput(ModelOutput):
    """
    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`paddle.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`paddle.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`paddle.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`ErnieModel`].
        image_embeds(`paddle.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`VisionTransformer`].
        text_model_output(:class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`ErnieModel`].
        vision_model_output(:class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`VisionTransformer`].
    """

    loss: Optional[paddle.Tensor] = None
    logits_per_image: paddle.Tensor = None
    logits_per_text: paddle.Tensor = None
    text_embeds: paddle.Tensor = None
    image_embeds: paddle.Tensor = None
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    vision_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None

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

    pretrained_init_configuration = {
        "ernie_vil-2.0-base-zh": {
            "image_resolution": 224,
            "vision_layers": 12,
            "vision_heads": 12,
            "vision_embed_dim": 768,
            "vision_patch_size": 16,
            "vision_mlp_ratio": 4,
            "vision_hidden_act": "quick_gelu",
            "vision_epsilon": 1e-6,
            "vocab_size": 40000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 2048,
            "type_vocab_size": 0,
            "task_type_vocab_size": 3,
            "hidden_act": "gelu",
            "task_id": 0,
            "use_task_id": False,
            "text_epsilon": 1e-5,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "disco_diffusion_ernie_vil-2.0-base-zh": {
            "image_resolution": 224,
            "vision_layers": 12,
            "vision_heads": 12,
            "vision_embed_dim": 768,
            "vision_patch_size": 16,
            "vision_mlp_ratio": 4,
            "vision_hidden_act": "quick_gelu",
            "vision_epsilon": 1e-6,
            "vocab_size": 40000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 2048,
            "type_vocab_size": 0,
            "task_type_vocab_size": 3,
            "hidden_act": "gelu",
            "task_id": 0,
            "use_task_id": False,
            "text_epsilon": 1e-5,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    pretrained_resource_files_map = {
        "model_state": {
            "ernie_vil-2.0-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_vil/ernie_vil-2.0-base-zh/model_state.pdparams",
            "disco_diffusion_ernie_vil-2.0-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_vil/disco_diffusion_ernie_vil-2.0-base-zh/model_state.pdparams",
        }
    }
    base_model_prefix = "ernie_vil"

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, VisionTransformer):
            # find nn.LayerNorm
            for sub_layer in layer.sublayers():
                if isinstance(sub_layer, nn.LayerNorm):
                    sub_layer._epsilon = (
                        self.vision_epsilon
                        if hasattr(self, "vision_epsilon")
                        else self.ernie_vil.config["vision_epsilon"]
                    )

        elif isinstance(layer, ErnieModel):
            # find nn.LayerNorm
            for sub_layer in layer.sublayers():
                if isinstance(sub_layer, nn.LayerNorm):
                    sub_layer._epsilon = (
                        self.text_epsilon if hasattr(self, "text_epsilon") else self.ernie_vil.config["text_epsilon"]
                    )
                elif isinstance(layer, (nn.Linear, nn.Embedding)):
                    if isinstance(layer.weight, paddle.Tensor):
                        layer.weight.set_value(
                            paddle.normal(
                                mean=0.0,
                                std=self.initializer_range
                                if hasattr(self, "initializer_range")
                                else self.ernie_vil.config["initializer_range"],
                                shape=layer.weight.shape,
                            )
                        )


@register_base_model
class ErnieViLModel(ErnieViLPretrainedModel):
    r"""
    The bare ErnieViL Model outputting logits_per_image and logits_per_text.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    Args:
        image_resolution (int, optional):
            The size (resolution) of each image.
            Defaults to `224`.
        vision_layers (int, optional):
            Number of hidden layers in the vision model.
            Defaults to `12`.
        vision_heads (int, optional):
            Number of attention heads for each attention layer in the vision model.
            Defaults to `12`.
        vision_embed_dim (int, optional):
            Dimensionality of the embedding layer and encoder layers in vision model.
            Defaults to `768`.
        vision_patch_size(int, optional):
            The size (resolution) of each patch.
            Defaults to `32`.
        vision_mlp_ratio(int, optional):
            The ratio between dim_feedforward and vision_hidden_dim. `radio = dim_feedforward/vision_hidden_dim`
            Defaults to `4`.
        vision_hidden_act (str, optional):
            The non-linear activation function of the ffn layer in the vision model.
            ``"gelu"``, ``"relu"``, ``"quick_gelu"`` and any other paddle supported activation functions are supported.
            Defaults to `"quick_gelu"`.
        vision_epsilon (float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for initializing layer normalization layers.
            Default to `1e-6`.
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `ErnieModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ErnieModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layers and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `2048`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`.
            Defaults to `4`.
        task_type_vocab_size (int, optional):
            The vocabulary size of the `task_ids`.
            Defaults to `3`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer of Ernie.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"gelu"`.
        task_id (int, optional):
            Task id. Defaults to `0`.
        use_task_id (bool, optional):
            Whether or not use task_id. Defaults to `False`.
        text_epsilon (float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for initializing layer normalization layers.
            Default to `1e-5`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer for initializing all weight matrices.
            Defaults to `0.02`.
        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

    """

    def __init__(
        self,
        # vision
        image_resolution: int = 224,
        vision_layers: int = 12,
        vision_heads: int = 12,
        vision_embed_dim: int = 768,
        vision_patch_size: int = 16,
        vision_mlp_ratio: int = 4,
        vision_hidden_act: str = "quick_gelu",
        vision_epsilon: float = 1e-6,
        # ernie
        vocab_size: int = 40000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 4,
        task_type_vocab_size: int = 3,
        hidden_act: str = "gelu",
        task_id: int = 0,
        use_task_id: bool = False,
        text_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.vision_epsilon = vision_epsilon
        self.text_epsilon = text_epsilon
        self.vision_model = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_embed_dim,
            layers=vision_layers,
            heads=vision_heads,
            activation=vision_hidden_act,
            mlp_ratio=vision_mlp_ratio,
            normalize_before=True,
        )

        self.text_model = ErnieModel(
            vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            task_type_vocab_size=task_type_vocab_size,
            task_id=task_id,
            use_task_id=use_task_id,
        )

        self.temperature = self.create_parameter(
            shape=(1,), default_initializer=nn.initializer.Constant(2.65926), dtype=paddle.get_default_dtype()
        )

        self.apply(self.init_weights)

    def get_image_features(
        self, pixel_values=None, output_attentions=False, output_hidden_states=False, return_dict=False
    ):
        r"""
        Returns:
            image_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`VisionTransformer`].

        Examples:
            .. code-block::

                import requests
                from PIL import Image
                from paddlenlp.transformers import ErnieViLProcessor, ErnieViLModel

                model = ErnieViLModel.from_pretrained("ernie_vil-2.0-base-zh")
                processor = ErnieViLProcessor.from_pretrained("ernie_vil-2.0-base-zh")

                url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                image = Image.open(requests.get(url, stream=True).raw)
                inputs = processor(images=image, return_tensors="pd")
                image_features = model.get_image_features(**inputs)

        """
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
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        task_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        Returns:
            text_features (`paddle.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`ErnieModel`].

        Example:
            .. code-block::

                from paddlenlp.transformers import ErnieViLModel, ErnieViLTokenizer

                model = ErnieViLModel.from_pretrained("ernie_vil-2.0-base-zh")
                tokenizer = ErnieViLTokenizer.from_pretrained("ernie_vil-2.0-base-zh")

                inputs = tokenizer(["一只猫的照片", "一条狗的照片"], padding=True, return_tensors="pd")
                text_features = model.get_text_features(**inputs)

        """
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
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        task_type_ids=None,
        return_loss=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
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

                processor = ErnieViLProcessor.from_pretrained('ernie_vil-2.0-base-zh')
                model = ErnieViLModel.from_pretrained('ernie_vil-2.0-base-zh')

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


class ErnieViLForImageGeneration(ErnieViLPretrainedModel, DiscoDiffusionMixin):
    r"""
    ErnieViLModel with diffusion model on top.
    Args:
        ernie_vil (:class:`ErnieViLModel`):
            An instance of ErnieViLModel.
    """

    def __init__(self, ernie_vil):
        super().__init__()
        self.ernie_vil = ernie_vil
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
            model_name_or_path = 'disco_diffusion_ernie_vil-2.0-base-zh'
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
        target_text_embeds = self.get_text_features(
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

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e
