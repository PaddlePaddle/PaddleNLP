# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn

from ...models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler

################################################################################
# Code for the text transformer model
################################################################################
""" 
Paddle LDMBERT model.
"""
from paddlenlp.transformers import PretrainedModel, register_base_model, PretrainedTokenizer
from paddlenlp.transformers.model_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class LDMBertPretrainedModel(PretrainedModel):
    pretrained_init_configuration = {}
    pretrained_resource_files_map = {}
    base_model_prefix = "ldmbert"

    def init_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(
                paddle.normal(mean=0.0,
                              std=self.initializer_range if hasattr(
                                  self, "initializer_range") else
                              self.ldmbert.config["initializer_range"],
                              shape=layer.weight.shape))


class LDMBertEmbeddings(nn.Layer):

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.0,
                 max_position_embeddings=512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embedings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 head_dim=64):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         attn_dropout, act_dropout, normalize_before,
                         weight_attr, bias_attr)
        # update self attn
        self.self_attn = LDMBertAttention(d_model,
                                          head_dim,
                                          nhead,
                                          dropout=attn_dropout,
                                          weight_attr=weight_attr,
                                          bias_attr=False)


@register_base_model
class LDMBertModel(LDMBertPretrainedModel):

    def __init__(self,
                 vocab_size=30522,
                 max_position_embeddings=77,
                 encoder_layers=32,
                 encoder_ffn_dim=5120,
                 encoder_attention_heads=8,
                 head_dim=64,
                 activation_function="gelu",
                 d_model=1280,
                 dropout=0.0,
                 attention_dropout=0.0,
                 activation_dropout=0.0,
                 init_std=0.02,
                 pad_token_id=0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = init_std
        self.embeddings = LDMBertEmbeddings(vocab_size, d_model, dropout,
                                            max_position_embeddings)
        encoder_layer = TransformerEncoderLayer(d_model,
                                                encoder_attention_heads,
                                                encoder_ffn_dim,
                                                dropout=dropout,
                                                activation=activation_function,
                                                attn_dropout=attention_dropout,
                                                act_dropout=activation_dropout,
                                                normalize_before=True,
                                                head_dim=head_dim)

        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):

        if attention_mask is not None and attention_mask.ndim == 2:
            # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
            attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(
                paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4

        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids)

        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = self.final_layer_norm(encoder_outputs)
            return (sequence_output, )
        else:
            sequence_output = encoder_outputs[0]
            sequence_output = self.final_layer_norm(sequence_output)
            if not return_dict:
                return (sequence_output, ) + encoder_outputs[1:]
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions)


class LDMBertAttention(nn.MultiHeadAttention):

    def __init__(self,
                 embed_dim,
                 head_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None):
        super().__init__(embed_dim, num_heads, dropout, kdim, vdim,
                         need_weights, weight_attr, bias_attr)
        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but recieved {}".format(embed_dim))
        assert num_heads > 0, ("Expected num_heads to be greater than 0, "
                               "but recieved {}".format(num_heads))

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim,
                                self.inner_dim,
                                weight_attr,
                                bias_attr=bias_attr)
        self.k_proj = nn.Linear(self.kdim,
                                self.inner_dim,
                                weight_attr,
                                bias_attr=bias_attr)
        self.v_proj = nn.Linear(self.vdim,
                                self.inner_dim,
                                weight_attr,
                                bias_attr=bias_attr)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim, weight_attr)


class LDMBertModelForMaskedLM(LDMBertPretrainedModel):

    def __init__(self, ldmbert):
        super().__init__()
        self.ldmbert = ldmbert
        self.to_logits = nn.Linear(ldmbert.config["hidden_size"],
                                   ldmbert.config["vocab_size"])
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.ldmbert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class LDMTextToImagePipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) Model to encode and decode images to and from latent representations.
        bert ([`LDMBertModel`]):
            Text-encoder model based on [BERT](https://huggingface.co/docs/transformers/model_doc/bert) architecture.
        tokenizer (`transformers.BertTokenizer`):
            Tokenizer of class
            [BertTokenizer](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vqvae: Union[VQModel, AutoencoderKL],
        bert: PretrainedModel,
        tokenizer: PretrainedTokenizer,
        unet: Union[UNet2DModel, UNet2DConditionModel],
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(vqvae=vqvae,
                              bert=bert,
                              tokenizer=tokenizer,
                              unet=unet,
                              scheduler=scheduler)

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 1.0,
        eta: Optional[float] = 0.0,
        seed: Optional[int] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 256):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 256):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt` at
                the, usually at the expense of lower image quality.
            generator (`int`, *optional*):
                A random seed.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        if seed is not None:
            paddle.seed(seed)
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # get unconditional embeddings for classifier free guidance
        if guidance_scale != 1.0:
            uncond_input = self.tokenizer([""] * batch_size,
                                          padding="max_length",
                                          max_length=77,
                                          return_tensors="pd")
            uncond_embeddings = self.bert(uncond_input.input_ids)[0]

        # get prompt text embeddings
        text_input = self.tokenizer(prompt,
                                    padding="max_length",
                                    max_length=77,
                                    return_tensors="pd")
        text_embeddings = self.bert(text_input.input_ids)[0]

        latents = paddle.randn(
            [batch_size, self.unet.in_channels, height // 8, width // 8], )
        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale == 1.0:
                # guidance_scale of 1 means no guidance
                latents_input = latents
                context = text_embeddings
            else:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                latents_input = paddle.concat([latents] * 2)
                context = paddle.concat([uncond_embeddings, text_embeddings])

            # predict the noise residual
            noise_pred = self.unet(latents_input,
                                   t,
                                   encoder_hidden_states=context).sample
            # perform guidance
            if guidance_scale != 1.0:
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_prediction_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vqvae.decode(latents).sample

        image = (image / 2 + 0.5).clip(0, 1)
        image = image.transpose([0, 2, 3, 1]).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, )

        return ImagePipelineOutput(images=image)
