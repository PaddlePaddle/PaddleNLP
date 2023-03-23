# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 Alibaba PAI team. All Rights Reserved.
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


import paddle
import paddle.nn.functional as F

from ...utils.log import logger
from ..dallebart.modeling import VQGanDetokenizer
from ..gpt.modeling import GPTLMHead, GPTLMHeadModel, GPTModel
from .configuration import (
    ARTIST_PRETRAINED_INIT_CONFIGURATION,
    ARTIST_PRETRAINED_RESOURCE_FILES_MAP,
    ArtistConfig,
)

__all__ = [
    "ArtistModel",
    "ArtistForImageGeneration",
    "ArtistForConditionalGeneration",
]

# set gelu_new
F.gelu_python = F.gelu


class ArtistModel(GPTModel):
    config_class = ArtistConfig
    pretrained_init_configuration = ARTIST_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ARTIST_PRETRAINED_RESOURCE_FILES_MAP


class ArtistForConditionalGeneration(GPTLMHeadModel):
    """
    The ArtistT(GPT) Model with a `language modeling` head on top.

    Args:
        gpt (:class:`ArtistModel`):
            An instance of :class:`ArtistModel`.

    """

    config_class = ArtistConfig
    pretrained_init_configuration = ARTIST_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ARTIST_PRETRAINED_RESOURCE_FILES_MAP

    def __init__(self, config: ArtistConfig):
        super().__init__(config)
        self.lm_head = GPTLMHead(config.hidden_size, config.vocab_size)
        self.apply(self.init_weights)

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        # we don't use attention_mask
        attention_mask = paddle.zeros_like(input_ids, dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])


class ArtistForImageGeneration(ArtistForConditionalGeneration):
    r"""
    Artist Model with a `language modeling` head and `VQGanTokenizer` on top.
    Args:
        gpt (:class:`ArtistModel`):
            An instance of ArtistModel.
        image_vocab_size (int, optional):
            The vocabulary size of image.
            Defaults to `16384`.
    """
    config_class = ArtistConfig
    pretrained_init_configuration = ARTIST_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ARTIST_PRETRAINED_RESOURCE_FILES_MAP

    def __init__(self, config: ArtistConfig, image_vocab_size=16384):
        super().__init__(config)
        logger.warning(
            f"'{__class__.__name__}' is now deprecated and will be removed after v2.6.0"
            "Please Refer to PPDiffusers for Text-to-Image Capabilities"
        )
        self.vqgan_detokenizer = VQGanDetokenizer(image_vocab_size, 256)

    @paddle.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        num_return_sequences=1,
        use_fast=False,
        use_fp16_decoding=False,
        **kwargs
    ):
        r"""
        The ArtistForImageGeneration generate method.
        Args:
            input_ids (Tensor):
                See :class:`ArtistForConditionalGeneration`.
            attention_mask (Tensor, optional):
                See :class:`ArtistForConditionalGeneration`.
            top_k (int, optional): The number of highest probability tokens to
                keep for top-k-filtering in the "sampling" strategy. Default to
                0, which means no effect.
            top_p (float, optional): The cumulative probability for
                top-p-filtering in the "sampling" strategy. The value should
                satisfy :math:`0 <= top\_p < 1`. Default to 1.0, which means no
                effect.
            temperature (float, optional): The value used to module the next
                token probabilities in the "sampling" strategy. Default to 1.0,
                which means no effect.
            num_return_sequences (int, optional): The number of returned
                sequences for each sequence in the batch. Default to 1.
            use_fast: (bool, optional): Whether to use fast entry of model
                for FastGeneration. Default to False.
            use_fp16_decoding: (bool, optional): Whether to use fp16 for decoding.
                Only works when fast entry is avalible. Default to False.

        Returns:
            Tensor: Returns tensor `images`, which is the output of :class:`VQGanDetokenizer`.
            Its data type should be uint8 and has a shape of [batch_size, num_return_sequences, 256, 256, 3].

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import AutoModelForImageGeneration, AutoTokenizer
                from PIL import Image

                # Initialize the model and tokenizer
                model_name_or_path = 'pai-painter-painting-base-zh'
                model = AutoModelForImageGeneration.from_pretrained(model_name_or_path)
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                model.eval()

                # Prepare the model inputs.
                prompts = ["风阁水帘今在眼，且来先看早梅红", "见说春风偏有贺，露花千朵照庭闹"]
                tokenized_inputs = tokenizer(prompts, return_tensors="pd")
                top_k = 32
                num_return_sequences = 4
                images = model.generate(**tokenized_inputs,
                                      top_k=top_k,
                                      num_return_sequences=num_return_sequences)
                print(images.shape) # [2, 4, 256, 256, 3]
                # [2, 256, 4*256, 3]
                images = images.numpy().transpose([0, 2, 1, 3,
                                        4]).reshape([-1, images.shape[-3],
                                                    num_return_sequences * images.shape[-2],
                                                    images.shape[-1]])
                for i, image in enumerate(images):
                    image = Image.fromarray(image)
                    image.save(f"figure_{i}.png")
        """
        image_tokens = super().generate(
            input_ids=input_ids,
            max_length=256,
            decode_strategy="sampling",
            attention_mask=attention_mask,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            use_fast=use_fast,
            use_fp16_decoding=use_fp16_decoding,
            seq_len=paddle.ones((input_ids.shape[0],), dtype="int32") * 32,
            **kwargs,
        )[0]
        images = self.vqgan_detokenizer(image_tokens)
        # images shape [bs, num_return_sequences, 256, 256, 3]
        images = images.reshape([-1, num_return_sequences, images.shape[1], images.shape[2], images.shape[3]])
        images = ((images + 1.0) * 127.5).clip(0, 255).astype("uint8")
        return images
