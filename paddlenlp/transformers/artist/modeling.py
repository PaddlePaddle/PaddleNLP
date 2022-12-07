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
from ..dallebart.modeling import VQGanDetokenizer
from ..gpt.modeling import GPTLMHeadModel, GPTLMHead, GPTModel

__all__ = [
    "ArtistModel",
    "ArtistForImageGeneration",
    "ArtistForConditionalGeneration",
]

# set gelu_new
F.gelu_python = F.gelu

pretrained_init_configuration = {
    "pai-painter-base-zh": {
        "vocab_size": 37512,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
    "pai-painter-painting-base-zh": {
        "vocab_size": 37512,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
    "pai-painter-scenery-base-zh": {
        "vocab_size": 37512,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
    "pai-painter-commercial-base-zh": {
        "vocab_size": 37512,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
    "pai-painter-large-zh": {
        "vocab_size": 37512,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
}
pretrained_resource_files_map = {
    "model_state": {
        "pai-painter-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-base-zh/model_state.pdparams",
        "pai-painter-painting-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-painting-base-zh/model_state.pdparams",
        "pai-painter-scenery-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-scenery-base-zh/model_state.pdparams",
        "pai-painter-commercial-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-commercial-base-zh/model_state.pdparams",
        "pai-painter-large-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-large-zh/model_state.pdparams",
    }
}


class ArtistModel(GPTModel):
    pretrained_init_configuration = pretrained_init_configuration
    pretrained_resource_files_map = pretrained_resource_files_map


class ArtistForConditionalGeneration(GPTLMHeadModel):
    """
    The ArtistT(GPT) Model with a `language modeling` head on top.

    Args:
        gpt (:class:`ArtistModel`):
            An instance of :class:`ArtistModel`.

    """

    pretrained_init_configuration = pretrained_init_configuration
    pretrained_resource_files_map = pretrained_resource_files_map

    def __init__(self, gpt):
        super().__init__(gpt)
        self.lm_head = GPTLMHead(self.gpt.config["hidden_size"], self.gpt.config["vocab_size"])
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
    pretrained_init_configuration = pretrained_init_configuration
    pretrained_resource_files_map = pretrained_resource_files_map

    def __init__(self, gpt, image_vocab_size=16384):
        super().__init__(gpt)
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
        use_faster=False,
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
            use_faster: (bool, optional): Whether to use faster entry of model
                for FasterGeneration. Default to False.
            use_fp16_decoding: (bool, optional): Whether to use fp16 for decoding.
                Only works when faster entry is avalible. Default to False.

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
            use_faster=use_faster,
            use_fp16_decoding=use_fp16_decoding,
            seq_len=paddle.ones((input_ids.shape[0],), dtype="int32") * 32,
            **kwargs,
        )[0]
        images = self.vqgan_detokenizer(image_tokens)
        # images shape [bs, num_return_sequences, 256, 256, 3]
        images = images.reshape([-1, num_return_sequences, images.shape[1], images.shape[2], images.shape[3]])
        images = ((images + 1.0) * 127.5).clip(0, 255).astype("uint8")
        return images
