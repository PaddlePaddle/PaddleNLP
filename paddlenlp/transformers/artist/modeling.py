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
from ..dallebart.modeling import VQGanDetokenizer
from ..gpt.modeling import GPTLMHeadModel, GPTLMHead, GPTModel as ArtistModel

__all__ = [
    'ArtistModel',
    'ArtistLMHeadModel',
    'ArtistForImageGeneration',
    'ArtistForCausalLM',
]


class ArtistLMHeadModel(GPTLMHeadModel):
    """
    The ArtistT(GPT) Model with a `language modeling` head on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """
    pretrained_init_configuration = {
        "pai-painter-base-zh": {
            "vocab_size": 37512,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 288,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 16384,  # 0 + 16384
            "eos_token_id": 10486,  # 102 + 16384
            "bos_token_id": 10485,  # 101 + 16384
            "eol_token_id": 10486,  # 102 + 16384
        },
        "pai-painter-painting-base-zh": {
            "vocab_size": 37512,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 288,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 16384,  # 0 + 16384
            "eos_token_id": 10486,  # 102 + 16384
            "bos_token_id": 10485,  # 101 + 16384
            "eol_token_id": 10486,  # 102 + 16384
        },
        "pai-painter-scenery-base-zh": {
            "vocab_size": 37512,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 288,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 16384,  # 0 + 16384
            "eos_token_id": 10486,  # 102 + 16384
            "bos_token_id": 10485,  # 101 + 16384
            "eol_token_id": 10486,  # 102 + 16384
        },
        "pai-painter-commercial-base-zh": {
            "vocab_size": 37512,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 288,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 16384,  # 0 + 16384
            "eos_token_id": 10486,  # 102 + 16384
            "bos_token_id": 10485,  # 101 + 16384
            "eol_token_id": 10486,  # 102 + 16384
        },
        "pai-painter-large-zh": {
            "vocab_size": 37512,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 288,
            "type_vocab_size": 1,
            "initializer_range": 0.02,
            "pad_token_id": 16384,  # 0 + 16384
            "eos_token_id": 10486,  # 102 + 16384
            "bos_token_id": 10485,  # 101 + 16384
            "eol_token_id": 10486,  # 102 + 16384
        },
    }
    pretrained_resource_files_map = {
        "model_state": {
            "pai-painter-base-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-base-zh/model_state.pdparams",
            "pai-painter-painting-base-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-painting-base-zh/model_state.pdparams",
            "pai-painter-scenery-base-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-scenery-base-zh/model_state.pdparams",
            "pai-painter-commercial-base-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-commercial-base-zh/model_state.pdparams",
            "pai-painter-large-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-large-zh/model_state.pdparams",
        }
    }

    def __init__(self, gpt, image_vocab_size=16384):
        super().__init__(gpt)
        self.image_vocab_size = image_vocab_size
        self.lm_head = GPTLMHead(self.gpt.config["hidden_size"],
                                 image_vocab_size)
        self.apply(self.init_weights)

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id,
                                              eos_token_id):
        # we don't use attention_mask
        attention_mask = paddle.zeros_like(input_ids,
                                           dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    def prepare_faster_entry(self, kwargs):
        from paddlenlp.ops import FasterGPT
        use_fp16_decoding = kwargs.get('use_fp16_decoding', False)
        decode_strategy = kwargs.get('decode_strategy')
        if decode_strategy == "beam_search":
            raise AttributeError(
                "'beam_search' is not supported yet in the faster version of GPT"
            )
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.gpt.config["hidden_size"] // self.gpt.config[
            "num_attention_heads"]
        if size_per_head not in [32, 64, 80, 96, 128]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the faster version of GPT"
                % size_per_head)
        if kwargs['forced_bos_token_id'] is not None:
            # not support for min_length yet in the faster version
            raise AttributeError(
                "'forced_bos_token_id != None' is not supported yet in the faster version"
            )
        if kwargs['min_length'] != 0:
            # not support for min_length yet in the faster version
            raise AttributeError(
                "'min_length != 0' is not supported yet in the faster version")

        image_vocab_size, hidden_size = self.lm_head.decoder_weight.shape
        decoder_weight = paddle.concat([
            self.lm_head.decoder_weight,
            paddle.zeros(
                (self.gpt.config["vocab_size"] - image_vocab_size, hidden_size),
                dtype=paddle.get_default_dtype())
        ],
                                       axis=0)
        self.lm_head.decoder_weight = self.create_parameter(
            shape=[self.gpt.config["vocab_size"], hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Assign(decoder_weight))
        self._faster_entry = FasterGPT(
            self, use_fp16_decoding=use_fp16_decoding).forward
        return self._faster_entry


class ArtistForImageGeneration(ArtistLMHeadModel):
    r"""
    Artist Model with a `language modeling` head and `VQGanTokenizer` on top.
    Args:
        gpt (:class:`GPTModel`):
            An instance of GPTModel.
        image_vocab_size (int, optional):
            The vocabulary size of image.
            Defaults to `16384`. 
    """

    def __init__(self, gpt, image_vocab_size=16384):
        super().__init__(gpt, image_vocab_size)
        self.vqgan_detokenizer = VQGanDetokenizer(image_vocab_size, 256)

    @paddle.no_grad()
    def generate(self,
                 input_ids,
                 attention_mask=None,
                 top_k=0,
                 top_p=1.0,
                 temperature=1.0,
                 num_return_sequences=1,
                 use_faster=False,
                 use_fp16_decoding=False,
                 **kwargs):
        r"""
        The ArtistForImageGeneration generate method.
        Args:
            input_ids (Tensor):
                See :class:`ArtistLMHeadModel`.
            attention_mask (Tensor, optional):
                See :class:`ArtistLMHeadModel`.
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
            Its data type should be float32 and has a shape of [batch_size, num_return_sequences, 256, 256, 3].

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import ArtistForImageGeneration, ArtistTokenizer
                from PIL import Image

                # Initialize the model and tokenizer
                model_name_or_path = 'pai-painter-painting-base-zh'
                model = ArtistForImageGeneration.from_pretrained(model_name_or_path)
                tokenizer = ArtistTokenizer.from_pretrained(model_name_or_path)
                model.eval()

                # Prepare the model inputs.
                prompts = ["风阁水帘今在眼，且来先看早梅红", "见说春风偏有贺，露花千朵照庭闹"]
                tokenized_inputs = tokenizer(
                    prompts,
                    return_tensors="pd",
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                    max_length=32,
                )
                top_k = 32
                num_return_sequences = 4
                images = model.generate(**tokenized_inputs,
                                      top_k=top_k,
                                      num_return_sequences=num_return_sequences)
                print(images.shape)
                # [2, 4, 256, 256, 3]
                images = ((images.cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
                # [2, 256, 4*256, 3]
                images = images.transpose([0, 2, 1, 3,
                                        4]).reshape(-1, images.shape[-3],
                                                    num_return_sequences * images.shape[-2],
                                                    images.shape[-1])
                for i, image in enumerate(images):
                    image = Image.fromarray(image)
                    image.save(f"figure_{i}.png")
        """
        image_tokens = super().generate(
            input_ids=input_ids,
            max_length=256,
            decode_strategy='sampling',
            attention_mask=attention_mask,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            use_faster=use_faster,
            use_fp16_decoding=use_fp16_decoding,
            **kwargs)[0]
        images = self.vqgan_detokenizer(image_tokens)
        # images shape [bs, num_return_sequences, 256, 256, 3]
        images = images.reshape([
            -1, num_return_sequences, images.shape[1], images.shape[2],
            images.shape[3]
        ])
        return images


ArtistForCausalLM = ArtistLMHeadModel
