# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import json
from typing import Dict, List, Union, cast

import numpy as np
import paddle
import tqdm
from einops import rearrange

from paddlenlp.transformers import (
    AutoTokenizer,
    MistralModel,
    PretrainedConfig,
    PretrainedModel,
)


def _make_bidirection_mask(
    input_ids_shape: paddle.shape,
    dtype: paddle.dtype,
    past_key_values_length: int = 0,
):
    """
    Make bidirection mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = paddle.full(
        (tgt_len, tgt_len),
        fill_value=1,
    )
    mask = paddle.tril(tensor, diagonal=0)
    mask = paddle.ones_like(mask)  # here is for bidirection attention
    mask = paddle.log(mask).astype(dtype)

    if past_key_values_length > 0:
        mask = paddle.concat([paddle.zeros([tgt_len, past_key_values_length], dtype=dtype), mask], axis=-1)
    return mask[None, None, :, :].expand([bsz, 1, tgt_len, tgt_len + past_key_values_length])


def _expand_mask(mask: paddle.Tensor, dtype: paddle.dtype, tgt_len):
    expanded_mask = mask
    if len(mask.shape) == 2:
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand([bsz, 1, tgt_len, src_len]).astype(dtype)
    elif len(mask.shape) == 3:
        """
        Expands attention_mask from `[bsz, tgt_seq_len, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        expanded_mask = mask.unsqueeze(1).astype(dtype)

    inverted_mask = 1.0 - expanded_mask

    return paddle.where(inverted_mask > 0.5, paddle.full_like(inverted_mask, paddle.finfo(dtype).min), inverted_mask)


class LatentModel(PretrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)

        self.cross_attend_blocks_0_fn_to_kv = paddle.nn.Linear(in_features=config.hidden_size, out_features=2*config.max_position_embeddings, bias_attr=False)
        self.cross_attend_blocks_0_fn_to_out = paddle.nn.Linear(in_features=config.max_position_embeddings, out_features=config.hidden_size, bias_attr=False)
        self.cross_attend_blocks_0_fn_to_q = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.max_position_embeddings, bias_attr=False)
        self.cross_attend_blocks_0_norm = paddle.nn.LayerNorm(config.hidden_size)
        self.cross_attend_blocks_0_norm_context = paddle.nn.LayerNorm(config.hidden_size)

        self.cross_attend_blocks_1_fn_net_0 = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.max_position_embeddings)
        self.cross_attend_blocks_1_fn_net_2 = paddle.nn.Linear(in_features=config.max_position_embeddings//2, out_features=config.hidden_size)
        self.cross_attend_blocks_1_norm = paddle.nn.LayerNorm(config.hidden_size)

        self.latents = paddle.nn.Linear(in_features=config.hidden_size, out_features=512, bias_attr=False)

    def forward(self, last_hidden_states, pool_mask):
        latents = paddle.stack([self.latents.weight.T for _ in range(last_hidden_states.shape[0])]) 

        normed_x = self.cross_attend_blocks_0_norm(last_hidden_states) 
        normed_context = self.cross_attend_blocks_0_norm_context(latents) 

        q = self.cross_attend_blocks_0_fn_to_q(normed_x) 
        kv = self.cross_attend_blocks_0_fn_to_kv(normed_context) 
        k = kv[:, :, :self.config.max_position_embeddings]
        v = kv[:, :, self.config.max_position_embeddings:] 

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.config.num_key_value_heads), (q, k, v)) 
        
        out = paddle.nn.functional.scaled_dot_product_attention(q, k, v) 
        out = rearrange(out, 'b n h d -> b n (h d)', h=self.config.num_key_value_heads) 

        out_of_layer1 = self.cross_attend_blocks_0_fn_to_out(out) + last_hidden_states

        normed_x = self.cross_attend_blocks_1_norm(out_of_layer1)

        before_geglu = self.cross_attend_blocks_1_fn_net_0(normed_x)

        x_in_gegle = before_geglu[:, :, :self.config.max_position_embeddings//2] 
        gate_in_geglu = before_geglu[:, :, self.config.max_position_embeddings//2:] 
        x_after_geglu = x_in_gegle * paddle.nn.functional.gelu(gate_in_geglu)

        after_geglu = self.cross_attend_blocks_1_fn_net_2(x_after_geglu)

        out_of_layer2 = after_geglu + out_of_layer1 

        s = paddle.sum(out_of_layer2 * pool_mask.unsqueeze(-1), axis=1) 
        d = paddle.sum(pool_mask, axis=1, keepdim=True)
        hiddens = s / d 
        hiddens = paddle.nn.functional.normalize(hiddens, p=2, axis=-1)

        return hiddens


class NVEncodeModel(MistralModel):
    def __init__(
        self,
        config,
        tokenizer_path,
        eval_batch_size,
        query_instruction,
        document_instruction,
    ):
        super().__init__(config)  # get mistral model structure

        self.latent_model = LatentModel(config=config)  # get latent model structure

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.query_instruction = query_instruction
        self.document_instruction = document_instruction

        self.eval_batch_size = eval_batch_size

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = [self.query_instruction + q + self.tokenizer.eos_token for q in queries]
        instruction_len = len(self.tokenizer.encode(self.query_instruction, add_special_tokens=False)["input_ids"])
        return self.encode(input_texts, instruction_len)

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        if isinstance(corpus[0], dict):
            input_texts = ["{} {}".format(doc.get("title", ""), doc["text"]).strip() for doc in corpus]
        else:
            input_texts = corpus

        input_texts = [self.document_instruction + doc + self.tokenizer.eos_token for doc in input_texts]
        instruction_len = len(self.tokenizer.encode(self.document_instruction, add_special_tokens=False)["input_ids"])
        return self.encode(input_texts, instruction_len)

    @paddle.no_grad()
    def encode(self, sentences: List[str], instruction_len, **kwargs) -> np.ndarray:
        all_embeddings = []
        for start_index in tqdm.tqdm(list(range(0, len(sentences), self.eval_batch_size)), desc="Batches"):

            sentences_batch = sentences[start_index : start_index + self.eval_batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                max_length=4096,
                padding=True,
                return_token_type_ids=False,
                return_tensors="pd",
                truncation=True,
            )
            last_hidden_states = self.forward(**inputs)[0]  # get bs*len*4096
            pool_mask = inputs["attention_mask"]
            pool_mask[:, :instruction_len] = 0

            embeddings = self.latent_model.forward(last_hidden_states, pool_mask)
            embeddings = paddle.nn.functional.normalize(embeddings, p=2, axis=1)

            all_embeddings.append(embeddings.cpu().numpy().astype("float32"))

        return np.concatenate(all_embeddings, axis=0)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):

        combined_attention_mask = _make_bidirection_mask(
            input_shape,
            inputs_embeds.dtype,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
