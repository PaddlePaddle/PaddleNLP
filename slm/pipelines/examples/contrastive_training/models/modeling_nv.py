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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import tqdm
from einops import rearrange, repeat
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers import (
    AutoTokenizer,
    MistralModel,
    PretrainedConfig,
    PretrainedModel,
)
from paddlenlp.transformers.model_outputs import BaseModelOutputWithPast, ModelOutput
from paddlenlp.utils.log import logger


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[paddle.Tensor] = None
    p_reps: Optional[paddle.Tensor] = None
    loss: Optional[paddle.Tensor] = None
    scores: Optional[paddle.Tensor] = None


def scaled_dot_product_attention(q, k, v):  # [bs, len, num_heads, dim]
    matmul_qk = paddle.matmul(q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 3, 1]))
    dk = paddle.to_tensor(k.shape[-1], dtype=paddle.float32)
    scaled_attention_logits = matmul_qk / paddle.sqrt(dk)
    attention_weights = paddle.nn.functional.softmax(scaled_attention_logits, axis=-1)  # [bs, num_heads, q_len, k_len]
    output = paddle.matmul(attention_weights, v.transpose([0, 2, 1, 3]))  # [bs, num_heads, q_len, dim]
    output = output.transpose([0, 2, 1, 3])  # [bs, q_len, num_heads, dim]
    return output


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

        self.cross_attend_blocks_0_fn_to_kv = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=2 * config.max_position_embeddings, bias_attr=False
        )
        self.cross_attend_blocks_0_fn_to_out = paddle.nn.Linear(
            in_features=config.max_position_embeddings, out_features=config.hidden_size, bias_attr=False
        )
        self.cross_attend_blocks_0_fn_to_q = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.max_position_embeddings, bias_attr=False
        )
        self.cross_attend_blocks_0_norm = paddle.nn.LayerNorm(config.hidden_size)
        self.cross_attend_blocks_0_norm_context = paddle.nn.LayerNorm(config.hidden_size)

        self.cross_attend_blocks_1_fn_net_0 = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.max_position_embeddings
        )
        self.cross_attend_blocks_1_fn_net_2 = paddle.nn.Linear(
            in_features=config.max_position_embeddings // 2, out_features=config.hidden_size
        )
        self.cross_attend_blocks_1_norm = paddle.nn.LayerNorm(config.hidden_size)

        self.latents = paddle.nn.Linear(in_features=config.hidden_size, out_features=512, bias_attr=False)

    def forward(self, last_hidden_states, pool_mask):
        one = paddle.eye(
            num_rows=self.config.hidden_size,
            num_columns=self.config.hidden_size,
            dtype=str(self.latents.weight.dtype).split(".")[-1],
        )
        self_latents_weight_T = self.latents(one).T
        latents = repeat(self_latents_weight_T, "d h -> b d h", b=last_hidden_states.shape[0])

        normed_x = self.cross_attend_blocks_0_norm(last_hidden_states)
        normed_context = self.cross_attend_blocks_0_norm_context(latents)

        q = self.cross_attend_blocks_0_fn_to_q(normed_x)
        kv = self.cross_attend_blocks_0_fn_to_kv(normed_context)
        k = kv[:, :, : self.config.max_position_embeddings]
        v = kv[:, :, self.config.max_position_embeddings :]

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.config.num_key_value_heads), (q, k, v))

        # k.stop_gradient = False
        # v.stop_gradient = False
        # out = paddle.nn.functional.scaled_dot_product_attention(q, k, v) # if use this, must set k and v stop_gradient to False
        out = scaled_dot_product_attention(q, k, v)  # if use this, no need to manually set k and v
        out = rearrange(out, "b n h d -> b n (h d)", h=self.config.num_key_value_heads)

        out_of_layer1 = self.cross_attend_blocks_0_fn_to_out(out) + last_hidden_states

        normed_x = self.cross_attend_blocks_1_norm(out_of_layer1)

        before_geglu = self.cross_attend_blocks_1_fn_net_0(normed_x)

        x_in_gegle = before_geglu[:, :, : self.config.max_position_embeddings // 2]
        gate_in_geglu = before_geglu[:, :, self.config.max_position_embeddings // 2 :]
        x_after_geglu = x_in_gegle * paddle.nn.functional.gelu(gate_in_geglu)

        after_geglu = self.cross_attend_blocks_1_fn_net_2(x_after_geglu)

        out_of_layer2 = after_geglu + out_of_layer1

        s = paddle.sum(
            out_of_layer2 * pool_mask.unsqueeze(-1),
            axis=1,
            dtype=str(self.cross_attend_blocks_1_fn_net_2.weight.dtype).split(".")[-1],
        )
        d = paddle.sum(
            pool_mask, axis=1, keepdim=True, dtype=str(self.cross_attend_blocks_1_fn_net_2.weight.dtype).split(".")[-1]
        )
        hiddens = s / d
        hiddens = paddle.nn.functional.normalize(hiddens, p=2, axis=-1)

        return hiddens


class NVEncodeModel(MistralModel):
    def __init__(
        self,
        config,
        tokenizer_path,
        query_instruction,
        document_instruction,
        eval_batch_size=999,
        normalized=True,
        negatives_cross_device=False,
        temperature_=1,
        margin=0.01,
        use_inbatch_neg=True,
        matryoshka_dims=None,
        matryoshka_loss_weights=None,
    ):
        super().__init__(config)  # get mistral model structure

        self.latent_model = LatentModel(config=config)  # get latent model structure

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.query_instruction = query_instruction
        self.document_instruction = document_instruction

        self.eval_batch_size = eval_batch_size

        self.normalized = normalized
        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Distributed training has not been initialized for representation all gather.")
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.temperature = temperature_
        self.margin = margin
        self.use_inbatch_neg = use_inbatch_neg
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_loss_weights = matryoshka_loss_weights

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

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

    def get_model_config(
        self,
    ):
        return self.model_config.to_dict()

    def encode(self, features, instruction_len):
        last_hidden_states = self.m_forward(**features)[0]  # get bs*len*4096
        pool_mask = features["attention_mask"]
        pool_mask[:, :instruction_len] = 0
        embeddings = self.latent_model.forward(last_hidden_states, pool_mask)
        embeddings = paddle.nn.functional.normalize(embeddings, p=2, axis=1)
        return embeddings

    def compute_similarity(self, q_reps, p_reps):
        # q_reps [batch_size, embedding_dim]
        # p_reps [batch_size, embedding_dim]
        return paddle.matmul(q_reps, p_reps.transpose([1, 0]))

    def hard_negative_loss(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps)
        scores = scores / self.temperature
        scores = scores.reshape([q_reps.shape[0], -1])

        target = paddle.arange(scores.shape[0], dtype="int64")
        target = target * (p_reps.shape[0] // q_reps.shape[0])
        loss = self.compute_loss(scores, target)
        return scores, loss

    def in_batch_negative_loss(self, q_reps, p_reps):
        # In batch negatives
        scores = self.compute_similarity(q_reps, p_reps)
        # Substract margin from all positive samples cosine_sim()
        margin_diag = paddle.full(shape=[q_reps.shape[0]], fill_value=self.margin, dtype=q_reps.dtype)
        scores = scores - paddle.diag(margin_diag)
        # Scale cosine to ease training converge
        scores = scores / self.temperature
        target = paddle.arange(0, q_reps.shape[0], dtype="int64")
        loss = self.compute_loss(scores, target)
        return scores, loss

    def forward(
        self,
        query: Dict[str, paddle.Tensor] = None,
        passage: Dict[str, paddle.Tensor] = None,
        teacher_score: paddle.Tensor = None,
    ):
        instruction_len = len(self.tokenizer.encode(self.query_instruction, add_special_tokens=False)["input_ids"])
        q_reps = self.encode(query, instruction_len)
        instruction_len = len(self.tokenizer.encode(self.document_instruction, add_special_tokens=False)["input_ids"])
        p_reps = self.encode(passage, instruction_len)

        # For non-matryoshka loss, we normalize the representations
        if not self.matryoshka_dims:
            if self.normalized:
                q_reps = paddle.nn.functional.normalize(q_reps, axis=-1)
                p_reps = paddle.nn.functional.normalize(p_reps, axis=-1)

        if self.training:
            # Cross device negatives
            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            if self.matryoshka_dims:
                loss = 0.0
                scores = 0.0
                for loss_weight, dim in zip(self.matryoshka_loss_weights, self.matryoshka_dims):
                    reduced_q = q_reps[:, :dim]
                    reduced_d = p_reps[:, :dim]
                    if self.normalized:
                        reduced_q = paddle.nn.functional.normalize(reduced_q, axis=-1)
                        reduced_d = paddle.nn.functional.normalize(reduced_d, axis=-1)

                    if self.use_inbatch_neg:
                        dim_score, dim_loss = self.in_batch_negative_loss(reduced_q, reduced_d)
                    else:
                        dim_score, dim_loss = self.hard_negative_loss(reduced_q, reduced_d)
                    scores += dim_score
                    loss += loss_weight * dim_loss

            elif self.use_inbatch_neg:
                scores, loss = self.in_batch_negative_loss(q_reps, p_reps)
            else:
                scores, loss = self.hard_negative_loss(q_reps, p_reps)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[paddle.Tensor]):
        if t is None:
            return None

        all_tensors = [paddle.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = paddle.concat(all_tensors, axis=0)

        return all_tensors

    def save_pretrained(self, output_dir: str, **kwargs):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def m_forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=paddle.int64
            )
            position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length))
        else:
            position_ids = position_ids.reshape([-1, seq_length]).astype("int64")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if self.enable_recompute and has_gradient:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = recompute(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @paddle.no_grad()
    def encode_sentences(self, sentences: List[str], instruction_len, **kwargs) -> np.ndarray:
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
            last_hidden_states = self.m_forward(**inputs)[0]  # get bs*len*4096
            pool_mask = inputs["attention_mask"]
            pool_mask[:, :instruction_len] = 0

            embeddings = self.latent_model.forward(last_hidden_states, pool_mask)
            embeddings = paddle.nn.functional.normalize(embeddings, p=2, axis=1)

            all_embeddings.append(embeddings.cpu().numpy().astype("float32"))

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = [self.query_instruction + q + self.tokenizer.eos_token for q in queries]
        instruction_len = len(self.tokenizer.encode(self.query_instruction, add_special_tokens=False)["input_ids"])
        return self.encode_sentences(input_texts, instruction_len)

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        if isinstance(corpus[0], dict):
            input_texts = ["{} {}".format(doc.get("title", ""), doc["text"]).strip() for doc in corpus]
        else:
            input_texts = corpus

        input_texts = [self.document_instruction + doc + self.tokenizer.eos_token for doc in input_texts]
        instruction_len = len(self.tokenizer.encode(self.document_instruction, add_special_tokens=False)["input_ids"])
        return self.encode_sentences(input_texts, instruction_len)
