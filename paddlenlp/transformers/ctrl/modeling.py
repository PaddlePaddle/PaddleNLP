# coding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" paddle2.x CTRL model."""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import CrossEntropyLoss, MSELoss
from .. import PretrainedModel, register_base_model

from .utils import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    Config,
    SequenceClassifierOutput, )

__all__ = [
    'CTRLModel', "CTRLLMHeadModel", 'CTRLForSequenceClassification',
    'SinusoidalPositionalEmbedding'
]


class SinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        n_pos, dim = out.shape
        out.stop_gradient = True
        position_ids = paddle.arange(0, n_pos, dtype=out.dtype).unsqueeze(1)
        indices = paddle.arange(0, dim // 2, dtype=out.dtype).unsqueeze(0)

        indices = 10000.0**(-2 * indices / dim)
        embeddings = paddle.matmul(position_ids, indices)
        sentinel = dim // 2
        out[:, 0:sentinel] = paddle.sin(embeddings)
        out[:, sentinel:] = paddle.cos(embeddings)

        return out

    @paddle.no_grad()
    def forward(self, position_ids):
        return super().forward(position_ids)


def scaled_dot_product_attention(q, k, v, mask, attention_mask=None):
    # calculate attention
    matmul_qk = paddle.matmul(q, k, transpose_y=True)

    scaled_attention_logits = matmul_qk / np.sqrt(k.shape[-1])

    if mask is not None:
        nd, ns = scaled_attention_logits.shape[
            -2], scaled_attention_logits.shape[-1]
        scaled_attention_logits += mask[ns - nd:ns, :ns] * -1e4

    if attention_mask is not None:
        # Apply the attention mask
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = F.softmax(scaled_attention_logits, axis=-1)

    output = paddle.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Layer):
    def __init__(self, d_model_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model_size = d_model_size

        self.depth = d_model_size // self.num_heads

        self.Wq = nn.Linear(d_model_size, d_model_size)
        self.Wk = nn.Linear(d_model_size, d_model_size)
        self.Wv = nn.Linear(d_model_size, d_model_size)

        self.dense = nn.Linear(d_model_size, d_model_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape([batch_size, -1, self.num_heads, self.depth])
        return x.transpose(perm=[0, 2, 1, 3])

    def forward(
            self,
            v,
            k,
            q,
            mask,
            layer_past=None,
            attention_mask=None,
            use_cache=False,
            output_attentions=False, ):
        batch_size = q.shape[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            k = paddle.concat([past_key, k], axis=-2)
            v = paddle.concat([past_value, v], axis=-2)

        if use_cache is True:
            present = paddle.stack([k, v])
        else:
            present = (None, )

        scaled_attention, attn = scaled_dot_product_attention(q, k, v, mask,
                                                              attention_mask)
        scaled_attention = scaled_attention.transpose([0, 2, 1, 3])

        original_size_attention = scaled_attention.reshape(
            shape=[batch_size, -1, self.d_model_size])
        output = self.dense(original_size_attention)

        outputs = (output, present)
        if output_attentions:
            outputs = outputs + (attn, )
        return outputs


class EncoderLayer(nn.Layer):
    def __init__(self, d_model_size, num_heads, dff, rate=0.1, epsilon=1e-6):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model_size, dff),
            nn.ReLU(), nn.Linear(dff, d_model_size))
        self.layernorm1 = nn.LayerNorm(d_model_size, epsilon=epsilon)
        self.layernorm2 = nn.LayerNorm(d_model_size, epsilon=epsilon)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(
            self,
            x,
            mask,
            layer_past=None,
            attention_mask=None,
            use_cache=False,
            output_attentions=False, ):
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(
            normed,
            normed,
            normed,
            mask,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions, )
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output

        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output

        outputs = (out2, ) + attn_outputs[1:]
        return outputs


class CTRLPreTrainedModel(PretrainedModel):
    base_model_prefix = "transformer"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "ctrl": {
            "output_hidden_states": False,
            "output_attentions": False,
            "use_cache": True,
            "use_return_dict": True,
            "tie_word_embeddings": True,
            "attn_pdrop": 0.1,
            "dff": 8192,
            "embd_pdrop": 0.1,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-06,
            "n_embd": 1280,
            "n_head": 16,
            "n_layer": 48,
            "n_positions": 50000,
            "resid_pdrop": 0.1,
            "vocab_size": 246534,
            "pad_token_id": None
        },
        "sshleifer-tiny-ctrl": {
            "output_hidden_states": False,
            "output_attentions": False,
            "use_cache": True,
            "use_return_dict": True,
            "tie_word_embeddings": True,
            "attn_pdrop": 0.1,
            "dff": 2,
            "embd_pdrop": 0.1,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-06,
            "n_embd": 16,
            "n_head": 2,
            "n_layer": 2,
            "n_positions": 50000,
            "resid_pdrop": 0.1,
            "vocab_size": 246534,
            "pad_token_id": None
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ctrl":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ctrl/model_state.pdparams",
            "sshleifer-tiny-ctrl":
            "https://paddlenlp.bj.bcebos.com/models/transformers/sshleifer-tiny-ctrl/model_state.pdparams"
        }
    }

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.pd_config.initializer_range,
                    shape=layer.weight.shape, ))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.pd_config.initializer_range,
                    shape=layer.weight.shape, ))
            if layer._padding_idx is not None:
                emb_weight = layer.weight.numpy()
                emb_weight[layer._padding_idx] = np.zeros_like(emb_weight[
                    layer._padding_idx])
                layer.weight.set_value(paddle.to_tensor(emb_weight))
        elif isinstance(layer, nn.LayerNorm):
            layer.weight.set_value(paddle.ones_like(layer.weight))
            layer.bias.set_value(paddle.zeros_like(layer.bias))

    def greedy_search(
            self,
            input_ids,
            logits_processors,
            max_length,
            pad_token_id,
            eos_token_id,
            **model_kwargs, ):
        batch_size, cur_len = input_ids.shape
        origin_len = cur_len

        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full(
            [batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs)
            logits = outputs.logits
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)
            # greedy
            probs = F.softmax(logits)
            probs = paddle.log(probs)
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(
                    unfinished_flag,
                    next_tokens,
                    paddle.full_like(next_tokens, pad_token_id), )

            scores = self.update_scores_for_generation(
                scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(
                    unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break

            model_kwargs = self.update_model_kwargs_for_generation(outputs,
                                                                   model_kwargs)
        return input_ids[:, origin_len:], scores

    def beam_search(
            self,
            input_ids,
            beam_scorer,
            logits_processors,
            max_length,
            pad_token_id,
            eos_token_id,
            **model_kwargs, ):
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size)

        beam_scores = paddle.zeros(
            (batch_size, num_beams), dtype=paddle.get_default_dtype())
        beam_scores[:, 1:] = -1e9
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs)
            logits = outputs.logits
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # beam search
            # [batch_size * num_beams, vocab_size]
            next_scores = F.softmax(logits)
            next_scores = paddle.log(next_scores)

            next_scores = next_scores + beam_scores.unsqueeze(-1)
            # reshape for beam search
            vocab_size = next_scores.shape[-1]
            next_scores = next_scores.reshape(
                [batch_size, num_beams * vocab_size])

            next_scores, next_tokens = paddle.topk(
                next_scores, 2 * num_beams, axis=1)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                origin_len=origin_len,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id, )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            cur_len += 1
            input_ids = paddle.concat(
                [
                    paddle.index_select(input_ids, beam_idx),
                    beam_next_tokens.unsqueeze(-1),
                ],
                axis=-1, )

            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(outputs,
                                                                   model_kwargs)
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"],
                                                           beam_idx)

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id, )
        return pred_ids[:, origin_len:], scores

    def update_model_kwargs_for_generation(self, outputs, model_kwargs):
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        return model_kwargs


@register_base_model
class CTRLModel(CTRLPreTrainedModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.pd_config = pd_config = Config(**kwargs)
        self.d_model_size = pd_config.n_embd
        self.num_layers = pd_config.n_layer

        self.pos_encoding = SinusoidalPositionalEmbedding(pd_config.n_positions,
                                                          self.d_model_size)

        self.w = nn.Embedding(pd_config.vocab_size, pd_config.n_embd)

        self.dropout = nn.Dropout(pd_config.embd_pdrop)
        self.h = nn.LayerList([
            EncoderLayer(
                pd_config.n_embd,
                pd_config.n_head,
                pd_config.dff,
                pd_config.resid_pdrop,
                pd_config.layer_norm_epsilon, )
            for _ in range(pd_config.n_layer)
        ])
        self.layernorm = nn.LayerNorm(
            pd_config.n_embd, epsilon=pd_config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.w

    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):

        output_attentions = (output_attentions if output_attentions is not None
                             else self.pd_config.output_attentions)
        use_cache = use_cache if use_cache is not None else self.pd_config.use_cache
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.pd_config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else
                       self.pd_config.use_return_dict)

        seq_len = input_ids.shape[-1]
        input_ids = input_ids.reshape([-1, seq_len])
        batch_size = input_ids.shape[0]

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = paddle.arange(past_length, seq_len + past_length)
            position_ids = position_ids.unsqueeze(0).reshape(
                shape=[-1, seq_len])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.reshape(shape=[batch_size, -1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze([1, 2])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.astype(
                dtype=paddle.get_default_dtype())  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(shape=[-1, seq_len])
            token_type_embeds = self.w(token_type_ids) * np.sqrt(
                self.d_model_size)
        else:
            token_type_embeds = 0.0

        inputs_embeds = self.w(input_ids) * np.sqrt(self.d_model_size)
        pos_embeds = self.pos_encoding(position_ids)

        hidden_states = inputs_embeds + pos_embeds + token_type_embeds

        hidden_states = self.dropout(hidden_states)
        mask = paddle.triu(
            paddle.ones(shape=[seq_len + past_length, seq_len + past_length]),
            1)

        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, (h, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )
            outputs = h(
                hidden_states,
                mask,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions, )
            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present, )

            if output_attentions:
                all_attentions += (outputs[2], )

        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(
                v
                for v in
                [hidden_states, presents, all_hidden_states, all_attentions]
                if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions, )


class CTRLLMHeadModel(CTRLPreTrainedModel):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.pd_config = transformer.pd_config

        if self.pd_config.tie_word_embeddings:
            self.lm_head = self.transformer.w
            self.lm_head_bias = self.create_parameter(
                shape=[self.pd_config.vocab_size],
                dtype=self.lm_head.weight.dtype,
                is_bias=True, )
        else:
            self.lm_head = nn.Linear(self.pd_config.n_embd,
                                     self.pd_config.vocab_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past=None,
                                      use_cache=None,
                                      **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": use_cache
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        return_dict = (return_dict if return_dict is not None else
                       self.pd_config.use_return_dict)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        hidden_states = transformer_outputs[0]

        if self.pd_config.tie_word_embeddings:
            lm_logits = (paddle.matmul(
                hidden_states, self.lm_head.weight, transpose_y=True) +
                         self.lm_head_bias)
        else:
            lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[:, :-1]
            shift_labels = labels[:, 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape([-1, shift_logits.shape[-1]]),
                shift_labels.flatten(), )

        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions, )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        return tuple(
            tuple(
                past_state.index_select(beam_idx) for past_state in layer_past)
            for layer_past in past)


class CTRLForSequenceClassification(CTRLPreTrainedModel):
    def __init__(self, transformer, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.transformer = transformer
        self.pd_config = transformer.pd_config
        self.classifier = nn.Linear(
            self.pd_config.n_embd, self.num_labels, bias_attr=False)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):

        return_dict = (return_dict if return_dict is not None else
                       self.pd_config.use_return_dict)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)
        batch_size = input_ids.shape[0]

        assert (
            self.pd_config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."

        if self.pd_config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (
                paddle.not_equal(input_ids, self.pd_config.pad_token_id)
                .astype(paddle.int64).sum(-1).item() - 1)

        pooled_logits = logits[:, sequence_lengths]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(
                    pooled_logits.flatten(),
                    labels.astype(pooled_logits.dtype).flatten(), )
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.reshape([-1, self.num_labels]),
                    labels.flatten())

        if not return_dict:
            output = (pooled_logits, ) + transformer_outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions, )
