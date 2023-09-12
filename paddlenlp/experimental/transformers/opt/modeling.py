# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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
from __future__ import annotations

import collections
from functools import partial
from typing import Any, Dict, List
from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedMultiTransformer,
)

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.nn import Layer
from paddle.nn.functional.flash_attention import flash_attention
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from paddlenlp.transformers.conversion_utils import StateDictNameMapping
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model
from paddlenlp.transformers.opt.configuration import OPTConfig
from paddlenlp.utils.log import logger
from paddlenlp.transformers import OPTPretrainedModel
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationInferenceModel,
)
from paddlenlp.transformers.opt.modeling import OPTLMHead,OPTEmbeddings
from paddlenlp_ops import get_padding_offset

__all__ = ["OPTForCausalLMInferenceModel"]

@register_base_model
class OPTInferenceModel(OPTPretrainedModel):

    def __init__(self, config: OPTConfig):
        super(OPTInferenceModel, self).__init__(config)
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embeddings = OPTEmbeddings(config)
        
        self.past_key_values_length = 0
        self.num_layers = config.num_hidden_layers

        if config.normalize_before:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads

        weight_file = "/root/.paddlenlp/models/facebook/opt-2.7b/model_state.pdparams"
        self.state_dict = paddle.load(weight_file, return_numpy=True)

        for k  in self.state_dict.keys():
            pass

        #paddle.set_default_dtype("float16")
        ln_scale_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.norm1.weight".format(i)) for i in range(config.num_hidden_layers)]
        ln_bias_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.norm1.bias".format(i)) for i in range(config.num_hidden_layers)]

        qkv_weight_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.qkv_weight".format(i)) for i in range(config.num_hidden_layers)]
        qkv_bias_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.qkv_bias".format(i)) for i in range(config.num_hidden_layers)]

        out_proj_weight_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.self_attn.out_proj.weight".format(i)) for i in range(config.num_hidden_layers)]
        out_proj_bias_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.self_attn.out_proj.bias".format(i)) for i in range(config.num_hidden_layers)]

        ffn_ln_scale_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.norm2.weight".format(i)) for i in range(config.num_hidden_layers)]
        ffn_ln_bias_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.norm2.bias".format(i)) for i in range(config.num_hidden_layers)]

        ffn1_weight_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.linear1.weight".format(i)) for i in range(config.num_hidden_layers)]
        ffn1_bias_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.linear1.bias".format(i)) for i in range(config.num_hidden_layers)]
        ffn2_weight_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.linear2.weight".format(i)) for i in range(config.num_hidden_layers)]
        ffn2_bias_attrs = [paddle.ParamAttr(name="opt.decoder.layers.{}.linear2.bias".format(i)) for i in range(config.num_hidden_layers)]

        self.transformer_block = FusedMultiTransformer(config.hidden_size,
                                                    config.num_attention_heads,
                                                    config.intermediate_size,
                                                    dropout_rate=0.0,
                                                    activation="relu",
                                                    normalize_before=True,
                                                    num_layers=config.num_hidden_layers,
                                                    nranks=1,
                                                    ring_id=-1,
                                                    ln_scale_attrs=ln_scale_attrs,
                                                    ln_bias_attrs = ln_bias_attrs,
                                                    qkv_weight_attrs=qkv_weight_attrs,
                                                    qkv_bias_attrs=qkv_bias_attrs,
                                                    linear_weight_attrs=out_proj_weight_attrs,
                                                    linear_bias_attrs=out_proj_bias_attrs,
                                                    ffn_ln_scale_attrs=ffn_ln_scale_attrs,
                                                    ffn_ln_bias_attrs=ffn_ln_bias_attrs,
                                                    ffn1_weight_attrs=ffn1_weight_attrs,
                                                    ffn1_bias_attrs=ffn1_bias_attrs,
                                                    ffn2_weight_attrs=ffn2_weight_attrs,
                                                    ffn2_bias_attrs=ffn2_bias_attrs, 
                                                    epsilon=1e-5)

        self.cache_kvs = []

        for i in range(self.num_layers):
            break
            ln_scale = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.norm1.weight".format(i)])
            ln_scale = paddle.cast(ln_scale, "float32")
            ln_bias = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.norm1.bias".format(i)])
            ln_bias = paddle.cast(ln_bias, "float32")

            q_weight = self.state_dict["opt.decoder.layers.{}.self_attn.q_proj.weight".format(i)]
            k_weight = self.state_dict["opt.decoder.layers.{}.self_attn.k_proj.weight".format(i)]
            v_weight = self.state_dict["opt.decoder.layers.{}.self_attn.v_proj.weight".format(i)]
            q_bias = self.state_dict["opt.decoder.layers.{}.self_attn.q_proj.bias".format(i)]
            k_bias = self.state_dict["opt.decoder.layers.{}.self_attn.k_proj.bias".format(i)]
            v_bias = self.state_dict["opt.decoder.layers.{}.self_attn.v_proj.bias".format(i)]

            concated_qkv_weight = np.concatenate([q_weight, k_weight, v_weight], axis=-1)
            concated_qkv_weight = concated_qkv_weight.transpose(1, 0)
            concated_qkv_weight = concated_qkv_weight.reshape(3 * self.num_heads * self.head_size, self.hidden_size)
            concated_qkv_weight = paddle.to_tensor(concated_qkv_weight)
            concated_qkv_weight = paddle.cast(concated_qkv_weight, "float32")

            concated_qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=-1)
            concated_qkv_bias = concated_qkv_bias.reshape(3 * self.num_heads * self.head_size)
            concated_qkv_bias = paddle.to_tensor(concated_qkv_bias)
            concated_qkv_bias = paddle.cast(concated_qkv_bias, "float32")

            out_proj_weight = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.self_attn.out_proj.weight".format(i)])
            out_proj_bias =  paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.self_attn.out_proj.bias".format(i)])
            out_proj_weight = paddle.cast(out_proj_weight, "float32")
            out_proj_bias = paddle.cast(out_proj_bias, "float32")

            ffn_ln_scale = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.norm2.weight".format(i)])
            ffn_ln_bias = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.norm2.bias".format(i)])
            ffn_ln_scale = paddle.cast(ffn_ln_scale, "float32")
            ffn_ln_bias = paddle.cast(ffn_ln_bias, "float32")

            ffn1_weight = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.linear1.weight".format(i)])
            ffn1_bias = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.linear1.bias".format(i)])
            ffn2_weight = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.linear2.weight".format(i)])
            ffn2_bias = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.linear2.bias".format(i)])
            ffn1_weight = paddle.cast(ffn1_weight, "float32")
            ffn1_bias = paddle.cast(ffn1_bias, "float32")
            ffn2_weight = paddle.cast(ffn2_weight, "float32")
            ffn2_bias = paddle.cast(ffn2_bias, "float32")

            # qkv_weight = paddle.concat(q_weight, k_weight, v_weight)
            list_weight = [
                ln_scale, ln_bias,
                concated_qkv_weight, concated_qkv_bias,
                out_proj_weight, out_proj_bias, 
                ffn_ln_scale, ffn_ln_bias,
                ffn1_weight, ffn1_bias,
                ffn2_weight, ffn2_bias,
            ]
            self.transformer_block.ln_scales[i].set_value(list_weight[0])
            self.transformer_block.ln_biases[i].set_value(list_weight[1])

            self.transformer_block.qkv_weights[i].set_value(list_weight[2])
            self.transformer_block.qkv_biases[i].set_value(list_weight[3])

            self.transformer_block.linear_weights[i].set_value(list_weight[4])
            self.transformer_block.linear_biases[i].set_value(list_weight[5])

            self.transformer_block.ffn_ln_scales[i].set_value(list_weight[6])
            self.transformer_block.ffn_ln_biases[i].set_value(list_weight[7])

            self.transformer_block.ffn1_weights[i].set_value(list_weight[8])
            self.transformer_block.ffn1_biases[i].set_value(list_weight[9])

            self.transformer_block.ffn2_weights[i].set_value(list_weight[10])
            self.transformer_block.ffn2_biases[i].set_value(list_weight[11])

        paddle.set_default_dtype("float32")


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets

    # This function is a little different from prepare_input_ids_for_generation in paddlenlp/transformers/generation/utils.py
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None, # [batch, 1, max_seq, max_seq]
        inputs_embeds=None,
        use_cache=None,
        cache_kvs=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        # kwargs["cache"] is used used to distinguish between encoder and decoder phase.
        past_key_values = kwargs.get("cache", None)
        is_decoder = past_key_values is not None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # genereate a fake input_ids according to inputs_embeds
        # this is usually occurred in img2txt multimodal model when first enter into this forward function.
        if input_ids is None and inputs_embeds is not None:
            input_ids = self.prepare_input_ids_for_generation(self.config.bos_token_id, inputs_embeds)
        if inputs_embeds is not None:
            batch, seq_len, hidden_dim = inputs_embeds.shape
            # merge batch and seq_len dimension.
            inputs_embeds = inputs_embeds.reshape([batch * seq_len, hidden_dim])

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        cache_kvs = cache_kvs if cache_kvs is not None else self.cache_kvs

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.shape

        if (input_shape[1] > 1):
             self.past_key_values_length = paddle.to_tensor([0])
        self.past_key_values_length = self.past_key_values_length.reshape([1])
        past_key_values_length = paddle.to_tensor([self.past_key_values_length])
        self.past_key_values_length += input_shape[1]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=paddle.ones([input_shape[0], past_key_values_length + input_shape[1]], dtype="int64"),
            input_embeddings=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if not is_decoder:
            import numpy as np
            npzfile = np.load('/zhoukangkang/output.npz')
            embedding_output = paddle.to_tensor(np.array(npzfile['output']))
            embedding_output = paddle.cast(embedding_output, dtype='float32')
            print("embedding_output", embedding_output)

        if not is_decoder:
            batch, seq_len, hidden_dim = embedding_output.shape
            # merge batch and seq_len dimension.
            embedding_output = embedding_output.reshape([batch * seq_len, hidden_dim])
        
        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len_encoder)
        else:
            ids_remove_padding = input_ids
            padding_offset = None
            cum_offsets = None

        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder
        with paddle.fluid.framework._stride_in_no_check_dy2st_diff():

            hidden_states, _ = self.transformer_block(
                    input_ids,
                    embedding_output,
                    cum_offsets=cum_offsets,
                    padding_offset=padding_offset,
                    attn_mask=paddle.cast(attention_mask, dtype=embedding_output.dtype),
                    caches=cache_kvs,
                    seq_lens=seq_lens,
                    rotary_embs=None,
                    rotary_emb_dims=0,
                    time_step=paddle.increment(paddle.shape(attention_mask)[-1], -1) if is_decoder else None,
                )

        for i in range(self.num_layers):
            break
            batch = 1
            seq_len = 37
            embed_dim = self.hidden_size
            embedding_output.reshape([batch,seq_len,embed_dim])

            ln_scale = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.norm1.weight".format(i)])
            ln_bias = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.norm1.bias".format(i)])
            ln_scale = paddle.cast(ln_scale, "float32")
            ln_bias = paddle.cast(ln_bias, "float32")

            q_weight = self.state_dict["opt.decoder.layers.{}.self_attn.q_proj.weight".format(i)]
            k_weight = self.state_dict["opt.decoder.layers.{}.self_attn.k_proj.weight".format(i)]
            v_weight = self.state_dict["opt.decoder.layers.{}.self_attn.v_proj.weight".format(i)]
            q_bias = self.state_dict["opt.decoder.layers.{}.self_attn.q_proj.bias".format(i)]
            k_bias = self.state_dict["opt.decoder.layers.{}.self_attn.k_proj.bias".format(i)]
            v_bias = self.state_dict["opt.decoder.layers.{}.self_attn.v_proj.bias".format(i)]
            concated_qkv_weight = np.concatenate([q_weight, k_weight, v_weight], axis=-1)
            concated_qkv_weight = concated_qkv_weight.transpose(1, 0)
            concated_qkv_weight = concated_qkv_weight.reshape(3 * self.num_heads * self.head_size, self.hidden_size)
            concated_qkv_weight = paddle.to_tensor(concated_qkv_weight)
            concated_qkv_weight = paddle.cast(concated_qkv_weight, "float32")
            concated_qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=-1)
            concated_qkv_bias = concated_qkv_bias.reshape(3 * self.num_heads * self.head_size)
            concated_qkv_bias = paddle.to_tensor(concated_qkv_bias)
            concated_qkv_bias = paddle.cast(concated_qkv_bias, "float32")

            out_proj_weight = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.self_attn.out_proj.weight".format(i)])
            out_proj_bias =  paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.self_attn.out_proj.bias".format(i)])
            out_proj_weight = paddle.cast(out_proj_weight, "float32")
            out_proj_bias = paddle.cast(out_proj_bias, "float32")

            ffn_ln_scale = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.norm2.weight".format(i)])
            ffn_ln_bias = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.norm2.bias".format(i)])
            ffn_ln_scale = paddle.cast(ffn_ln_scale, "float32")
            ffn_ln_bias = paddle.cast(ffn_ln_bias, "float32")

            ffn1_weight = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.linear1.weight".format(i)])
            ffn1_bias = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.linear1.bias".format(i)])
            ffn2_weight = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.linear2.weight".format(i)])
            ffn2_bias = paddle.to_tensor(self.state_dict["opt.decoder.layers.{}.linear2.bias".format(i)])
            ffn1_weight = paddle.cast(ffn1_weight, "float32")
            ffn1_bias = paddle.cast(ffn1_bias, "float32")
            ffn2_weight = paddle.cast(ffn2_weight, "float32")
            ffn2_bias = paddle.cast(ffn2_bias, "float32")

            # enc_input是输入哦！
            # layer_norm
            enc_out2 = F.layer_norm(embedding_output, self.hidden_size, weight=ln_scale, bias = ln_bias, epsilon=1e-5)
            enc_out2 = paddle.matmul(enc_out2, concated_qkv_weight.reshape([3 * self.num_heads * self.hidden_size // self.num_heads, self.hidden_size]).transpose([1, 0]))
            enc_out2 += concated_qkv_bias.reshape([3 * self.num_heads * self.hidden_size // self.num_heads])
            #print("enc_out2",i, enc_out2)
            enc_out2 = enc_out2.reshape([batch, seq_len, 3, self.num_heads, self.hidden_size // self.num_heads])
            enc_out2 = enc_out2.transpose([2, 0, 3, 1, 4])
            q,k,v = paddle.unbind(enc_out2, axis=0)


            #q:[batch, head, seq, head_dim]
            #k:[batch, head, seq, head_dim]
            out = paddle.matmul(q, k.transpose([0, 1, 3, 2]))
            out = out / (np.sqrt(self.hidden_size / self.num_heads))
            # 记得屏蔽啊
            for k in range(seq_len):
                for j in range(k+1, seq_len):
                    out[:,:,k,j] = -1000.0
            out = paddle.nn.functional.softmax(out, -1)
            out =  paddle.matmul(out, v)
            out = out.transpose([0, 2, 1, 3])
            out = out.reshape([batch, seq_len, embed_dim])

            out = paddle.matmul(out, out_proj_weight)

            out += out_proj_bias

            out += embedding_output
            ## ffn

            ffn_out = F.layer_norm(out, embed_dim, weight=ffn_ln_scale, bias = ffn_ln_bias, epsilon=1e-5)

            ffn_out = paddle.matmul(ffn_out, ffn1_weight)
            ffn_out += ffn1_bias
            ffn_out = F.relu(ffn_out)
            ffn_out = paddle.matmul(ffn_out, ffn2_weight)
            ffn_out += ffn2_bias
            out = out + ffn_out

            embedding_output = out

        output = hidden_states

        if self.final_layer_norm:
            output = self.final_layer_norm(output)
        return output

    @paddle.no_grad()
    def set_state_dict(self, state_dict):

        self.embeddings.position_embeddings.weight.set_value((paddle.to_tensor(state_dict["opt.embeddings.position_embeddings.weight"].astype("float32"))))
        self.embeddings.word_embeddings.weight.set_value(paddle.to_tensor(state_dict["opt.embeddings.word_embeddings.weight"].astype("float32")))
        self.final_layer_norm.weight.set_value(paddle.to_tensor(state_dict["opt.decoder.final_layer_norm.weight"].astype("float32")))
        self.final_layer_norm.bias.set_value(paddle.to_tensor(state_dict["opt.decoder.final_layer_norm.bias"].astype("float32")))

        for i in range(self.num_layers):
            ln_scale = paddle.to_tensor(state_dict["opt.decoder.layers.{}.norm1.weight".format(i)])
            ln_bias = paddle.to_tensor(state_dict["opt.decoder.layers.{}.norm1.bias".format(i)])
            ln_scale = paddle.cast(ln_scale, "float32")
            ln_bias = paddle.cast(ln_bias, "float32")

            q_weight = state_dict["opt.decoder.layers.{}.self_attn.q_proj.weight".format(i)]
            k_weight = state_dict["opt.decoder.layers.{}.self_attn.k_proj.weight".format(i)]
            v_weight = state_dict["opt.decoder.layers.{}.self_attn.v_proj.weight".format(i)]
            q_bias = state_dict["opt.decoder.layers.{}.self_attn.q_proj.bias".format(i)]
            k_bias = state_dict["opt.decoder.layers.{}.self_attn.k_proj.bias".format(i)]
            v_bias = state_dict["opt.decoder.layers.{}.self_attn.v_proj.bias".format(i)]

            concated_qkv_weight = np.concatenate([q_weight, k_weight, v_weight], axis=-1)
            concated_qkv_weight = concated_qkv_weight.transpose(1, 0)
            concated_qkv_weight = concated_qkv_weight.reshape(3 * self.num_heads * self.head_size, self.hidden_size)
            concated_qkv_weight = paddle.to_tensor(concated_qkv_weight)
            concated_qkv_weight = paddle.cast(concated_qkv_weight, "float32")

            concated_qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=-1)
            concated_qkv_bias = concated_qkv_bias.reshape(3 * self.num_heads * self.head_size)
            concated_qkv_bias = paddle.to_tensor(concated_qkv_bias)
            concated_qkv_bias = paddle.cast(concated_qkv_bias, "float32")

            out_proj_weight = paddle.to_tensor(state_dict["opt.decoder.layers.{}.self_attn.out_proj.weight".format(i)])
            out_proj_bias =  paddle.to_tensor(state_dict["opt.decoder.layers.{}.self_attn.out_proj.bias".format(i)])
            out_proj_weight = paddle.cast(out_proj_weight, "float32")
            out_proj_bias = paddle.cast(out_proj_bias, "float32")

            ffn_ln_scale = paddle.to_tensor(state_dict["opt.decoder.layers.{}.norm2.weight".format(i)])
            ffn_ln_bias = paddle.to_tensor(state_dict["opt.decoder.layers.{}.norm2.bias".format(i)])
            ffn_ln_scale = paddle.cast(ffn_ln_scale, "float32")
            ffn_ln_bias = paddle.cast(ffn_ln_bias, "float32")

            ffn1_weight = paddle.to_tensor(state_dict["opt.decoder.layers.{}.linear1.weight".format(i)])
            ffn1_bias = paddle.to_tensor(state_dict["opt.decoder.layers.{}.linear1.bias".format(i)])
            ffn2_weight = paddle.to_tensor(state_dict["opt.decoder.layers.{}.linear2.weight".format(i)])
            ffn2_bias = paddle.to_tensor(state_dict["opt.decoder.layers.{}.linear2.bias".format(i)])
            ffn1_weight = paddle.cast(ffn1_weight, "float32")
            ffn1_bias = paddle.cast(ffn1_bias, "float32")
            ffn2_weight = paddle.cast(ffn2_weight, "float32")
            ffn2_bias = paddle.cast(ffn2_bias, "float32")

            # qkv_weight = paddle.concat(q_weight, k_weight, v_weight)
            list_weight = [
                ln_scale, ln_bias,
                concated_qkv_weight, concated_qkv_bias,
                out_proj_weight, out_proj_bias, 
                ffn_ln_scale, ffn_ln_bias,
                ffn1_weight, ffn1_bias,
                ffn2_weight, ffn2_bias,
            ]
            self.transformer_block.ln_scales[i].set_value(list_weight[0])
            self.transformer_block.ln_biases[i].set_value(list_weight[1])

            self.transformer_block.qkv_weights[i].set_value(list_weight[2])
            self.transformer_block.qkv_biases[i].set_value(list_weight[3])

            self.transformer_block.linear_weights[i].set_value(list_weight[4])
            self.transformer_block.linear_biases[i].set_value(list_weight[5])

            self.transformer_block.ffn_ln_scales[i].set_value(list_weight[6])
            self.transformer_block.ffn_ln_biases[i].set_value(list_weight[7])

            self.transformer_block.ffn1_weights[i].set_value(list_weight[8])
            self.transformer_block.ffn1_biases[i].set_value(list_weight[9])

            self.transformer_block.ffn2_weights[i].set_value(list_weight[10])
            self.transformer_block.ffn2_biases[i].set_value(list_weight[11])


class OPTForCausalLMInferenceModel(GenerationInferenceModel, OPTPretrainedModel):

    def __init__(self, config: OPTConfig, **kwargs):
        super(OPTForCausalLMInferenceModel, self).__init__(config)
        self.opt = OPTInferenceModel(config)
        self.lm_head = OPTLMHead(
            hidden_size=self.opt.config.hidden_size,
            vocab_size=self.opt.config.vocab_size,
            # embedding_weights=self.opt.embeddings.word_embeddings.weight,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, from_hf_hub: bool = False, subfolder: str | None = None, *args, **kwargs
    ):
        # TODO: Support safetensors loading.
        kwargs["use_safetensors"] = False
        return super().from_pretrained(pretrained_model_name_or_path, from_hf_hub, subfolder, *args, **kwargs)

    @classmethod
    def get_cache_kvs_shape(
        cls, config: OPTConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for llama model

        Args:
            max_batch_size (int): the max batch size
            max_length (int | None, optional): the max_length of cache_kvs. Defaults to None.

        Returns:
            list[paddle.Tensor]: the list tensor shape for cache
        """
        if max_length is None:
            max_length = config.max_position_embeddings

        cache_kvs = []
        for _ in range(config.num_hidden_layers):
            cache_kvs.append(
                [
                    2,
                    max_batch_size,
                    config.num_attention_heads // max(config.tensor_parallel_degree, 1),
                    max_length,
                    config.hidden_size // config.num_attention_heads,
                ]
            )
        return cache_kvs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache_kvs,
        seq_len_encoder,
        seq_len_decoder,
        tgt_ids,
        tgt_pos,
        tgt_generation_mask,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        cache = kwargs.get("cache", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if cache is not None:
            input_ids = tgt_ids
            position_ids = tgt_pos
            attention_mask = (tgt_generation_mask - 1) * 1e4
            # make inputs_embeds be none in decoder phase.
            # in forward function, it will be assigned according to input_ids.
            inputs_embeds = None
        else:
            attention_mask = (attention_mask - 1) * 1e4
        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_kvs": cache_kvs,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
            "cache": cache,
        }
        return model_inputs
    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        cache=None,
        cache_kvs=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.opt(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            cache_kvs=cache_kvs,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs
        
        print("hidden_states", hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits[:, -labels.shape[1] :, :]
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            loss_fct = CrossEntropyLoss(reduction="mean", label_smoothing=None)
            labels = shift_labels.reshape((-1,))

            valid_index = paddle.where(labels != -100)[0].flatten()
            logits = shift_logits.reshape((-1, shift_logits.shape[-1]))
            logits = paddle.gather(logits, valid_index, axis=0)
            labels = paddle.gather(labels, valid_index, axis=0)
            lm_loss = loss_fct(logits, labels)

            loss = lm_loss

        if not return_dict:
            if not use_cache:
                return (loss, logits) if loss is not None else logits

            outputs = (logits,) + outputs[1:]
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.decoder_weight" in state_dict:
            print("牛逼")
            self.lm_head.decoder_weight.set_value(state_dict["lm_head.decoder_weight"])
        self.opt.set_state_dict({k: state_dict[k] for k in state_dict.keys()})
