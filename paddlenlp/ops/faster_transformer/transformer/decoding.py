# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.framework import in_dygraph_mode

from paddle.fluid.layer_helper import LayerHelper
import paddle

from paddlenlp.ops.ext_utils import load
from paddlenlp.utils.log import logger


def infer_transformer_decoding(
        enc_output, memory_seq_lens, word_emb, slf_ln_weight, slf_ln_bias,
        slf_q_weight, slf_q_bias, slf_k_weight, slf_k_bias, slf_v_weight,
        slf_v_bias, slf_out_weight, slf_out_bias, cross_ln_weight,
        cross_ln_bias, cross_q_weight, cross_q_bias, cross_k_weight,
        cross_k_bias, cross_v_weight, cross_v_bias, cross_out_weight,
        cross_out_bias, ffn_ln_weight, ffn_ln_bias, ffn_inter_weight,
        ffn_inter_bias, ffn_out_weight, ffn_out_bias, decoder_ln_weight,
        decoder_ln_bias, linear_weight, linear_bias, pos_emb,
        _decoding_strategy, _beam_size, _topk, _topp, _n_head, _size_per_head,
        _n_layer, _bos_id, _eos_id, _max_out_len, _beam_search_diversity_rate):
    helper = LayerHelper('fusion_decoding', **locals())

    inputs = {
        'Input': enc_output,
        'MemSeqLen': memory_seq_lens,
        'WordEmbedding': word_emb,
        'SelfLayernormWeight@VECTOR': slf_ln_weight,
        'SelfLayernormBias@VECTOR': slf_ln_bias,
        'SelfQueryWeight@VECTOR': slf_q_weight,
        'SelfQueryBias@VECTOR': slf_q_bias,
        'SelfKeyWeight@VECTOR': slf_k_weight,
        'SelfKeyBias@VECTOR': slf_k_bias,
        'SelfValueWeight@VECTOR': slf_v_weight,
        'SelfValueBias@VECTOR': slf_v_bias,
        'SelfOutWeight@VECTOR': slf_out_weight,
        'SelfOutBias@VECTOR': slf_out_bias,
        'CrossLayernormWeight@VECTOR': cross_ln_weight,
        'CrossLayernormBias@VECTOR': cross_ln_bias,
        'CrossQueryWeight@VECTOR': cross_q_weight,
        'CrossQueryBias@VECTOR': cross_q_bias,
        'CrossKeyWeight@VECTOR': cross_k_weight,
        'CrossKeyBias@VECTOR': cross_k_bias,
        'CrossValueWeight@VECTOR': cross_v_weight,
        'CrossValueBias@VECTOR': cross_v_bias,
        'CrossOutWeight@VECTOR': cross_out_weight,
        'CrossOutBias@VECTOR': cross_out_bias,
        'FFNLayernormWeight@VECTOR': ffn_ln_weight,
        'FFNLayernormBias@VECTOR': ffn_ln_bias,
        'FFNInterWeight@VECTOR': ffn_inter_weight,
        'FFNInterBias@VECTOR': ffn_inter_bias,
        'FFNOutWeight@VECTOR': ffn_out_weight,
        'FFNOutBias@VECTOR': ffn_out_bias,
        'DecoderLayernormWeight': decoder_ln_weight,
        'DecoderLayernormBias': decoder_ln_bias,
        'EmbWeight': linear_weight,
        'EmbBias': linear_bias,
        'PositionEncEmb': pos_emb
    }

    attrs = {
        'decoding_strategy': _decoding_strategy,
        'beam_size': _beam_size,
        'topk': _topk,
        'topp': _topp,
        'n_head': _n_head,
        'size_per_head': _size_per_head,
        'num_layer': _n_layer,
        'bos_id': _bos_id,
        'eos_id': _eos_id,
        'max_len': _max_out_len,
        'beam_search_diversity_rate': _beam_search_diversity_rate
    }

    output_ids = helper.create_variable(dtype="int32")
    parent_ids = helper.create_variable(dtype="int32")
    sequence_length = helper.create_variable(dtype="int32")

    outputs = {
        'OutputIds': output_ids,
        'ParentIds': parent_ids,
        'SequenceLength': sequence_length
    }

    helper.append_op(
        type='fusion_decoding', inputs=inputs, outputs=outputs, attrs=attrs)

    return output_ids, parent_ids, sequence_length


def infer_gpt_decoding(
        input, word_emb, slf_ln_weight, slf_ln_bias, slf_q_weight, slf_q_bias,
        slf_k_weight, slf_k_bias, slf_v_weight, slf_v_bias, slf_out_weight,
        slf_out_bias, ffn_ln_weight, ffn_ln_bias, ffn_inter_weight,
        ffn_inter_bias, ffn_out_weight, ffn_out_bias, decoder_ln_weight,
        decoder_ln_bias, pos_emb, linear_weight, topk, topp, max_out_len,
        head_num, size_per_head, num_layer, bos_id, eos_id, temperature,
        use_fp16_decoding):
    helper = LayerHelper('fusion_gpt', **locals())

    inputs = {
        "Input": input,
        "WordEmbedding": word_emb,
        "SelfLayernormWeight@VECTOR": slf_ln_weight,
        "SelfLayernormBias@VECTOR": slf_ln_bias,
        "SelfQueryWeight@VECTOR": slf_q_weight,
        "SelfQueryBias@VECTOR": slf_q_bias,
        "SelfKeyWeight@VECTOR": slf_k_weight,
        "SelfKeyBias@VECTOR": slf_k_bias,
        "SelfValueWeight@VECTOR": slf_v_weight,
        "SelfValueBias@VECTOR": slf_v_bias,
        "SelfOutWeight@VECTOR": slf_out_weight,
        "SelfOutBias@VECTOR": slf_out_bias,
        "FFNLayernormWeight@VECTOR": ffn_ln_weight,
        "FFNLayernormBias@VECTOR": ffn_ln_bias,
        "FFNInterWeight@VECTOR": ffn_inter_weight,
        "FFNInterBias@VECTOR": ffn_inter_bias,
        "FFNOutWeight@VECTOR": ffn_out_weight,
        "FFNOutBias@VECTOR": ffn_out_bias,
        "DecoderLayernormWeight": decoder_ln_weight,
        "DecoderLayernormBias": decoder_ln_bias,
        "PositionEncEmb": pos_emb,
        "EmbWeight": linear_weight
    }

    attrs = {
        "topk": topk,
        "topp": topp,
        "max_len": max_out_len,
        "n_head": head_num,
        "size_per_head": size_per_head,
        "num_layer": num_layer,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "temperature": temperature,
        "use_fp16": use_fp16_decoding
    }

    output_ids = helper.create_variable(dtype="int32")
    outputs = {'OutputIds': output_ids}

    helper.append_op(
        type='fusion_gpt', inputs=inputs, outputs=outputs, attrs=attrs)

    return output_ids


def finalize(beam_size,
             output_ids,
             parent_ids,
             out_seq_lens,
             max_seq_len=None,
             decoding_strategy="beam_search"):
    if max_seq_len is None:
        max_seq_len = paddle.max(out_seq_lens)
    ids = paddle.slice(output_ids, [0], [0], [max_seq_len])
    if "beam_search" == decoding_strategy:
        parent_ids = paddle.slice(parent_ids, [0], [0],
                                  [max_seq_len]) % beam_size
        ids = paddle.nn.functional.gather_tree(ids, parent_ids)
    return ids


def transfer_param(p, is_bias=False, restore_data=False):
    param_shape = p.shape
    if restore_data:
        if in_dygraph_mode():
            param_data = p.numpy()
        else:
            param_data = np.array(paddle.static.global_scope().find_var(p.name)
                                  .get_tensor())
    del p
    return paddle.create_parameter(
        shape=param_shape,
        dtype="float16",
        is_bias=is_bias,
        default_initializer=paddle.nn.initializer.Assign(param_data)
        if restore_data else None)


class InferTransformerDecoding(nn.Layer):
    def __init__(self,
                 decoder,
                 word_embedding,
                 positional_embedding,
                 linear,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 bos_id=0,
                 eos_id=1,
                 decoding_strategy="beam_search",
                 beam_size=4,
                 topk=1,
                 topp=0.0,
                 max_out_len=256,
                 beam_search_diversity_rate=0.0,
                 decoding_lib=None,
                 use_fp16_decoding=False):
        # if decoding_lib is None:
        #     raise ValueError(
        #         "The args decoding_lib must be set to use Faster Transformer. ")
        # elif not os.path.exists(decoding_lib):
        #     raise ValueError("The path to decoding lib is not exist.")
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                decoding_lib)

        super(InferTransformerDecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in [
                    "self", "decoder", "word_embedding", "positional_embedding",
                    "linear"
            ]:
                setattr(self, "_" + arg, value)
        # process weights
        if use_fp16_decoding:
            for mod in decoder.layers:
                mod.norm1.weight = transfer_param(mod.norm1.weight)
                mod.norm1.bias = transfer_param(mod.norm1.bias, is_bias=True)
                mod.self_attn.q_proj.weight = transfer_param(
                    mod.self_attn.q_proj.weight)
                mod.self_attn.q_proj.bias = transfer_param(
                    mod.self_attn.q_proj.bias, is_bias=True)
                mod.self_attn.k_proj.weight = transfer_param(
                    mod.self_attn.k_proj.weight)
                mod.self_attn.k_proj.bias = transfer_param(
                    mod.self_attn.k_proj.bias, is_bias=True)
                mod.self_attn.v_proj.weight = transfer_param(
                    mod.self_attn.v_proj.weight)
                mod.self_attn.v_proj.bias = transfer_param(
                    mod.self_attn.v_proj.bias, is_bias=True)
                mod.self_attn.out_proj.weight = transfer_param(
                    mod.self_attn.out_proj.weight)
                mod.self_attn.out_proj.bias = transfer_param(
                    mod.self_attn.out_proj.bias, is_bias=True)

                mod.norm2.weight = transfer_param(mod.norm2.weight)
                mod.norm2.bias = transfer_param(mod.norm2.bias, is_bias=True)
                mod.cross_attn.q_proj.weight = transfer_param(
                    mod.cross_attn.q_proj.weight)
                mod.cross_attn.q_proj.bias = transfer_param(
                    mod.cross_attn.q_proj.bias, is_bias=True)
                mod.cross_attn.k_proj.weight = transfer_param(
                    mod.cross_attn.k_proj.weight)
                mod.cross_attn.k_proj.bias = transfer_param(
                    mod.cross_attn.k_proj.bias, is_bias=True)
                mod.cross_attn.v_proj.weight = transfer_param(
                    mod.cross_attn.v_proj.weight)
                mod.cross_attn.v_proj.bias = transfer_param(
                    mod.cross_attn.v_proj.bias, is_bias=True)
                mod.cross_attn.out_proj.weight = transfer_param(
                    mod.cross_attn.out_proj.weight)
                mod.cross_attn.out_proj.bias = transfer_param(
                    mod.cross_attn.out_proj.bias, is_bias=True)

                mod.norm3.weight = transfer_param(mod.norm3.weight)
                mod.norm3.bias = transfer_param(mod.norm3.bias, is_bias=True)
                mod.linear1.weight = transfer_param(mod.linear1.weight)
                mod.linear1.bias = transfer_param(
                    mod.linear1.bias, is_bias=True)
                mod.linear2.weight = transfer_param(mod.linear2.weight)
                mod.linear2.bias = transfer_param(
                    mod.linear2.bias, is_bias=True)

            decoder.norm.weight = transfer_param(decoder.norm.weight)
            decoder.norm.bias = transfer_param(decoder.norm.bias, is_bias=True)

            linear.weight = transfer_param(linear.weight)
            if "beam_search" != decoding_strategy:
                linear.bias = transfer_param(linear.bias)

            positional_embedding.weight = transfer_param(
                positional_embedding.weight)
            word_embedding.weight = transfer_param(word_embedding.weight)

        self.slf_ln_weight = []
        self.slf_ln_bias = []
        self.slf_q_weight = []
        self.slf_q_bias = []
        self.slf_k_weight = []
        self.slf_k_bias = []
        self.slf_v_weight = []
        self.slf_v_bias = []
        self.slf_out_weight = []
        self.slf_out_bias = []

        self.cross_ln_weight = []
        self.cross_ln_bias = []
        self.cross_q_weight = []
        self.cross_q_bias = []
        self.cross_k_weight = []
        self.cross_k_bias = []
        self.cross_v_weight = []
        self.cross_v_bias = []
        self.cross_out_weight = []
        self.cross_out_bias = []

        self.ffn_ln_weight = []
        self.ffn_ln_bias = []
        self.ffn_inter_weight = []
        self.ffn_inter_bias = []
        self.ffn_out_weight = []
        self.ffn_out_bias = []

        for mod in decoder.layers:
            self.slf_ln_weight.append(mod.norm1.weight)
            self.slf_ln_bias.append(mod.norm1.bias)
            self.slf_q_weight.append(mod.self_attn.q_proj.weight)
            self.slf_q_bias.append(mod.self_attn.q_proj.bias)
            self.slf_k_weight.append(mod.self_attn.k_proj.weight)
            self.slf_k_bias.append(mod.self_attn.k_proj.bias)
            self.slf_v_weight.append(mod.self_attn.v_proj.weight)
            self.slf_v_bias.append(mod.self_attn.v_proj.bias)
            self.slf_out_weight.append(mod.self_attn.out_proj.weight)
            self.slf_out_bias.append(mod.self_attn.out_proj.bias)

            self.cross_ln_weight.append(mod.norm2.weight)
            self.cross_ln_bias.append(mod.norm2.bias)
            self.cross_q_weight.append(mod.cross_attn.q_proj.weight)
            self.cross_q_bias.append(mod.cross_attn.q_proj.bias)
            self.cross_k_weight.append(mod.cross_attn.k_proj.weight)
            self.cross_k_bias.append(mod.cross_attn.k_proj.bias)
            self.cross_v_weight.append(mod.cross_attn.v_proj.weight)
            self.cross_v_bias.append(mod.cross_attn.v_proj.bias)
            self.cross_out_weight.append(mod.cross_attn.out_proj.weight)
            self.cross_out_bias.append(mod.cross_attn.out_proj.bias)

            self.ffn_ln_weight.append(mod.norm3.weight)
            self.ffn_ln_bias.append(mod.norm3.bias)
            self.ffn_inter_weight.append(mod.linear1.weight)
            self.ffn_inter_bias.append(mod.linear1.bias)
            self.ffn_out_weight.append(mod.linear2.weight)
            self.ffn_out_bias.append(mod.linear2.bias)

        self.decoder_ln_weight = [decoder.norm.weight]
        self.decoder_ln_bias = [decoder.norm.bias]

        self.pos_emb = [positional_embedding.weight]
        self.word_emb = [word_embedding.weight]

        self.linear_weight = [linear.weight]
        self.linear_bias = [linear.bias]

    def forward(self, enc_output, memory_seq_lens):
        if "beam_search" == self._decoding_strategy:
            enc_output = nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(
                enc_output, self._beam_size)
            memory_seq_lens = nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(
                memory_seq_lens, self._beam_size)

        output_ids, parent_ids, sequence_length = infer_transformer_decoding(
            [enc_output], [memory_seq_lens], self.word_emb, self.slf_ln_weight,
            self.slf_ln_bias, self.slf_q_weight, self.slf_q_bias,
            self.slf_k_weight, self.slf_k_bias, self.slf_v_weight,
            self.slf_v_bias, self.slf_out_weight, self.slf_out_bias,
            self.cross_ln_weight, self.cross_ln_bias, self.cross_q_weight,
            self.cross_q_bias, self.cross_k_weight, self.cross_k_bias,
            self.cross_v_weight, self.cross_v_bias, self.cross_out_weight,
            self.cross_out_bias, self.ffn_ln_weight, self.ffn_ln_bias,
            self.ffn_inter_weight, self.ffn_inter_bias, self.ffn_out_weight,
            self.ffn_out_bias, self.decoder_ln_weight, self.decoder_ln_bias,
            self.linear_weight, self.linear_bias, self.pos_emb,
            self._decoding_strategy, self._beam_size, self._topk, self._topp,
            self._n_head,
            int(self._d_model / self._n_head), self._num_decoder_layers,
            self._bos_id, self._eos_id, self._max_out_len,
            self._beam_search_diversity_rate)

        ids = finalize(
            self._beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            decoding_strategy=self._decoding_strategy)

        return ids


class InferGptDecoding(nn.Layer):
    def __init__(self,
                 model,
                 topk=4,
                 topp=0.0,
                 max_out_len=256,
                 bos_id=50256,
                 eos_id=50256,
                 temperature=1,
                 decoding_lib=None,
                 use_fp16_decoding=False):
        if os.path.exists(decoding_lib):
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                decoding_lib)
        else:
            logger.warning(
                "The specified decoding_lib does not exist, and it will be built automatically."
            )
            load("FasterTransformer", verbose=True)

        super(InferGptDecoding, self).__init__()
        self.topk = topk
        self.topp = topp
        self.max_out_len = max_out_len
        self.temperature = temperature
        self.use_fp16_decoding = use_fp16_decoding
        self.model = model
        self.head_num = self.model.gpt.config['num_attention_heads']
        self.size_per_head = int(self.model.gpt.config['hidden_size'] /
                                 self.head_num)
        self.num_layer = self.model.gpt.config['num_hidden_layers']
        self.bos_id = bos_id
        self.eos_id = eos_id

        data_type = "float32"
        if self.use_fp16_decoding:
            data_type = "float16"
            for mod in self.model.gpt.decoder.layers:
                mod.norm1.weight = transfer_param(
                    mod.norm1.weight, restore_data=True)
                mod.norm1.bias = transfer_param(
                    mod.norm1.bias, is_bias=True, restore_data=True)
                mod.self_attn.q_proj.weight = transfer_param(
                    mod.self_attn.q_proj.weight, restore_data=True)
                mod.self_attn.q_proj.bias = transfer_param(
                    mod.self_attn.q_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.k_proj.weight = transfer_param(
                    mod.self_attn.k_proj.weight, restore_data=True)
                mod.self_attn.k_proj.bias = transfer_param(
                    mod.self_attn.k_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.v_proj.weight = transfer_param(
                    mod.self_attn.v_proj.weight, restore_data=True)
                mod.self_attn.v_proj.bias = transfer_param(
                    mod.self_attn.v_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.out_proj.weight = transfer_param(
                    mod.self_attn.out_proj.weight, restore_data=True)
                mod.self_attn.out_proj.bias = transfer_param(
                    mod.self_attn.out_proj.bias,
                    is_bias=True,
                    restore_data=True)

                mod.norm2.weight = transfer_param(
                    mod.norm2.weight, restore_data=True)
                mod.norm2.bias = transfer_param(
                    mod.norm2.bias, is_bias=True, restore_data=True)
                mod.linear1.weight = transfer_param(
                    mod.linear1.weight, restore_data=True)
                mod.linear1.bias = transfer_param(
                    mod.linear1.bias, is_bias=True, restore_data=True)
                mod.linear2.weight = transfer_param(
                    mod.linear2.weight, restore_data=True)
                mod.linear2.bias = transfer_param(
                    mod.linear2.bias, is_bias=True, restore_data=True)

            self.model.gpt.embeddings.word_embeddings.weight = transfer_param(
                self.model.gpt.embeddings.word_embeddings.weight,
                restore_data=True)
            self.model.gpt.embeddings.position_embeddings.weight = transfer_param(
                self.model.gpt.embeddings.position_embeddings.weight,
                restore_data=True)
            self.model.gpt.decoder.norm.weight = transfer_param(
                self.model.gpt.decoder.norm.weight, restore_data=True)
            self.model.gpt.decoder.norm.bias = transfer_param(
                self.model.gpt.decoder.norm.bias, restore_data=True)

        self.linear_weight = [self.model.gpt.embeddings.word_embeddings.weight]

        self.slf_ln_weight = []
        self.slf_ln_bias = []
        self.slf_q_weight = []
        self.slf_q_bias = []
        self.slf_k_weight = []
        self.slf_k_bias = []
        self.slf_v_weight = []
        self.slf_v_bias = []
        self.slf_out_weight = []
        self.slf_out_bias = []

        self.ffn_ln_weight = []
        self.ffn_ln_bias = []
        self.ffn_inter_weight = []
        self.ffn_inter_bias = []
        self.ffn_out_weight = []
        self.ffn_out_bias = []

        for mod in self.model.gpt.decoder.layers:
            self.slf_ln_weight.append(mod.norm1.weight)
            self.slf_ln_bias.append(mod.norm1.bias)
            self.slf_q_weight.append(mod.self_attn.q_proj.weight)
            self.slf_q_bias.append(mod.self_attn.q_proj.bias)
            self.slf_k_weight.append(mod.self_attn.k_proj.weight)
            self.slf_k_bias.append(mod.self_attn.k_proj.bias)
            self.slf_v_weight.append(mod.self_attn.v_proj.weight)
            self.slf_v_bias.append(mod.self_attn.v_proj.bias)
            self.slf_out_weight.append(mod.self_attn.out_proj.weight)
            self.slf_out_bias.append(mod.self_attn.out_proj.bias)

            self.ffn_ln_weight.append(mod.norm2.weight)
            self.ffn_ln_bias.append(mod.norm2.bias)
            self.ffn_inter_weight.append(mod.linear1.weight)
            self.ffn_inter_bias.append(mod.linear1.bias)
            self.ffn_out_weight.append(mod.linear2.weight)
            self.ffn_out_bias.append(mod.linear2.bias)

        self.decoder_ln_weight = [self.model.gpt.decoder.norm.weight]
        self.decoder_ln_bias = [self.model.gpt.decoder.norm.bias]

        self.pos_emb = [self.model.gpt.embeddings.position_embeddings.weight]
        self.word_emb = [self.model.gpt.embeddings.word_embeddings.weight]

    def forward(self, input_ids):
        output_ids = infer_gpt_decoding(
            input=[input_ids],
            word_emb=self.word_emb,
            slf_ln_weight=self.slf_ln_weight,
            slf_ln_bias=self.slf_ln_bias,
            slf_q_weight=self.slf_q_weight,
            slf_q_bias=self.slf_q_bias,
            slf_k_weight=self.slf_k_weight,
            slf_k_bias=self.slf_k_bias,
            slf_v_weight=self.slf_v_weight,
            slf_v_bias=self.slf_v_bias,
            slf_out_weight=self.slf_out_weight,
            slf_out_bias=self.slf_out_bias,
            ffn_ln_weight=self.ffn_ln_weight,
            ffn_ln_bias=self.ffn_ln_bias,
            ffn_inter_weight=self.ffn_inter_weight,
            ffn_inter_bias=self.ffn_inter_bias,
            ffn_out_weight=self.ffn_out_weight,
            ffn_out_bias=self.ffn_out_bias,
            decoder_ln_weight=self.decoder_ln_weight,
            decoder_ln_bias=self.decoder_ln_bias,
            pos_emb=self.pos_emb,
            linear_weight=self.linear_weight,
            topk=self.topk,
            topp=self.topp,
            max_out_len=self.max_out_len,
            head_num=self.head_num,
            size_per_head=self.size_per_head,
            num_layer=self.num_layer,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            temperature=self.temperature,
            use_fp16_decoding=self.use_fp16_decoding)

        return output_ids
