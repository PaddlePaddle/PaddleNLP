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
import functools
import os
from collections import defaultdict
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.common_ops_import import LayerHelper
from paddle.framework import core

import paddlenlp
from paddlenlp.ops.ext_utils import LOADED_EXT, load
from paddlenlp.transformers import OPTForCausalLM
from paddlenlp.transformers.t5.modeling import T5DenseGatedGeluDense, T5DenseReluDense
from paddlenlp.transformers.utils import fn_args_to_dict
from paddlenlp.utils.log import logger


def run_custom(op_name, inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype=None):
    ret = []

    if paddle.in_dynamic_mode():
        new_inputs_var = []
        for k, v in zip(inputs_names, inputs_var):
            if not k.endswith("@VECTOR") and isinstance(v, (list, tuple)) and len(v) == 1:
                new_inputs_var.append(v[0])
            else:
                new_inputs_var.append(v)
        outs = core.eager._run_custom_op(op_name, *new_inputs_var, *attrs_val)
        return outs[0] if len(outs) == 1 else outs
    else:
        inputs = dict(zip(inputs_names, inputs_var))
        attrs = dict(zip(attrs_names, attrs_val))
        outputs = {}

        helper = LayerHelper(op_name, **locals())

        for i, name in enumerate(outputs_names):
            outputs[name] = helper.create_variable(dtype=outputs_dtype[i])
            ret.append(outputs[name])

        helper.append_op(type=op_name, inputs=inputs, outputs=outputs, attrs=attrs)

    return ret


def infer_transformer_decoding(
    enc_output,
    memory_seq_lens,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    cross_ln_weight,
    cross_ln_bias,
    cross_q_weight,
    cross_q_bias,
    cross_k_weight,
    cross_k_bias,
    cross_v_weight,
    cross_v_bias,
    cross_out_weight,
    cross_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    linear_weight,
    linear_bias,
    pos_emb,
    _decoding_strategy,
    _beam_size,
    _topk,
    _topp,
    _n_head,
    _size_per_head,
    _n_layer,
    _bos_id,
    _eos_id,
    _max_out_len,
    _diversity_rate,
    _rel_len,
    _alpha,
):

    inputs_names = [
        "Input",
        "MemSeqLen",
        "WordEmbedding",
        "SelfLayernormWeight@VECTOR",
        "SelfLayernormBias@VECTOR",
        "SelfQueryWeight@VECTOR",
        "SelfQueryBias@VECTOR",
        "SelfKeyWeight@VECTOR",
        "SelfKeyBias@VECTOR",
        "SelfValueWeight@VECTOR",
        "SelfValueBias@VECTOR",
        "SelfOutWeight@VECTOR",
        "SelfOutBias@VECTOR",
        "CrossLayernormWeight@VECTOR",
        "CrossLayernormBias@VECTOR",
        "CrossQueryWeight@VECTOR",
        "CrossQueryBias@VECTOR",
        "CrossKeyWeight@VECTOR",
        "CrossKeyBias@VECTOR",
        "CrossValueWeight@VECTOR",
        "CrossValueBias@VECTOR",
        "CrossOutWeight@VECTOR",
        "CrossOutBias@VECTOR",
        "FFNLayernormWeight@VECTOR",
        "FFNLayernormBias@VECTOR",
        "FFNInterWeight@VECTOR",
        "FFNInterBias@VECTOR",
        "FFNOutWeight@VECTOR",
        "FFNOutBias@VECTOR",
        "DecoderLayernormWeight",
        "DecoderLayernormBias",
        "EmbWeight",
        "EmbBias",
        "PositionEncEmb",
    ]

    inputs_var = [
        enc_output,
        memory_seq_lens,
        word_emb,
        slf_ln_weight,
        slf_ln_bias,
        slf_q_weight,
        slf_q_bias,
        slf_k_weight,
        slf_k_bias,
        slf_v_weight,
        slf_v_bias,
        slf_out_weight,
        slf_out_bias,
        cross_ln_weight,
        cross_ln_bias,
        cross_q_weight,
        cross_q_bias,
        cross_k_weight,
        cross_k_bias,
        cross_v_weight,
        cross_v_bias,
        cross_out_weight,
        cross_out_bias,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        decoder_ln_weight,
        decoder_ln_bias,
        linear_weight,
        linear_bias,
        pos_emb,
    ]

    attrs_names = [
        "decoding_strategy",
        "beam_size",
        "topk",
        "topp",
        "n_head",
        "size_per_head",
        "num_layer",
        "bos_id",
        "eos_id",
        "max_len",
        "beam_search_diversity_rate",
        "rel_len",
        "alpha",
    ]

    attrs_val = [
        _decoding_strategy,
        _beam_size,
        _topk,
        _topp,
        _n_head,
        _size_per_head,
        _n_layer,
        _bos_id,
        _eos_id,
        _max_out_len,
        _diversity_rate,
        _rel_len,
        _alpha,
    ]

    outputs_names = ["OutputIds", "ParentIds", "SequenceLength"]

    outputs_dtype = ["int32"] * len(outputs_names)

    return run_custom(
        "fusion_decoding", inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype
    )


def infer_force_decoding(
    enc_output,
    memory_seq_lens,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    cross_ln_weight,
    cross_ln_bias,
    cross_q_weight,
    cross_q_bias,
    cross_k_weight,
    cross_k_bias,
    cross_v_weight,
    cross_v_bias,
    cross_out_weight,
    cross_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    linear_weight,
    linear_bias,
    pos_emb,
    trg_word,
    _decoding_strategy,
    _beam_size,
    _topk,
    _topp,
    _n_head,
    _size_per_head,
    _n_layer,
    _bos_id,
    _eos_id,
    _max_out_len,
    _diversity_rate,
    _rel_len,
    _alpha,
):

    inputs_names = [
        "Input",
        "MemSeqLen",
        "WordEmbedding",
        "SelfLayernormWeight@VECTOR",
        "SelfLayernormBias@VECTOR",
        "SelfQueryWeight@VECTOR",
        "SelfQueryBias@VECTOR",
        "SelfKeyWeight@VECTOR",
        "SelfKeyBias@VECTOR",
        "SelfValueWeight@VECTOR",
        "SelfValueBias@VECTOR",
        "SelfOutWeight@VECTOR",
        "SelfOutBias@VECTOR",
        "CrossLayernormWeight@VECTOR",
        "CrossLayernormBias@VECTOR",
        "CrossQueryWeight@VECTOR",
        "CrossQueryBias@VECTOR",
        "CrossKeyWeight@VECTOR",
        "CrossKeyBias@VECTOR",
        "CrossValueWeight@VECTOR",
        "CrossValueBias@VECTOR",
        "CrossOutWeight@VECTOR",
        "CrossOutBias@VECTOR",
        "FFNLayernormWeight@VECTOR",
        "FFNLayernormBias@VECTOR",
        "FFNInterWeight@VECTOR",
        "FFNInterBias@VECTOR",
        "FFNOutWeight@VECTOR",
        "FFNOutBias@VECTOR",
        "DecoderLayernormWeight",
        "DecoderLayernormBias",
        "EmbWeight",
        "EmbBias",
        "PositionEncEmb",
        # The input of custom op must be given.
        # Dispensable() and Intermediate() are not supported.
        "TrgWord",
    ]

    inputs_var = [
        enc_output,
        memory_seq_lens,
        word_emb,
        slf_ln_weight,
        slf_ln_bias,
        slf_q_weight,
        slf_q_bias,
        slf_k_weight,
        slf_k_bias,
        slf_v_weight,
        slf_v_bias,
        slf_out_weight,
        slf_out_bias,
        cross_ln_weight,
        cross_ln_bias,
        cross_q_weight,
        cross_q_bias,
        cross_k_weight,
        cross_k_bias,
        cross_v_weight,
        cross_v_bias,
        cross_out_weight,
        cross_out_bias,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        decoder_ln_weight,
        decoder_ln_bias,
        linear_weight,
        linear_bias,
        pos_emb,
        # The input of custom op must be given.
        # Dispensable() and Intermediate() are not supported.
        trg_word,
    ]

    attrs_names = [
        "decoding_strategy",
        "beam_size",
        "topk",
        "topp",
        "n_head",
        "size_per_head",
        "num_layer",
        "bos_id",
        "eos_id",
        "max_len",
        "beam_search_diversity_rate",
        "rel_len",
        "alpha",
    ]

    attrs_val = [
        _decoding_strategy,
        _beam_size,
        _topk,
        _topp,
        _n_head,
        _size_per_head,
        _n_layer,
        _bos_id,
        _eos_id,
        _max_out_len,
        _diversity_rate,
        _rel_len,
        _alpha,
    ]

    outputs_names = ["OutputIds", "ParentIds", "SequenceLength"]

    outputs_dtype = ["int32"] * len(outputs_names)

    return run_custom(
        "fusion_force_decoding", inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype
    )


def infer_opt_decoding(
    input,
    attn_mask,
    mem_seq_len,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    pos_emb,
    linear_weight,
    normalize_before,
    topk,
    topp,
    max_out_len,
    head_num,
    size_per_head,
    num_layer,
    bos_id,
    eos_id,
    temperature,
    use_fp16_decoding,
):
    helper = LayerHelper("fusion_opt", **locals())

    inputs = {
        "Input": input,
        "AttentionMask": attn_mask,
        "StartLength": mem_seq_len,
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
        "EmbWeight": linear_weight,
    }
    tensor_para_size = get_ft_para_conf().tensor_para_size
    layer_para_size = get_ft_para_conf().layer_para_size
    layer_para_batch_size = get_ft_para_conf().layer_para_batch_size
    attrs = {
        "normalize_before": normalize_before,
        "topk": topk,
        "topp": topp,
        "max_len": max_out_len,
        "n_head": head_num,
        "size_per_head": size_per_head,
        "num_layer": num_layer,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "temperature": temperature,
        "use_fp16": use_fp16_decoding,
        "tensor_para_size": tensor_para_size,
        "layer_para_size": layer_para_size,
        "layer_para_batch_size": layer_para_batch_size,
    }

    output_ids = helper.create_variable(dtype="int32")
    outputs = {"OutputIds": output_ids}

    helper.append_op(type="fusion_opt", inputs=inputs, outputs=outputs, attrs=attrs)

    return output_ids


def infer_gpt_decoding(
    input,
    attn_mask,
    mem_seq_len,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    pos_emb,
    linear_weight,
    topk,
    topp,
    max_out_len,
    head_num,
    size_per_head,
    num_layer,
    bos_id,
    eos_id,
    temperature,
    use_fp16_decoding,
):

    tensor_para_size = get_ft_para_conf().tensor_para_size
    layer_para_size = get_ft_para_conf().layer_para_size
    layer_para_batch_size = get_ft_para_conf().layer_para_batch_size

    inputs_names = [
        "Input",
        "AttentionMask",
        "StartLength",
        "WordEmbedding",
        "SelfLayernormWeight@VECTOR",
        "SelfLayernormBias@VECTOR",
        "SelfQueryWeight@VECTOR",
        "SelfQueryBias@VECTOR",
        "SelfKeyWeight@VECTOR",
        "SelfKeyBias@VECTOR",
        "SelfValueWeight@VECTOR",
        "SelfValueBias@VECTOR",
        "SelfOutWeight@VECTOR",
        "SelfOutBias@VECTOR",
        "FFNLayernormWeight@VECTOR",
        "FFNLayernormBias@VECTOR",
        "FFNInterWeight@VECTOR",
        "FFNInterBias@VECTOR",
        "FFNOutWeight@VECTOR",
        "FFNOutBias@VECTOR",
        "DecoderLayernormWeight",
        "DecoderLayernormBias",
        "PositionEncEmb",
        "EmbWeight",
    ]

    inputs_var = [
        input,
        attn_mask,
        mem_seq_len,
        word_emb,
        slf_ln_weight,
        slf_ln_bias,
        slf_q_weight,
        slf_q_bias,
        slf_k_weight,
        slf_k_bias,
        slf_v_weight,
        slf_v_bias,
        slf_out_weight,
        slf_out_bias,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        decoder_ln_weight,
        decoder_ln_bias,
        pos_emb,
        linear_weight,
    ]

    attrs_names = [
        "topk",
        "topp",
        "max_len",
        "n_head",
        "size_per_head",
        "num_layer",
        "bos_id",
        "eos_id",
        "temperature",
        "use_fp16",
        "tensor_para_size",
        "layer_para_size",
        "layer_para_batch_size",
    ]

    attrs_val = [
        topk,
        topp,
        max_out_len,
        head_num,
        size_per_head,
        num_layer,
        bos_id,
        eos_id,
        temperature,
        use_fp16_decoding,
        tensor_para_size,
        layer_para_size,
        layer_para_batch_size,
    ]

    outputs_names = ["OutputIds"]

    outputs_dtype = ["int32"]

    return run_custom("fusion_gpt", inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype)


def infer_unified_decoding(
    input_ids,
    attn_mask,
    memory_seq_lens,
    type_id,
    decoder_type_id,
    logits_mask,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    trans_weight,
    trans_bias,
    lm_ln_weight,
    lm_ln_bias,
    linear_weight,
    linear_bias,
    pos_emb,
    type_emb,
    role_id,
    decoder_role_id,
    role_emb,
    position_id,
    decoder_position_id,
    _decoding_strategy,
    _beam_size,
    _topk,
    _topp,
    _n_head,
    _size_per_head,
    _n_layer,
    _bos_id,
    _eos_id,
    _max_out_len,
    _diversity_rate,
    _unk_id,
    _mask_id,
    _temperature,
    _len_penalty,
    _normalize_before,
    _pos_bias,
    _hidden_act,
    _rel_len,
    _early_stopping,
    _min_length,
):

    tensor_para_size = get_ft_para_conf().tensor_para_size
    layer_para_size = get_ft_para_conf().layer_para_size
    layer_para_batch_size = get_ft_para_conf().layer_para_batch_size

    inputs_names = [
        "InputIds",
        "AttnMask",
        "MemSeqLen",
        "TypeIds",
        "DecTypeIds",
        "LogitsMask",
        "WordEmbedding",
        "SelfLayernormWeight@VECTOR",
        "SelfLayernormBias@VECTOR",
        "SelfQueryWeight@VECTOR",
        "SelfQueryBias@VECTOR",
        "SelfKeyWeight@VECTOR",
        "SelfKeyBias@VECTOR",
        "SelfValueWeight@VECTOR",
        "SelfValueBias@VECTOR",
        "SelfOutWeight@VECTOR",
        "SelfOutBias@VECTOR",
        "FFNLayernormWeight@VECTOR",
        "FFNLayernormBias@VECTOR",
        "FFNInterWeight@VECTOR",
        "FFNInterBias@VECTOR",
        "FFNOutWeight@VECTOR",
        "FFNOutBias@VECTOR",
        "DecoderLayernormWeight",
        "DecoderLayernormBias",
        "TransWeight",
        "TransBias",
        "LMLayernormWeight",
        "LMLayernormBias",
        "EmbWeight",
        "EmbBias",
        "PositionEncEmb",
        "TypeEmb",
        "RoleIds",
        "DecRoleIds",
        "RoleEmbedding",
        "PositionIds",
        "DecPositionIds",
    ]

    inputs_var = [
        input_ids,
        attn_mask,
        memory_seq_lens,
        type_id,
        decoder_type_id,
        logits_mask,
        word_emb,
        slf_ln_weight,
        slf_ln_bias,
        slf_q_weight,
        slf_q_bias,
        slf_k_weight,
        slf_k_bias,
        slf_v_weight,
        slf_v_bias,
        slf_out_weight,
        slf_out_bias,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        decoder_ln_weight,
        decoder_ln_bias,
        trans_weight,
        trans_bias,
        lm_ln_weight,
        lm_ln_bias,
        linear_weight,
        linear_bias,
        pos_emb,
        type_emb,
        role_id,
        decoder_role_id,
        role_emb,
        position_id,
        decoder_position_id,
    ]

    attrs_names = [
        "decoding_strategy",
        "beam_size",
        "topk",
        "topp",
        "n_head",
        "size_per_head",
        "num_layer",
        "bos_id",
        "eos_id",
        "max_len",
        "beam_search_diversity_rate",
        "unk_id",
        "mask_id",
        "temperature",
        "len_penalty",
        "normalize_before",
        "pos_bias",
        "hidden_act",
        "rel_len",
        "early_stopping",
        "min_length",
        "tensor_para_size",
        "layer_para_size",
        "layer_para_batch_size",
    ]

    attrs_val = [
        _decoding_strategy,
        _beam_size,
        _topk,
        float(_topp),
        _n_head,
        _size_per_head,
        _n_layer,
        _bos_id,
        _eos_id,
        _max_out_len,
        _diversity_rate,
        _unk_id,
        _mask_id,
        _temperature,
        _len_penalty,
        _normalize_before,
        _pos_bias,
        _hidden_act,
        _rel_len,
        _early_stopping,
        _min_length,
        tensor_para_size,
        layer_para_size,
        layer_para_batch_size,
    ]

    outputs_names = ["OutputIds", "ParentIds", "SequenceLength", "OutputScores"]

    outputs_dtype = ["int32", "int32", "int32", "float32"]

    return run_custom(
        "fusion_unified_decoding", inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype
    )


def infer_miro_decoding(
    input_ids,
    attn_mask,
    memory_seq_lens,
    type_id,
    decoder_type_id,
    logits_mask,
    word_emb,
    pre_decoder_ln_weight,
    pre_decoder_ln_bias,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    trans_weight,
    trans_bias,
    lm_ln_weight,
    lm_ln_bias,
    linear_weight,
    linear_bias,
    pos_emb,
    type_emb,
    role_id,
    decoder_role_id,
    role_emb,
    position_id,
    decoder_position_id,
    _decoding_strategy,
    _beam_size,
    _topk,
    _topp,
    _n_head,
    _size_per_head,
    _n_layer,
    _bos_id,
    _eos_id,
    _max_out_len,
    _diversity_rate,
    _unk_id,
    _mask_id,
    _temperature,
    _len_penalty,
    _normalize_before,
    _pos_bias,
    _hidden_act,
    _rel_len,
    _early_stopping,
    _min_length,
):

    tensor_para_size = get_ft_para_conf().tensor_para_size
    layer_para_size = get_ft_para_conf().layer_para_size
    layer_para_batch_size = get_ft_para_conf().layer_para_batch_size

    inputs_names = [
        "InputIds",
        "AttnMask",
        "MemSeqLen",
        "TypeIds",
        "DecTypeIds",
        "LogitsMask",
        "WordEmbedding",
        "PreDecoderLayernormWeight",
        "PreDecoderLayernormBias",
        "SelfLayernormWeight@VECTOR",
        "SelfLayernormBias@VECTOR",
        "SelfQueryWeight@VECTOR",
        "SelfQueryBias@VECTOR",
        "SelfKeyWeight@VECTOR",
        "SelfKeyBias@VECTOR",
        "SelfValueWeight@VECTOR",
        "SelfValueBias@VECTOR",
        "SelfOutWeight@VECTOR",
        "SelfOutBias@VECTOR",
        "FFNLayernormWeight@VECTOR",
        "FFNLayernormBias@VECTOR",
        "FFNInterWeight@VECTOR",
        "FFNInterBias@VECTOR",
        "FFNOutWeight@VECTOR",
        "FFNOutBias@VECTOR",
        "DecoderLayernormWeight",
        "DecoderLayernormBias",
        "TransWeight",
        "TransBias",
        "LMLayernormWeight",
        "LMLayernormBias",
        "EmbWeight",
        "EmbBias",
        "PositionEncEmb",
        "TypeEmb",
        "RoleIds",
        "DecRoleIds",
        "RoleEmbedding",
        "PositionIds",
        "DecPositionIds",
    ]

    inputs_var = [
        input_ids,
        attn_mask,
        memory_seq_lens,
        type_id,
        decoder_type_id,
        logits_mask,
        word_emb,
        pre_decoder_ln_weight,
        pre_decoder_ln_bias,
        slf_ln_weight,
        slf_ln_bias,
        slf_q_weight,
        slf_q_bias,
        slf_k_weight,
        slf_k_bias,
        slf_v_weight,
        slf_v_bias,
        slf_out_weight,
        slf_out_bias,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        decoder_ln_weight,
        decoder_ln_bias,
        trans_weight,
        trans_bias,
        lm_ln_weight,
        lm_ln_bias,
        linear_weight,
        linear_bias,
        pos_emb,
        type_emb,
        role_id,
        decoder_role_id,
        role_emb,
        position_id,
        decoder_position_id,
    ]

    attrs_names = [
        "decoding_strategy",
        "beam_size",
        "topk",
        "topp",
        "n_head",
        "size_per_head",
        "num_layer",
        "bos_id",
        "eos_id",
        "max_len",
        "beam_search_diversity_rate",
        "unk_id",
        "mask_id",
        "temperature",
        "len_penalty",
        "normalize_before",
        "pos_bias",
        "hidden_act",
        "rel_len",
        "early_stopping",
        "min_length",
        "tensor_para_size",
        "layer_para_size",
        "layer_para_batch_size",
    ]

    attrs_val = [
        _decoding_strategy,
        _beam_size,
        _topk,
        float(_topp),
        _n_head,
        _size_per_head,
        _n_layer,
        _bos_id,
        _eos_id,
        _max_out_len,
        _diversity_rate,
        _unk_id,
        _mask_id,
        _temperature,
        _len_penalty,
        _normalize_before,
        _pos_bias,
        _hidden_act,
        _rel_len,
        _early_stopping,
        _min_length,
        tensor_para_size,
        layer_para_size,
        layer_para_batch_size,
    ]

    outputs_names = ["OutputIds", "ParentIds", "SequenceLength", "OutputScores"]

    outputs_dtype = ["int32", "int32", "int32", "float32"]

    return run_custom("fusion_miro", inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype)


def infer_bart_decoding(
    enc_output,
    memory_seq_lens,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    cross_ln_weight,
    cross_ln_bias,
    cross_q_weight,
    cross_q_bias,
    cross_k_weight,
    cross_k_bias,
    cross_v_weight,
    cross_v_bias,
    cross_out_weight,
    cross_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    linear_weight,
    linear_bias,
    pos_emb,
    _decoding_strategy,
    _beam_size,
    _topk,
    _topp,
    _temperature,
    _n_head,
    _size_per_head,
    _n_layer,
    _bos_id,
    _eos_id,
    _max_out_len,
    _min_out_len,
    _diversity_rate,
    _rel_len,
    _alpha,
    _early_stopping,
):

    inputs_names = [
        "Input",
        "MemSeqLen",
        "WordEmbedding",
        "SelfLayernormWeight@VECTOR",
        "SelfLayernormBias@VECTOR",
        "SelfQueryWeight@VECTOR",
        "SelfQueryBias@VECTOR",
        "SelfKeyWeight@VECTOR",
        "SelfKeyBias@VECTOR",
        "SelfValueWeight@VECTOR",
        "SelfValueBias@VECTOR",
        "SelfOutWeight@VECTOR",
        "SelfOutBias@VECTOR",
        "CrossLayernormWeight@VECTOR",
        "CrossLayernormBias@VECTOR",
        "CrossQueryWeight@VECTOR",
        "CrossQueryBias@VECTOR",
        "CrossKeyWeight@VECTOR",
        "CrossKeyBias@VECTOR",
        "CrossValueWeight@VECTOR",
        "CrossValueBias@VECTOR",
        "CrossOutWeight@VECTOR",
        "CrossOutBias@VECTOR",
        "FFNLayernormWeight@VECTOR",
        "FFNLayernormBias@VECTOR",
        "FFNInterWeight@VECTOR",
        "FFNInterBias@VECTOR",
        "FFNOutWeight@VECTOR",
        "FFNOutBias@VECTOR",
        "DecoderLayernormWeight",
        "DecoderLayernormBias",
        "EmbWeight",
        "EmbBias",
        "PositionEncEmb",
    ]

    inputs_var = [
        enc_output,
        memory_seq_lens,
        word_emb,
        slf_ln_weight,
        slf_ln_bias,
        slf_q_weight,
        slf_q_bias,
        slf_k_weight,
        slf_k_bias,
        slf_v_weight,
        slf_v_bias,
        slf_out_weight,
        slf_out_bias,
        cross_ln_weight,
        cross_ln_bias,
        cross_q_weight,
        cross_q_bias,
        cross_k_weight,
        cross_k_bias,
        cross_v_weight,
        cross_v_bias,
        cross_out_weight,
        cross_out_bias,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        decoder_ln_weight,
        decoder_ln_bias,
        linear_weight,
        linear_bias,
        pos_emb,
    ]

    attrs_names = [
        "decoding_strategy",
        "beam_size",
        "topk",
        "topp",
        "temperature",
        "n_head",
        "size_per_head",
        "num_layer",
        "bos_id",
        "eos_id",
        "max_len",
        "min_len",
        "beam_search_diversity_rate",
        "rel_len",
        "alpha",
        "early_stopping",
    ]

    attrs_val = [
        _decoding_strategy,
        _beam_size,
        _topk,
        _topp,
        _temperature,
        _n_head,
        _size_per_head,
        _n_layer,
        _bos_id,
        _eos_id,
        _max_out_len,
        _min_out_len,
        _diversity_rate,
        _rel_len,
        _alpha,
        _early_stopping,
    ]

    outputs_names = ["OutputIds", "ParentIds", "SequenceLength"]

    outputs_dtype = ["int32"] * len(outputs_names)

    return run_custom(
        "fusion_bart_decoding", inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype
    )


def infer_mbart_decoding(
    enc_output,
    memory_seq_lens,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    cross_ln_weight,
    cross_ln_bias,
    cross_q_weight,
    cross_q_bias,
    cross_k_weight,
    cross_k_bias,
    cross_v_weight,
    cross_v_bias,
    cross_out_weight,
    cross_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    mbart_ln_weight,
    mbart_ln_bias,
    linear_weight,
    linear_bias,
    pos_emb,
    trg_word,
    _decoding_strategy,
    _beam_size,
    _topk,
    _topp,
    _n_head,
    _size_per_head,
    _n_layer,
    _bos_id,
    _eos_id,
    _max_out_len,
    _diversity_rate,
    _rel_len,
    _alpha,
    _temperature,
    _early_stopping,
    _hidden_act,
):

    inputs_names = [
        "Input",
        "MemSeqLen",
        "WordEmbedding",
        "SelfLayernormWeight@VECTOR",
        "SelfLayernormBias@VECTOR",
        "SelfQueryWeight@VECTOR",
        "SelfQueryBias@VECTOR",
        "SelfKeyWeight@VECTOR",
        "SelfKeyBias@VECTOR",
        "SelfValueWeight@VECTOR",
        "SelfValueBias@VECTOR",
        "SelfOutWeight@VECTOR",
        "SelfOutBias@VECTOR",
        "CrossLayernormWeight@VECTOR",
        "CrossLayernormBias@VECTOR",
        "CrossQueryWeight@VECTOR",
        "CrossQueryBias@VECTOR",
        "CrossKeyWeight@VECTOR",
        "CrossKeyBias@VECTOR",
        "CrossValueWeight@VECTOR",
        "CrossValueBias@VECTOR",
        "CrossOutWeight@VECTOR",
        "CrossOutBias@VECTOR",
        "FFNLayernormWeight@VECTOR",
        "FFNLayernormBias@VECTOR",
        "FFNInterWeight@VECTOR",
        "FFNInterBias@VECTOR",
        "FFNOutWeight@VECTOR",
        "FFNOutBias@VECTOR",
        "DecoderLayernormWeight",
        "DecoderLayernormBias",
        "MBARTLayernormWeight",
        "MBARTLayernormBias",
        "EmbWeight",
        "EmbBias",
        "PositionEncEmb",
        # The input of custom op must be given.
        # Dispensable() and Intermediate() are not supported.
        "TrgWord",
    ]

    inputs_var = [
        enc_output,
        memory_seq_lens,
        word_emb,
        slf_ln_weight,
        slf_ln_bias,
        slf_q_weight,
        slf_q_bias,
        slf_k_weight,
        slf_k_bias,
        slf_v_weight,
        slf_v_bias,
        slf_out_weight,
        slf_out_bias,
        cross_ln_weight,
        cross_ln_bias,
        cross_q_weight,
        cross_q_bias,
        cross_k_weight,
        cross_k_bias,
        cross_v_weight,
        cross_v_bias,
        cross_out_weight,
        cross_out_bias,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        decoder_ln_weight,
        decoder_ln_bias,
        mbart_ln_weight,
        mbart_ln_bias,
        linear_weight,
        linear_bias,
        pos_emb,
        # The input of custom op must be given.
        # Dispensable() and Intermediate() are not supported.
        trg_word,
    ]

    attrs_names = [
        "decoding_strategy",
        "beam_size",
        "topk",
        "topp",
        "n_head",
        "size_per_head",
        "num_layer",
        "bos_id",
        "eos_id",
        "temperature",
        "max_len",
        "beam_search_diversity_rate",
        "rel_len",
        "alpha",
        "early_stopping",
        "hidden_act",
    ]

    attrs_val = [
        _decoding_strategy,
        _beam_size,
        _topk,
        _topp,
        _n_head,
        _size_per_head,
        _n_layer,
        _bos_id,
        _eos_id,
        _temperature,
        _max_out_len,
        _diversity_rate,
        _rel_len,
        _alpha,
        _early_stopping,
        _hidden_act,
    ]

    outputs_names = ["OutputIds", "ParentIds", "SequenceLength"]

    outputs_dtype = ["int32"] * len(outputs_names)

    return run_custom(
        "fusion_mbart_decoding", inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype
    )


def infer_gptj_decoding(
    input,
    attn_mask,
    mem_seq_len,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_out_weight,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    linear_weight,
    linear_bias,
    topk,
    topp,
    max_out_len,
    head_num,
    size_per_head,
    num_layer,
    bos_id,
    eos_id,
    temperature,
    rotary_embedding_dim,
    repetition_penalty,
    min_length,
    use_fp16_decoding,
):
    tensor_para_size = get_ft_para_conf().tensor_para_size
    layer_para_size = get_ft_para_conf().layer_para_size
    layer_para_batch_size = get_ft_para_conf().layer_para_batch_size

    inputs = {
        "Input": input,
        "AttentionMask": attn_mask,
        "StartLength": mem_seq_len,
        "WordEmbedding": word_emb,
        "SelfLayernormWeight@VECTOR": slf_ln_weight,
        "SelfLayernormBias@VECTOR": slf_ln_bias,
        "SelfQueryWeight@VECTOR": slf_q_weight,
        "SelfOutWeight@VECTOR": slf_out_weight,
        "FFNInterWeight@VECTOR": ffn_inter_weight,
        "FFNInterBias@VECTOR": ffn_inter_bias,
        "FFNOutWeight@VECTOR": ffn_out_weight,
        "FFNOutBias@VECTOR": ffn_out_bias,
        "DecoderLayernormWeight": decoder_ln_weight,
        "DecoderLayernormBias": decoder_ln_bias,
        "EmbWeight": linear_weight,
        "EmbBias": linear_bias,
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
        "rotary_embedding_dim": rotary_embedding_dim,
        "repetition_penalty": repetition_penalty,
        "min_length": min_length,
        "use_fp16": use_fp16_decoding,
        "tensor_para_size": tensor_para_size,
        "layer_para_size": layer_para_size,
        "layer_para_batch_size": layer_para_batch_size,
    }

    outputs_names = ["OutputIds"]
    outputs_dtype = ["int32"]

    return run_custom(
        op_name="fusion_gptj",
        inputs_names=inputs.keys(),
        inputs_var=inputs.values(),
        attrs_names=attrs.keys(),
        attrs_val=attrs.values(),
        outputs_names=outputs_names,
        outputs_dtype=outputs_dtype,
    )


def infer_pegasus_decoding(
    enc_output,
    memory_seq_lens,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    cross_ln_weight,
    cross_ln_bias,
    cross_q_weight,
    cross_q_bias,
    cross_k_weight,
    cross_k_bias,
    cross_v_weight,
    cross_v_bias,
    cross_out_weight,
    cross_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight,
    ffn_inter_bias,
    ffn_out_weight,
    ffn_out_bias,
    decoder_ln_weight,
    decoder_ln_bias,
    linear_weight,
    linear_bias,
    pos_emb,
    _decoding_strategy,
    _beam_size,
    _topk,
    _topp,
    _n_head,
    _size_per_head,
    _n_layer,
    _bos_id,
    _eos_id,
    _max_out_len,
    _min_out_len,
    _diversity_rate,
    _rel_len,
    _alpha,
    _temperature,
    _early_stopping,
    _hidden_act,
):

    inputs_names = [
        "Input",
        "MemSeqLen",
        "WordEmbedding",
        "SelfLayernormWeight@VECTOR",
        "SelfLayernormBias@VECTOR",
        "SelfQueryWeight@VECTOR",
        "SelfQueryBias@VECTOR",
        "SelfKeyWeight@VECTOR",
        "SelfKeyBias@VECTOR",
        "SelfValueWeight@VECTOR",
        "SelfValueBias@VECTOR",
        "SelfOutWeight@VECTOR",
        "SelfOutBias@VECTOR",
        "CrossLayernormWeight@VECTOR",
        "CrossLayernormBias@VECTOR",
        "CrossQueryWeight@VECTOR",
        "CrossQueryBias@VECTOR",
        "CrossKeyWeight@VECTOR",
        "CrossKeyBias@VECTOR",
        "CrossValueWeight@VECTOR",
        "CrossValueBias@VECTOR",
        "CrossOutWeight@VECTOR",
        "CrossOutBias@VECTOR",
        "FFNLayernormWeight@VECTOR",
        "FFNLayernormBias@VECTOR",
        "FFNInterWeight@VECTOR",
        "FFNInterBias@VECTOR",
        "FFNOutWeight@VECTOR",
        "FFNOutBias@VECTOR",
        "DecoderLayernormWeight",
        "DecoderLayernormBias",
        "EmbWeight",
        "EmbBias",
        "PositionEncEmb",
        # The input of custom op must be given.
        # Dispensable() and Intermediate() are not supported.
    ]

    inputs_var = [
        enc_output,
        memory_seq_lens,
        word_emb,
        slf_ln_weight,
        slf_ln_bias,
        slf_q_weight,
        slf_q_bias,
        slf_k_weight,
        slf_k_bias,
        slf_v_weight,
        slf_v_bias,
        slf_out_weight,
        slf_out_bias,
        cross_ln_weight,
        cross_ln_bias,
        cross_q_weight,
        cross_q_bias,
        cross_k_weight,
        cross_k_bias,
        cross_v_weight,
        cross_v_bias,
        cross_out_weight,
        cross_out_bias,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        decoder_ln_weight,
        decoder_ln_bias,
        linear_weight,
        linear_bias,
        pos_emb,
        # The input of custom op must be given.
        # Dispensable() and Intermediate() are not supported.
    ]

    attrs_names = [
        "decoding_strategy",
        "beam_size",
        "topk",
        "topp",
        "n_head",
        "size_per_head",
        "num_layer",
        "bos_id",
        "eos_id",
        "temperature",
        "max_len",
        "min_len",
        "beam_search_diversity_rate",
        "rel_len",
        "alpha",
        "early_stopping",
        "hidden_act",
        "emb_scale",
    ]

    attrs_val = [
        _decoding_strategy,
        _beam_size,
        _topk,
        _topp,
        _n_head,
        _size_per_head,
        _n_layer,
        _bos_id,
        _eos_id,
        _temperature,
        _max_out_len,
        _min_out_len,
        _diversity_rate,
        _rel_len,
        _alpha,
        _early_stopping,
        _hidden_act,
    ]

    outputs_names = ["OutputIds", "ParentIds", "SequenceLength"]

    outputs_dtype = ["int32"] * len(outputs_names)

    return run_custom(
        "fusion_pegasus_decoding", inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype
    )


def infer_t5_decoding(
    enc_output,
    memory_seq_lens,
    word_emb,
    slf_ln_weight,
    slf_ln_bias,
    slf_q_weight,
    slf_q_bias,
    slf_k_weight,
    slf_k_bias,
    slf_v_weight,
    slf_v_bias,
    slf_out_weight,
    slf_out_bias,
    cross_ln_weight,
    cross_ln_bias,
    cross_q_weight,
    cross_q_bias,
    cross_k_weight,
    cross_k_bias,
    cross_v_weight,
    cross_v_bias,
    cross_out_weight,
    cross_out_bias,
    ffn_ln_weight,
    ffn_ln_bias,
    ffn_inter_weight_0,
    ffn_inter_bias_0,
    ffn_inter_weight_1,
    ffn_inter_bias_1,
    ffn_out_weight,
    ffn_out_bias,
    relative_attention_bias_weight,
    decoder_ln_weight,
    decoder_ln_bias,
    linear_weight,
    linear_bias,
    decoding_strategy,
    beam_size,
    top_k,
    top_p,
    head_num,
    size_per_head,
    num_decoder_layers,
    start_id,
    end_id,
    max_out_len,
    diversity_rate,
    rel_len,
    alpha,
    temperature,
    early_stopping,
    max_distance,
    relative_attention_num_buckets,
    tie_word_embeddings,
    act,
):

    inputs_names = [
        "Input",
        "MemSeqLen",
        "WordEmbedding",
        "SelfLayernormWeight@VECTOR",
        "SelfLayernormBias@VECTOR",
        "SelfQueryWeight@VECTOR",
        "SelfQueryBias@VECTOR",
        "SelfKeyWeight@VECTOR",
        "SelfKeyBias@VECTOR",
        "SelfValueWeight@VECTOR",
        "SelfValueBias@VECTOR",
        "SelfOutWeight@VECTOR",
        "SelfOutBias@VECTOR",
        "CrossLayernormWeight@VECTOR",
        "CrossLayernormBias@VECTOR",
        "CrossQueryWeight@VECTOR",
        "CrossQueryBias@VECTOR",
        "CrossKeyWeight@VECTOR",
        "CrossKeyBias@VECTOR",
        "CrossValueWeight@VECTOR",
        "CrossValueBias@VECTOR",
        "CrossOutWeight@VECTOR",
        "CrossOutBias@VECTOR",
        "FFNLayernormWeight@VECTOR",
        "FFNLayernormBias@VECTOR",
        "FFNInterWeight0@VECTOR",
        "FFNInterBias0@VECTOR",
        "FFNInterWeight1@VECTOR",
        "FFNInterBias1@VECTOR",
        "FFNOutWeight@VECTOR",
        "FFNOutBias@VECTOR",
        "SelfRelativeAttentionBiasWeight",
        "DecoderLayernormWeight",
        "DecoderLayernormBias",
        "EmbWeight",
        "EmbBias",
    ]

    inputs_var = [
        enc_output,
        memory_seq_lens,
        word_emb,
        slf_ln_weight,
        slf_ln_bias,
        slf_q_weight,
        slf_q_bias,
        slf_k_weight,
        slf_k_bias,
        slf_v_weight,
        slf_v_bias,
        slf_out_weight,
        slf_out_bias,
        cross_ln_weight,
        cross_ln_bias,
        cross_q_weight,
        cross_q_bias,
        cross_k_weight,
        cross_k_bias,
        cross_v_weight,
        cross_v_bias,
        cross_out_weight,
        cross_out_bias,
        ffn_ln_weight,
        ffn_ln_bias,
        ffn_inter_weight_0,
        ffn_inter_bias_0,
        ffn_inter_weight_1,
        ffn_inter_bias_1,
        ffn_out_weight,
        ffn_out_bias,
        relative_attention_bias_weight,
        decoder_ln_weight,
        decoder_ln_bias,
        linear_weight,
        linear_bias,
    ]

    attrs_names = [
        "decoding_strategy",
        "beam_size",
        "topk",
        "topp",
        "n_head",
        "size_per_head",
        "num_layer",
        "bos_id",
        "eos_id",
        "max_len",
        "beam_search_diversity_rate",
        "rel_len",
        "alpha",
        "temperature",
        "early_stopping",
        "max_distance",
        "num_buckets",
        "tie_word_embeddings",
        "act",
    ]

    attrs_val = [
        decoding_strategy,
        beam_size,
        top_k,
        top_p,
        head_num,
        size_per_head,
        num_decoder_layers,
        start_id,
        end_id,
        max_out_len,
        diversity_rate,
        rel_len,
        alpha,
        temperature,
        early_stopping,
        max_distance,
        relative_attention_num_buckets,
        tie_word_embeddings,
        act,
    ]

    outputs_names = ["OutputIds", "ParentIds", "SequenceLength"]

    outputs_dtype = ["int32"] * len(outputs_names)

    return run_custom(
        "fusion_t5_decoding", inputs_names, inputs_var, attrs_names, attrs_val, outputs_names, outputs_dtype
    )


def finalize(
    beam_size,
    output_ids,
    parent_ids,
    out_seq_lens,
    forced_eos_token_id=None,
    max_seq_len=None,
    decoding_strategy="beam_search",
):
    if max_seq_len is None:
        max_seq_len = paddle.max(out_seq_lens)
    ids = paddle.slice(output_ids, [0], [0], [max_seq_len])
    if decoding_strategy.startswith("beam_search"):
        parent_ids = paddle.slice(parent_ids, [0], [0], [max_seq_len]) % (
            beam_size * 2 if decoding_strategy.endswith("_v2") or decoding_strategy.endswith("_v3") else beam_size
        )
        ids = paddle.nn.functional.gather_tree(ids, parent_ids)
        if forced_eos_token_id is not None:
            ids[-1, :, :] = forced_eos_token_id
    else:
        if forced_eos_token_id is not None:
            ids[-1, :] = forced_eos_token_id
    return ids


def transfer_param(p, is_bias=False, dtype="float16", restore_data=False):
    param_shape = p.shape
    # Allow CPU/GPU and float16/float32 transfer
    # NOTE: str(p.place) differs between paddle develop and 2.2
    if str(p.dtype)[-len(dtype) :] == dtype and ("gpu" in str(p.place).lower() or "cuda" in str(p.place).lower()):
        return p
    if restore_data:
        if paddle.in_dynamic_mode():
            param_data = p.numpy()
            # Creating parameters with Assign initializer is too slow. Maybe we
            # can cast to fp16 directly and get a tensor, while we do it more
            # elaborately to get a ParamBase. Also note `VarBase.set_value`
            # enforce the same dtype and can not be used directly.
            new_p = type(p)(shape=param_shape, dtype=dtype, is_bias=is_bias)
            new_p.value().get_tensor().set(param_data.astype(dtype), paddle.framework._current_expected_place())
            return new_p
        else:
            param_data = np.array(paddle.static.global_scope().find_var(p.name).get_tensor())
    return paddle.create_parameter(
        shape=param_shape,
        dtype=dtype,
        is_bias=is_bias,
        default_initializer=paddle.nn.initializer.Assign(param_data) if restore_data else None,
    )


def _convert_qkv(q_proj, k_proj, v_proj, attr="weight", use_numpy=True, del_param=False, dummy_tensor=None):
    ft_para_conf = get_ft_para_conf()
    # TODO(guosheng): maybe static graph need this
    # p = fast_model.create_parameter(
    #     shape=[q.shape[0], q.shape[1] + k.shape[1] + v.shape[1]],
    #     dtype=q.dtype,
    #     is_bias=is_bias)
    q = getattr(q_proj, attr)
    k = getattr(k_proj, attr)
    v = getattr(v_proj, attr)
    if use_numpy:
        q = q.numpy()
        if del_param:
            if attr == "weight":
                del q_proj.weight
            else:
                del q_proj.bias
        k = k.numpy()
        if del_param:
            if attr == "weight":
                del k_proj.weight
            else:
                del k_proj.bias
        v = v.numpy()
        if del_param:
            if attr == "weight":
                del v_proj.weight
            else:
                del v_proj.bias
    else:
        if del_param:
            for i in [q_proj, k_proj, v_proj]:
                if attr == "weight":
                    del i.weight
                else:
                    del i.bias
    q = ft_para_conf.slice_weight(q, 1)
    k = ft_para_conf.slice_weight(k, 1)
    v = ft_para_conf.slice_weight(v, 1)
    if del_param:
        # NOTE: dygraph_to_static/convert_call_func.py would log the converted
        # function. For linear layer, if we delete the params, log would fail.
        # And the log requires weight to be a 2D tensor.
        # NOTE: Assignment to parameter 'weight' should be of type
        # Parameter or None, thus delete before in case of tensor.
        setattr(q_proj, attr, dummy_tensor)
        setattr(k_proj, attr, dummy_tensor)
        setattr(v_proj, attr, dummy_tensor)
    if use_numpy:
        p = paddle.to_tensor(np.concatenate([q, k, v], axis=-1))
    else:
        p = paddle.concat([q, k, v], axis=-1)
    return p


def convert_params(fast_model, model, fuse_qkv=1, use_fp16=False, restore_data=False):
    r"""
    Convert parameters included in Transformer layer (`nn.TransformerEncoder`
    and `gpt.modeling.TransformerDecoder`) from original models to the format
    of faster models.

    Args:
        fast_model (Layer): The faster model object.
        model (Layer): The Transformer layer. It can be an instance of
            `nn.TransformerEncoder` or `gpt.modeling.TransformerDecoder`
            currently, and `nn.TransformerDecoder` would be supported soon.
        fuse_qkv (int): 0 for nofuse, 1 for fuse, 2 for fuse and delete the
            unfused parameters. If environment variable `PPFG_QKV_MEM_OPT` is
            set and the weights of q/k/v is fused, it will try to delete the
            original unfused weights. Note the rollback to original model would
            not be guarantee anymore when the faster model failed if the original
            weights are deleted. Default to 1.
        use_fp16 (bool): Whether to use float16. Maybe we should use the default
            dtype as the highest priority later. Default to `False`.
        restore_data (bool): If `False`, need to reload the weight values. It
            should be `True` for weight loaded models. Default to `False`.

    Returns:
        defaultdict: Each value is a list including converted parameters in all
            layers. For other parameters not included in Transformer module to
            be converted, such as embeddings, you can achieve it by using the
            returned dict `params` though `params['word_emb'].append()` directly
            which would do CPU/GPU and fp32/fp16 transfer automatically.
    """
    if fuse_qkv == 1:
        fuse_qkv = 2 if os.getenv("PPFG_QKV_MEM_OPT", "0") == "1" else 1
    ft_para_conf = get_ft_para_conf()

    class _list(list):
        def append(self, item):
            def attr_handle_func(x):
                return x

            if isinstance(item[0], nn.Layer):
                # Axis is used for tensor slice in tensor parallel.
                # Use None to make no slice on the tensor.
                if len(item) == 2:
                    layer, attr = item
                    axis = None
                else:
                    layer, attr, axis = item
                param = getattr(layer, attr)
                if axis is not None and isinstance(layer, nn.Linear):
                    param = ft_para_conf.slice_weight(param, axis)
                param = transfer_param(
                    param,
                    is_bias=attr.endswith("bias"),
                    dtype="float16" if use_fp16 else "float32",
                    restore_data=restore_data,
                )
                # NOTE: Assignment to parameter 'weight' should be of type
                # Parameter or None, thus delete first in case of param is
                # a tensor.
                # TODO(guosheng): Make slice_weight use `output_param=True`
                # and remove delattr. Currently, if `param` is Tensor rather
                # than Parameter, it would not be in state_dict.
                delattr(layer, attr)
                setattr(layer, attr, param)
            else:
                # NOTE: Compared with if branch, there is no layer attribute
                # refered to the transfered param, thus we should set it as
                # the layer attribute to be able to convert to static graph.
                # Additionally, we suppose no need to process tensor parallel
                # here since the param passed in might have been processed.
                if len(item) == 2:
                    param, is_bias = item
                    attr_handle = attr_handle_func
                else:
                    param, is_bias, attr_handle = item
                param = transfer_param(
                    param, is_bias=is_bias, dtype="float16" if use_fp16 else "float32", restore_data=restore_data
                )
                attr_handle(param)
            return super().append(param)

    params = defaultdict(_list)

    def _convert(module):
        if isinstance(
            module,
            (
                nn.TransformerEncoder,
                nn.TransformerDecoder,
                paddlenlp.transformers.gpt.modeling.TransformerDecoder,
                paddlenlp.transformers.opt.modeling.TransformerDecoder,
            ),
        ):
            num_layer = len(module.layers)
            for i, layer in enumerate(module.layers):
                if not ft_para_conf.is_load(i, num_layer):
                    continue
                # fuse_qkv: 0 for nofuse, 1 for fuse,
                # 2 for fuse and delete the unfused
                if fuse_qkv == 0:
                    params["slf_q_weight"].append((layer.self_attn.q_proj, "weight", 1))
                    params["slf_q_bias"].append((layer.self_attn.q_proj, "bias", 1))
                    params["slf_k_weight"].append((layer.self_attn.k_proj, "weight", 1))
                    params["slf_k_bias"].append((layer.self_attn.k_proj, "bias", 1))
                    params["slf_v_weight"].append((layer.self_attn.v_proj, "weight", 1))
                    params["slf_v_bias"].append((layer.self_attn.v_proj, "bias", 1))

                else:
                    # TODO(guosheng): Tensor with size 0 might be failed in
                    # paddle develop, thus use tensor with size 1 instead
                    # temporarily. Besides, we use 2D tensor since jit log
                    # requires that on linear weight. While size 0 seems all
                    # right in jit.to_static/jit.save.
                    dummy_tensor = paddle.zeros([1, 1])
                    w = _convert_qkv(
                        layer.self_attn.q_proj,
                        layer.self_attn.k_proj,
                        layer.self_attn.v_proj,
                        attr="weight",
                        use_numpy=fuse_qkv == 2,
                        del_param=fuse_qkv == 2,
                        dummy_tensor=dummy_tensor,
                    )
                    b = _convert_qkv(
                        layer.self_attn.q_proj,
                        layer.self_attn.k_proj,
                        layer.self_attn.v_proj,
                        attr="bias",
                        use_numpy=fuse_qkv == 2,
                        del_param=fuse_qkv == 2,
                        dummy_tensor=dummy_tensor,
                    )
                    params["slf_q_weight"].append((w, False))
                    params["slf_q_bias"].append((b, True))
                    # NOTE: Use `params["slf_q_weight"][-1]` rather than `w`,
                    # since the appended tensor might be a new transfered tensor.
                    # Besides, to allow convert_params be called more than once,
                    # we find a attr name not existing to avoid overwriting the
                    # existing attr.
                    attr = "slf_q_weight_" + str(i)
                    while hasattr(fast_model, attr):
                        attr += "_"
                    setattr(fast_model, attr, params["slf_q_weight"][-1])
                    attr = "slf_q_bias_" + str(i)
                    while hasattr(fast_model, attr):
                        attr += "_"
                    setattr(fast_model, attr, params["slf_q_bias"][-1])
                    for key in [f"slf_{m}_{n}" for m in ("k", "v") for n in ("weight", "bias")]:
                        params[key].append((dummy_tensor, True if key.endswith("bias") else False))
                        attr = key + "_" + str(i)
                        while hasattr(fast_model, attr):
                            attr += "_"
                        setattr(fast_model, attr, params[key][-1])
                if hasattr(layer, "cross_attn"):
                    # nn.TransformerDecoder
                    params["cross_q_weight"].append((layer.cross_attn.q_proj, "weight", 1))
                    params["cross_q_bias"].append((layer.cross_attn.q_proj, "bias", 1))
                    params["cross_k_weight"].append((layer.cross_attn.k_proj, "weight", 1))
                    params["cross_k_bias"].append((layer.cross_attn.k_proj, "bias", 1))
                    params["cross_v_weight"].append((layer.cross_attn.v_proj, "weight", 1))
                    params["cross_v_bias"].append((layer.cross_attn.v_proj, "bias", 1))
                    params["cross_out_weight"].append((layer.cross_attn.out_proj, "weight", 0))
                    params["cross_out_bias"].append((layer.cross_attn.out_proj, "bias", 0))

                params["slf_out_weight"].append((layer.self_attn.out_proj, "weight", 0))
                params["slf_out_bias"].append((layer.self_attn.out_proj, "bias"))
                params["slf_ln_weight"].append((layer.norm1, "weight"))
                params["slf_ln_bias"].append((layer.norm1, "bias"))
                # Slice tensor when append according to axis(1 or 0) if parallel
                # is enable.
                params["ffn_inter_weight"].append((layer.linear1, "weight", 1))
                params["ffn_inter_bias"].append((layer.linear1, "bias", 1))
                params["ffn_out_weight"].append((layer.linear2, "weight", 0))
                params["ffn_out_bias"].append((layer.linear2, "bias"))
                if hasattr(layer, "norm3"):
                    # nn.TransformerDecoder
                    params["cross_ln_weight"].append((layer.norm2, "weight"))
                    params["cross_ln_bias"].append((layer.norm2, "bias"))
                    params["ffn_ln_weight"].append((layer.norm3, "weight"))
                    params["ffn_ln_bias"].append((layer.norm3, "bias"))
                else:
                    params["ffn_ln_weight"].append((layer.norm2, "weight"))
                    params["ffn_ln_bias"].append((layer.norm2, "bias"))

            if getattr(module, "norm", None) is not None:
                params["decoder_ln_weight"].append((module.norm, "weight"))
                params["decoder_ln_bias"].append((module.norm, "bias"))
        elif isinstance(module, (paddlenlp.transformers.t5.modeling.T5Stack)) and module.is_decoder:
            num_layer = len(module.block)
            for i, block in enumerate(module.block):
                if not ft_para_conf.is_load(i, num_layer):
                    continue
                # fuse_qkv: 0 for nofuse, 1 for fuse,
                # 2 for fuse and delete the unfused
                if fuse_qkv == 0:
                    params["slf_q_weight"].append((block.layer[0].SelfAttention.q, "weight", 1))
                    if getattr(block.layer[0].SelfAttention.q, "bias", None) is not None:
                        params["slf_q_bias"].append((block.layer[0].SelfAttention.q, "bias", 1))

                    params["slf_k_weight"].append((block.layer[0].SelfAttention.k, "weight", 1))
                    if getattr(block.layer[0].SelfAttention.k, "bias", None) is not None:
                        params["slf_k_bias"].append((block.layer[0].SelfAttention.k, "bias", 1))

                    params["slf_v_weight"].append((block.layer[0].SelfAttention.v, "weight", 1))
                    if getattr(block.layer[0].SelfAttention.v, "bias", None) is not None:
                        params["slf_k_bias"].append((block.layer[0].SelfAttention.v, "bias", 1))

                else:
                    dummy_tensor = paddle.zeros([1, 1])
                    w = _convert_qkv(
                        block.layer[0].SelfAttention.q,
                        block.layer[0].SelfAttention.k,
                        block.layer[0].SelfAttention.v,
                        attr="weight",
                        use_numpy=(fuse_qkv == 2),
                        del_param=(fuse_qkv == 2),
                        dummy_tensor=dummy_tensor,
                    )
                    params["slf_q_weight"].append((w, False))

                    if (
                        getattr(block.layer[0].SelfAttention.q, "bias", None) is not None
                        and getattr(block.layer[0].SelfAttention.k, "bias", None) is not None
                        and getattr(block.layer[0].SelfAttention.v, "bias", None) is not None
                    ):
                        b = _convert_qkv(
                            block.layer[0].SelfAttention.q,
                            block.layer[0].SelfAttention.k,
                            block.layer[0].SelfAttention.v,
                            attr="bias",
                            use_numpy=(fuse_qkv == 2),
                            del_param=(fuse_qkv == 2),
                            dummy_tensor=dummy_tensor,
                        )
                        params["slf_q_bias"].append((b, True))

                    # NOTE: Use `params["slf_q_weight"][-1]` rather than `w`,
                    # since the appended tensor might be a new transfered tensor.
                    # Besides, to allow convert_params be called more than once,
                    # we find a attr name not existing to avoid overwriting the
                    # existing attr.
                    attr = "slf_q_weight_" + str(i)
                    while hasattr(fast_model, attr):
                        attr += "_"
                    setattr(fast_model, attr, params["slf_q_weight"][-1])

                    param_type = "weight"
                    if "slf_q_bias" in params.keys():
                        attr = "slf_q_bias_" + str(i)
                        while hasattr(fast_model, attr):
                            attr += "_"
                        setattr(fast_model, attr, params["slf_q_bias"][-1])
                        param_type.append("bias")

                    for key in [f"slf_{m}_{n}" for m in ("k", "v") for n in param_type]:
                        params[key].append((dummy_tensor, True if key.endswith("bias") else False))
                        attr = key + "_" + str(i)
                        while hasattr(fast_model, attr):
                            attr += "_"
                        setattr(fast_model, attr, params[key][-1])

                ffn_index = 1
                if len(block.layer) == 3:
                    ffn_index = 2

                    params["cross_q_weight"].append((block.layer[1].EncDecAttention.q, "weight", 1))
                    if getattr(block.layer[1].EncDecAttention.q, "bias", None) is not None:
                        params["cross_q_bias"].append((block.layer[1].EncDecAttention.q, "bias", 1))

                    params["cross_k_weight"].append((block.layer[1].EncDecAttention.k, "weight", 1))
                    if getattr(block.layer[1].EncDecAttention.k, "bias", None) is not None:
                        params["cross_k_bias"].append((block.layer[1].EncDecAttention.k, "bias", 1))

                    params["cross_v_weight"].append((block.layer[1].EncDecAttention.v, "weight", 1))
                    if getattr(block.layer[1].EncDecAttention.v, "bias", None) is not None:
                        params["cross_v_bias"].append((block.layer[1].EncDecAttention.v, "bias", 1))

                    params["cross_out_weight"].append((block.layer[1].EncDecAttention.o, "weight", 0))
                    if getattr(block.layer[1].EncDecAttention.o, "bias", None) is not None:
                        params["cross_out_bias"].append((block.layer[1].EncDecAttention.o, "bias", 0))

                    params["cross_ln_weight"].append((block.layer[1].layer_norm, "weight", 0))
                    if getattr(block.layer[1].layer_norm, "bias", None) is not None:
                        params["cross_ln_bias"].append((block.layer[1].layer_norm, "bias", 0))

                if hasattr(block.layer[ffn_index], "DenseReluDense"):
                    if isinstance(block.layer[ffn_index].DenseReluDense, (T5DenseReluDense)):
                        params["ffn_inter_weight_0"].append((block.layer[ffn_index].DenseReluDense.wi, "weight", 1))
                        if getattr(block.layer[ffn_index].DenseReluDense.wi, "bias", None) is not None:
                            params["ffn_inter_bias_0"].append((block.layer[ffn_index].DenseReluDense.wi, "bias", 1))

                        params["ffn_out_weight"].append((block.layer[ffn_index].DenseReluDense.wo, "weight", 0))
                        if getattr(block.layer[ffn_index].DenseReluDense.wo, "bias", None) is not None:
                            params["ffn_out_bias"].append((block.layer[ffn_index].DenseReluDense.wo, "bias"))
                    elif isinstance(block.layer[ffn_index].DenseReluDense, (T5DenseGatedGeluDense)):
                        params["ffn_inter_weight_0"].append((block.layer[ffn_index].DenseReluDense.wi_0, "weight", 1))
                        if getattr(block.layer[ffn_index].DenseReluDense.wi_0, "bias", None) is not None:
                            params["ffn_inter_bias_0"].append((block.layer[ffn_index].DenseReluDense.wi_0, "bias", 1))

                        params["ffn_inter_weight_1"].append((block.layer[ffn_index].DenseReluDense.wi_1, "weight", 1))
                        if getattr(block.layer[ffn_index].DenseReluDense.wi_1, "bias", None) is not None:
                            params["ffn_inter_bias_1"].append((block.layer[ffn_index].DenseReluDense.wi_1, "bias", 1))

                        params["ffn_out_weight"].append((block.layer[ffn_index].DenseReluDense.wo, "weight", 0))
                        if getattr(block.layer[ffn_index].DenseReluDense.wo, "bias", None) is not None:
                            params["ffn_out_bias"].append((block.layer[ffn_index].DenseReluDense.wo, "bias"))
                    else:
                        raise NotImplementedError("Faster only support T5DenseReluDense and T5DenseGatedGeluDense. ")

                params["ffn_ln_weight"].append((block.layer[ffn_index].layer_norm, "weight"))
                if getattr(block.layer[ffn_index].layer_norm, "bias", None) is not None:
                    params["ffn_ln_bias"].append((block.layer[ffn_index].layer_norm, "bias"))

                params["slf_out_weight"].append((block.layer[0].SelfAttention.o, "weight", 0))
                if getattr(block.layer[0].SelfAttention.o, "bias", None) is not None:
                    params["slf_out_bias"].append((block.layer[0].SelfAttention.o, "bias"))

                params["slf_ln_weight"].append((block.layer[0].layer_norm, "weight"))
                if getattr(block.layer[0].layer_norm, "bias", None) is not None:
                    params["slf_ln_bias"].append((block.layer[0].layer_norm, "bias"))

            if getattr(module, "norm", None) is not None:
                params["decoder_ln_weight"].append((module.final_layer_norm, "weight"))
                if getattr(module.final_layer_norm, "bias", None) is not None:
                    params["decoder_ln_bias"].append((module.final_layer_norm, "bias"))

    model.apply(_convert)
    return params


class InferBase(nn.Layer):
    def __init__(self, use_fp16_decoding):
        super(InferBase, self).__init__()
        self._use_fp16_decoding = use_fp16_decoding

    def default_bias(self, weight, index, is_null=False):
        if is_null:
            size = 1
        elif isinstance(weight, (list, tuple)):
            size = weight[0].shape[index]
        else:
            size = weight.shape[index]

        if not hasattr(self, "default_bias_" + str(size)):
            setattr(
                self,
                "default_bias_" + str(size),
                paddle.zeros(shape=[size], dtype="float16" if self._use_fp16_decoding else "float32"),
            )

        if isinstance(weight, (list, tuple)):
            return [getattr(self, "default_bias_" + str(size))] * len(weight)
        else:
            return [getattr(self, "default_bias_" + str(size))]


class InferTransformerDecoding(nn.Layer):
    def __init__(
        self,
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
        diversity_rate=0.0,
        decoding_lib=None,
        use_fp16_decoding=False,
        rel_len=False,
        alpha=0.6,
    ):
        # if decoding_lib is None:
        #     raise ValueError(
        #         "The args decoding_lib must be set to use FastGeneration. ")
        # elif not os.path.exists(decoding_lib):
        #     raise ValueError("The path to decoding lib is not exist.")
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FastGeneration" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FastGeneration"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load("FastGeneration", verbose=True)

        size_per_head = d_model / n_head
        # fuse_qkv can only support size_per_head of [32, 64, 128].
        if size_per_head in [32, 64, 128]:
            self._fuse_qkv = True
        else:
            self._fuse_qkv = False

        super(InferTransformerDecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in ["self", "decoder", "word_embedding", "positional_embedding", "linear"]:
                setattr(self, "_" + arg, value)
        # process weights
        if use_fp16_decoding:
            for mod in decoder.layers:
                mod.norm1.weight = transfer_param(mod.norm1.weight)
                mod.norm1.bias = transfer_param(mod.norm1.bias, is_bias=True)
                mod.self_attn.q_proj.weight = transfer_param(mod.self_attn.q_proj.weight)
                mod.self_attn.q_proj.bias = transfer_param(mod.self_attn.q_proj.bias, is_bias=True)
                mod.self_attn.k_proj.weight = transfer_param(mod.self_attn.k_proj.weight)
                mod.self_attn.k_proj.bias = transfer_param(mod.self_attn.k_proj.bias, is_bias=True)
                mod.self_attn.v_proj.weight = transfer_param(mod.self_attn.v_proj.weight)
                mod.self_attn.v_proj.bias = transfer_param(mod.self_attn.v_proj.bias, is_bias=True)
                mod.self_attn.out_proj.weight = transfer_param(mod.self_attn.out_proj.weight)
                mod.self_attn.out_proj.bias = transfer_param(mod.self_attn.out_proj.bias, is_bias=True)

                mod.norm2.weight = transfer_param(mod.norm2.weight)
                mod.norm2.bias = transfer_param(mod.norm2.bias, is_bias=True)
                mod.cross_attn.q_proj.weight = transfer_param(mod.cross_attn.q_proj.weight)
                mod.cross_attn.q_proj.bias = transfer_param(mod.cross_attn.q_proj.bias, is_bias=True)
                mod.cross_attn.k_proj.weight = transfer_param(mod.cross_attn.k_proj.weight)
                mod.cross_attn.k_proj.bias = transfer_param(mod.cross_attn.k_proj.bias, is_bias=True)
                mod.cross_attn.v_proj.weight = transfer_param(mod.cross_attn.v_proj.weight)
                mod.cross_attn.v_proj.bias = transfer_param(mod.cross_attn.v_proj.bias, is_bias=True)
                mod.cross_attn.out_proj.weight = transfer_param(mod.cross_attn.out_proj.weight)
                mod.cross_attn.out_proj.bias = transfer_param(mod.cross_attn.out_proj.bias, is_bias=True)

                mod.norm3.weight = transfer_param(mod.norm3.weight)
                mod.norm3.bias = transfer_param(mod.norm3.bias, is_bias=True)
                mod.linear1.weight = transfer_param(mod.linear1.weight)
                mod.linear1.bias = transfer_param(mod.linear1.bias, is_bias=True)
                mod.linear2.weight = transfer_param(mod.linear2.weight)
                mod.linear2.bias = transfer_param(mod.linear2.bias, is_bias=True)

            decoder.norm.weight = transfer_param(decoder.norm.weight)
            decoder.norm.bias = transfer_param(decoder.norm.bias, is_bias=True)

            linear.weight = transfer_param(linear.weight)
            linear.bias = transfer_param(linear.bias, is_bias=True)

            positional_embedding.weight = transfer_param(positional_embedding.weight)
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

        for i, mod in enumerate(decoder.layers):
            self.slf_ln_weight.append(mod.norm1.weight)
            self.slf_ln_bias.append(mod.norm1.bias)

            if self._fuse_qkv:
                q_weight_shape = mod.self_attn.q_proj.weight.shape
                k_weight_shape = mod.self_attn.k_proj.weight.shape
                v_weight_shape = mod.self_attn.v_proj.weight.shape

                q_weights = self.create_parameter(
                    shape=[q_weight_shape[0], q_weight_shape[1] + k_weight_shape[1] + v_weight_shape[1]],
                    dtype="float16" if use_fp16_decoding else "float32",
                )
                setattr(self, "slf_q_weight_" + str(i), q_weights)
                self.slf_q_weight.append(getattr(self, "slf_q_weight_" + str(i)))

                q_bias_shape = mod.self_attn.q_proj.bias.shape
                k_bias_shape = mod.self_attn.k_proj.bias.shape
                v_bias_shape = mod.self_attn.v_proj.bias.shape

                q_biases = self.create_parameter(
                    shape=[q_bias_shape[0] + k_bias_shape[0] + v_bias_shape[0]],
                    dtype="float16" if use_fp16_decoding else "float32",
                    is_bias=True,
                )
                setattr(self, "slf_q_bias_" + str(i), q_biases)
                self.slf_q_bias.append(getattr(self, "slf_q_bias_" + str(i)))
            else:
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

    def forward(self, enc_output, memory_seq_lens, trg_word=None):
        def parse_function(func_name):
            return partial(
                func_name,
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
                cross_ln_weight=self.cross_ln_weight,
                cross_ln_bias=self.cross_ln_bias,
                cross_q_weight=self.cross_q_weight,
                cross_q_bias=self.cross_q_bias,
                cross_k_weight=self.cross_k_weight,
                cross_k_bias=self.cross_k_bias,
                cross_v_weight=self.cross_v_weight,
                cross_v_bias=self.cross_v_bias,
                cross_out_weight=self.cross_out_weight,
                cross_out_bias=self.cross_out_bias,
                ffn_ln_weight=self.ffn_ln_weight,
                ffn_ln_bias=self.ffn_ln_bias,
                ffn_inter_weight=self.ffn_inter_weight,
                ffn_inter_bias=self.ffn_inter_bias,
                ffn_out_weight=self.ffn_out_weight,
                ffn_out_bias=self.ffn_out_bias,
                decoder_ln_weight=self.decoder_ln_weight,
                decoder_ln_bias=self.decoder_ln_bias,
                linear_weight=self.linear_weight,
                linear_bias=self.linear_bias,
                pos_emb=self.pos_emb,
                _decoding_strategy=self._decoding_strategy,
                _beam_size=self._beam_size,
                _topk=self._topk,
                _topp=self._topp,
                _n_head=self._n_head,
                _size_per_head=int(self._d_model / self._n_head),
                _n_layer=self._num_decoder_layers,
                _bos_id=self._bos_id,
                _eos_id=self._eos_id,
                _max_out_len=self._max_out_len,
                _diversity_rate=self._diversity_rate,
                _rel_len=self._rel_len,
                _alpha=self._alpha,
            )

        if self._decoding_strategy.startswith("beam_search"):
            # TODO: Due to paddle.tile bug in static graph, tile_beam_merge_with_batch
            # cannot work properly. These comments should be opened after PaddlePaddle v2.2.2.
            if paddle.__version__ <= "2.1.3":
                enc_output = nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(enc_output, self._beam_size)
                memory_seq_lens = nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(
                    memory_seq_lens, self._beam_size
                )
            else:
                enc_output_shape = paddle.shape(enc_output)
                batch_size = enc_output_shape[0]
                max_seq_len = enc_output_shape[1]
                enc_output = enc_output.unsqueeze([1])
                memory_seq_lens = memory_seq_lens.unsqueeze([1])
                enc_output = paddle.expand(
                    enc_output, shape=[batch_size, self._beam_size, max_seq_len, self._d_model]
                ).reshape([batch_size * self._beam_size, max_seq_len, self._d_model])
                memory_seq_lens = paddle.expand(memory_seq_lens, shape=[batch_size, self._beam_size]).reshape(
                    [batch_size * self._beam_size]
                )

        if trg_word is None:
            output_ids, parent_ids, sequence_length = parse_function(infer_transformer_decoding)(
                enc_output=[enc_output], memory_seq_lens=[memory_seq_lens]
            )
        else:
            output_ids, parent_ids, sequence_length = parse_function(infer_force_decoding)(
                enc_output=[enc_output], memory_seq_lens=[memory_seq_lens], trg_word=[trg_word]
            )

        ids = finalize(
            self._beam_size, output_ids, parent_ids, sequence_length, decoding_strategy=self._decoding_strategy
        )

        return ids


# Patch for parallel inference to save memory
class FTParaConf(object):
    r"""
    Configurations for model parallel in FastGeneration. Currently only
    support GPT. Please refer to  `Megatron <https://arxiv.org/pdf/2104.04473.pdf>`__
    for details.

    Args:
        tensor_para_size (int, optional): The size for tensor parallel. If it is
            1, tensor parallel would not be used. Default to 1.
        layer_para_size (int, optional): The size for layer parallel. If it is
            1, layer parallel would not be used. Default to 1.
        layer_para_batch_size (int, optional): The local batch size for pipeline
            parallel. It is suggested to use `batch_size // layer_para_size`.
            Default to 1.
    """

    def __init__(self, tensor_para_size=None, layer_para_size=None, layer_para_batch_size=1):
        self.world_size = self._env2int(
            [  # MPICH, OpenMPI, IMPI
                "MPI_LOCALNRANKS",
                "OMPI_COMM_WORLD_SIZE",
                "PMI_SIZE",
                "MV2_COMM_WORLD_SIZE",
                "WORLD_SIZE",
            ],
            1,
        )
        self.rank = self._env2int(
            [  # MPICH, OpenMPI, IMPI
                "MPI_LOCALRANKID",
                "OMPI_COMM_WORLD_RANK",
                "PMI_RANK",
                "MV2_COMM_WORLD_RANK",
                "RANK",
            ],
            0,
        )
        if layer_para_size is None:
            layer_para_size = 1
        if tensor_para_size is None:
            tensor_para_size = self.world_size // layer_para_size
        self.no_para = tensor_para_size == 1 and layer_para_size == 1
        self.tensor_para_size = tensor_para_size
        self.layer_para_size = layer_para_size
        self.layer_para_batch_size = layer_para_batch_size

        assert (
            self.world_size == tensor_para_size * layer_para_size
        ), "tensor_para_size * layer_para_size must be equal to world_size."
        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.layer_para_rank = self.rank // self.tensor_para_size
        self.is_partial_model = False

    @staticmethod
    def _env2int(env_list, default=-1):
        for e in env_list:
            val = int(os.environ.get(e, -1))
            if val >= 0:
                return val
        return default

    def is_last_group(self):
        r"""
        For layer parallel, only the process corresponding to the last layer
        group can get the predict results. It is used to check whether this is
        the process corresponding to the last layer group.
        """
        return self.layer_para_rank == self.layer_para_size - 1

    def is_load(self, i, num_layer):
        r"""
        Whether or not the given transformer layer of should be loaded to the
        current parallel model. For layer parallel, there is no need not to load
        other layer groups.

        Args:
            i (int): The index of Transformer layer.
            num_layer (int): The number of Transformer layers.

        Returns:
            bool: Indicate whether or not the given transformer layer of should
                be loaded to the current parallel model.
        """
        if self.no_para:
            return True
        # Take into account model only including partial weights.
        if self.is_partial_model:
            return True
        layers_per_device = num_layer // self.layer_para_size
        return (i >= layers_per_device * self.layer_para_rank) and i < layers_per_device * (self.layer_para_rank + 1)

    def slice_weight(self, weight, axis, phase=1, out_param=False):
        r"""
        Get the weight slice for tensor parallel.

        Args:
            weight (Tensor or ndarray): The weight or bias to be sliced.
            axis (int): The axis to perform slice.
            phase (int, optional): 0 is used for creating partial model when
                initializing and `from_pretrained`. While 1 is used in converting
                parameters to FastGeneration. No slice would be performed if
                it is 1, since parameters have been sliced in `phase=0`.
            out_param (bool, optional): If true, `weight` should be a Parameter
                and force the output to be a Parameter.

        Returns:
            Tensor or ndarray: The sliced weight.
        """
        # weight can be parameter/tensor/ndarray
        if self.no_para:
            return weight
        # Take into account model only including partial weights.
        if self.is_partial_model:
            if phase == 1:
                # 0 for init
                # 1 for convert param to FT
                # TODO(guosheng): Maybe we can remove slice_weight in converting
                # parameters to FT if we have sliced parameters at phase 0, while
                # we allow to use non-partial model when converting parameters
                # to FT currently.
                return weight
        if len(weight.shape) == 1:
            axis = 0
        local_size = weight.shape[axis] // self.tensor_para_size
        start_offset = self.tensor_para_rank * local_size
        end_offset = start_offset + local_size
        if len(weight.shape) == 1:
            w_slice = weight[start_offset:end_offset]
        else:
            w_slice = weight[:, start_offset:end_offset] if axis == 1 else weight[start_offset:end_offset, :]
        if out_param:
            # Assume weight is also a Parameter.
            w = type(weight)(shape=w_slice.shape, dtype=weight.dtype, is_bias=len(weight.shape) == 1)
            # NOTE: `VarBase.set_value` would use `w.numpy()` while w is not
            # initialized and can not be used directly.
            # TODO(guosheng): If `w.place `can be used here, use `w.place` to
            # avoid w.place and _current_expected_place are different.
            w.value().get_tensor().set(w_slice, paddle.framework._current_expected_place())
            return w
        else:
            return w_slice

    def set_partial_model(self, is_partial_model):
        r"""
        This is used to set whether or not the current model has complete
        parameters.

        Args:
            is_partial_model (bool): It is used to set whether or not the
                current model has complete parameters.
        """
        self.is_partial_model = is_partial_model

    def fit_partial_model(self, model, state_to_load):
        r"""
        Slice every values included in `state_to_load` according to the shape
        of corresponding parameters in `model`. This is used in `from_pratrained`
        to get sliced parameter values.

        Args:
            model (PretrainedModel): The model to use.
            state_to_load (dict): The state dict including complete parameter
                values of model.

        Returns:
            dict: The state dict contains adjusted values.
        """
        if self.no_para or not self.is_partial_model:
            return state_to_load

        def fit_param(p, v):
            if p.shape[0] != v.shape[0]:
                return _ft_para_conf.slice_weight(v, axis=0, phase=0)
            if len(p.shape) == 2 and p.shape[1] != v.shape[1]:
                return _ft_para_conf.slice_weight(v, axis=1, phase=0)
            return v

        for k, v in model.state_dict().items():
            if k in state_to_load:
                state_to_load[k] = fit_param(v, state_to_load[k])
        return state_to_load


# TODO(guosheng): Maybe use context-manager to allow multiple models.
_ft_para_conf = FTParaConf()


def get_ft_para_conf():
    r"""
    Get settings for model parallel.

    Returns:
        FTParaConf: The settings for model parallel.
    """
    return _ft_para_conf


def enable_ft_para(tensor_para_size=None, layer_para_size=None, layer_para_batch_size=1):
    r"""
    Enable model parallel with the given settings in FastGeneration. Currently only
    support GPT. Please refer to `Megatron <https://arxiv.org/pdf/2104.04473.pdf>`__
    for details.

    Args:
        tensor_para_size (int, optional): The size for tensor parallel. If it is
            1, tensor parallel would not be used. When it is None, tensor parallel
            size would be set as `world_size / layer_para_size`. Default to None.
        layer_para_size (int, optional): The size for layer parallel. If it is
            1, layer parallel would not be used. When it is None, it would be set
            as 1. Default to None.
        layer_para_batch_size (int, optional): The local batch size for pipeline
            parallel. It is suggested to use `batch_size // layer_para_size`.
            Default to 1.
    """
    global _ft_para_conf
    _ft_para_conf = FTParaConf(tensor_para_size, layer_para_size, layer_para_batch_size)
    if _ft_para_conf.no_para:
        return

    def reset_param(layer, attr, axis):
        param = getattr(layer, attr)
        # NOTE: Assignment to parameter 'weight' should be of type Parameter or
        # None. Additionaly, we cannot delattr and setattr which would remove
        # the param from layer._parameters and state_dict, thus cannot fit_partial_model
        param = _ft_para_conf.slice_weight(param, axis, phase=0, out_param=True)
        setattr(layer, attr, param)

    def layer_init_wrapper(func):
        @functools.wraps(func)
        def _impl(self, *args, **kwargs):
            init_dict = fn_args_to_dict(func, *((self,) + args), **kwargs)
            init_dict.pop("self")
            assert (
                init_dict["nhead"] % _ft_para_conf.tensor_para_size == 0
            ), "The number of heads(%d) cannot be evenly divisible by `tensor_para_size`(%d)." % (
                init_dict["nhead"],
                _ft_para_conf.tensor_para_size,
            )
            func(self, *args, **kwargs)
            # Reset parameters with corresponding slice.
            for x, attr in [(m, n) for m in ("q", "k", "v") for n in ("weight", "bias")]:
                reset_param(getattr(self.self_attn, x + "_proj"), attr, 1)
            reset_param(self.self_attn.out_proj, "weight", 0)
            reset_param(self.linear1, "weight", 1)
            reset_param(self.linear1, "bias", 1)
            reset_param(self.linear2, "weight", 0)

        return _impl

    def block_init_wrapper(func):
        @functools.wraps(func)
        def _impl(self, *args, **kwargs):
            init_dict = fn_args_to_dict(func, *((self,) + args), **kwargs)
            init_dict.pop("self")
            num_layers = init_dict["num_hidden_layers"]
            init_dict["num_hidden_layers"] //= _ft_para_conf.layer_para_size
            func(self, **init_dict)
            self.num_layers = num_layers
            self.config["num_hidden_layers"] = num_layers

        return _impl

    def block_state_wrapper(func):
        # TODO(guosheng): Uset state hook instead of block_state_wrapper.
        # self.register_state_dict_hook(reidx_state_layer)
        @functools.wraps(func)
        def _impl(self, *args, **kwargs):
            state_dict = func(self, *args, **kwargs)
            arg_dict = fn_args_to_dict(func, *((self,) + args), **kwargs)
            structured_name_prefix = arg_dict["structured_name_prefix"]

            def reidx_state_layer(state_dict):
                prefix = structured_name_prefix + "decoder.layers."
                prefix_len = len(prefix)
                for name, param in list(state_dict.items()):
                    if name.startswith(prefix):
                        layer_idx_len = 0
                        for i in name[prefix_len:]:
                            if i == ".":
                                break
                            else:
                                layer_idx_len += 1
                        layer_idx = int(name[prefix_len : prefix_len + layer_idx_len])
                        new_name = (
                            name[:prefix_len]
                            + str(_ft_para_conf.layer_para_rank * len(self.decoder.layers) + layer_idx)
                            + name[prefix_len + layer_idx_len :]
                        )
                        state_dict[new_name] = state_dict.pop(name)

            reidx_state_layer(state_dict)
            return state_dict

        return _impl

    # GPT
    layer_init_fn = paddlenlp.transformers.gpt.modeling.TransformerDecoderLayer.__init__
    paddlenlp.transformers.gpt.modeling.TransformerDecoderLayer.__init__ = layer_init_wrapper(layer_init_fn)
    # Note that Transformer block in GPT is not created in TransformerDecoder
    # but in GPTModel.
    block_init_fn = paddlenlp.transformers.gpt.modeling.GPTModel.__init__
    paddlenlp.transformers.gpt.modeling.GPTModel.__init__ = block_init_wrapper(block_init_fn)
    block_state_fn = paddlenlp.transformers.gpt.modeling.GPTModel.state_dict
    paddlenlp.transformers.gpt.modeling.GPTModel.state_dict = block_state_wrapper(block_state_fn)
    # PLATO
    paddle.nn.TransformerEncoderLayer.__init__ = layer_init_wrapper(paddle.nn.TransformerEncoderLayer.__init__)
    _ft_para_conf.set_partial_model(True)
    # TODO(guosheng): Should we set device here, sometimes we want to create
    # models on CPU first to save memory.
    # paddle.set_device("gpu:" + str(_ft_para_conf.rank))
    # yield


class InferOptDecoding(nn.Layer):
    """extract infer model parameters and feed it into the cuda decoder"""

    def __init__(self, model: OPTForCausalLM, decoding_lib=None, use_fp16_decoding=False):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            if "FastGeneration" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FastGeneration"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load(
                "FastGeneration" if get_ft_para_conf().no_para else "FasterTransformerParallel",
                verbose=True,
                need_parallel=not get_ft_para_conf().no_para,
            )

        super(InferOptDecoding, self).__init__()

        self.use_fp16_decoding = use_fp16_decoding
        self.model = model
        self.head_num = self.model.opt.config["num_attention_heads"]
        self.size_per_head = int(self.model.opt.config["hidden_size"] / self.head_num)
        self.num_layer = self.model.opt.config["num_hidden_layers"]
        self.inner_size = self.model.opt.config["intermediate_size"]

        params = convert_params(self, model, fuse_qkv=1, use_fp16=use_fp16_decoding, restore_data=True)

        if self.model.opt.embeddings.project_in is not None:
            self.word_emb = paddle.matmul(
                self.model.opt.embeddings.word_embeddings.weight, self.model.opt.embeddings.project_in.weight
            )
            # set the linear_weight
            self.linear_weight = paddle.matmul(
                self.model.opt.embeddings.word_embeddings.weight, self.model.opt.decoder.project_out.weight.T
            )
        else:
            self.word_emb = self.model.opt.embeddings.word_embeddings.weight
            self.linear_weight = self.model.opt.embeddings.word_embeddings.weight

        # reset the offset in position embedding
        position_embedding = self.model.opt.embeddings.position_embeddings
        self.pos_emb = paddle.concat([position_embedding.weight[2:], position_embedding.weight[:2]])

        # if there is no final layer norm, pass empty tensor to fusion opt op
        final_layer_norm = self.model.opt.decoder.final_layer_norm
        if final_layer_norm is None:
            self.decoder_ln_weight = paddle.empty(shape=[0])
            self.decoder_ln_bias = paddle.empty(shape=[0])
        else:
            self.decoder_ln_weight = final_layer_norm.weight
            self.decoder_ln_bias = final_layer_norm.bias

        self.normalize_before = self.model.decoder.final_layer_norm is not None

        for k, v in params.items():
            setattr(self, k, v)

        # check the dtype of embedding
        dtype = "float16" if use_fp16_decoding else "float32"
        self.word_emb = transfer_param(self.word_emb, dtype=dtype, is_bias=False, restore_data=True)
        self.linear_weight = transfer_param(self.linear_weight, dtype=dtype, is_bias=False, restore_data=True)
        self.pos_emb = transfer_param(self.pos_emb, dtype=dtype, is_bias=False, restore_data=True)
        self.decoder_ln_weight = transfer_param(self.decoder_ln_weight, dtype=dtype, is_bias=False, restore_data=True)
        self.decoder_ln_bias = transfer_param(self.decoder_ln_bias, dtype=dtype, is_bias=True, restore_data=True)

    def forward(
        self,
        input_ids,
        mem_seq_len,
        attention_mask=None,
        topk=4,
        topp=0.0,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        forced_eos_token_id=None,
        max_out_len=256,
        temperature=1,
    ):
        if attention_mask is None:
            batch_size = paddle.shape(input_ids)[0]
            attention_mask = paddle.tril(
                paddle.ones(
                    [batch_size, mem_seq_len, mem_seq_len], dtype="float16" if self.use_fp16_decoding else "float32"
                )
            )
        elif self.use_fp16_decoding and attention_mask.dtype == paddle.float32:
            attention_mask = paddle.cast(attention_mask, dtype="float16")

        output_ids = infer_opt_decoding(
            input=[input_ids],
            attn_mask=[attention_mask],
            mem_seq_len=[mem_seq_len],
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
            normalize_before=self.normalize_before,
            topk=topk,
            topp=topp,
            max_out_len=max_out_len,
            head_num=self.head_num,
            size_per_head=self.size_per_head,
            num_layer=self.num_layer,
            bos_id=bos_token_id,
            eos_id=eos_token_id,
            temperature=temperature,
            use_fp16_decoding=self.use_fp16_decoding,
        )

        output_ids = output_ids[paddle.shape(input_ids)[-1] :, :]
        if forced_eos_token_id is not None:
            output_ids[:, -1] = forced_eos_token_id
        return output_ids


class InferGptDecoding(nn.Layer):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            if "FastGeneration" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FastGeneration"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load(
                "FastGeneration" if get_ft_para_conf().no_para else "FasterTransformerParallel",
                verbose=True,
                need_parallel=not get_ft_para_conf().no_para,
            )

        super(InferGptDecoding, self).__init__()

        self.use_fp16_decoding = use_fp16_decoding
        self.model = model
        self.head_num = self.model.gpt.config["num_attention_heads"]
        self.size_per_head = int(self.model.gpt.config["hidden_size"] / self.head_num)
        self.num_layer = self.model.gpt.config["num_hidden_layers"]
        self.inner_size = self.model.gpt.config["intermediate_size"]

        params = convert_params(self, model, fuse_qkv=1, use_fp16=use_fp16_decoding, restore_data=True)
        params["word_emb"].append((self.model.gpt.embeddings.word_embeddings, "weight"))
        params["pos_emb"].append((self.model.gpt.embeddings.position_embeddings, "weight"))

        # if model share word_embeddings weight
        if id(self.model.gpt.embeddings.word_embeddings) == id(self.model.lm_head.decoder.weight):
            params["linear_weight"].append((self.model.gpt.embeddings.word_embeddings, "weight"))
        else:
            params["linear_weight"].append(
                (self.model.lm_head.decoder.weight, False, partial(setattr, self, "weight"))
            )

        for k, v in params.items():
            setattr(self, k, v)

    def forward(
        self,
        input_ids,
        mem_seq_len,
        attention_mask=None,
        topk=4,
        topp=0.0,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        forced_eos_token_id=None,
        max_out_len=256,
        temperature=1,
    ):
        if attention_mask is None:
            batch_size = paddle.shape(input_ids)[0]
            attention_mask = paddle.tril(
                paddle.ones(
                    [batch_size, paddle.max(mem_seq_len), paddle.max(mem_seq_len)],
                    dtype="float16" if self.use_fp16_decoding else "float32",
                )
            )
        elif self.use_fp16_decoding and attention_mask.dtype == paddle.float32:
            attention_mask = paddle.cast(attention_mask, dtype="float16")

        (output_ids,) = infer_gpt_decoding(
            input=[input_ids],
            attn_mask=[attention_mask],
            mem_seq_len=[mem_seq_len],
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
            topk=topk,
            topp=topp,
            max_out_len=max_out_len,
            head_num=self.head_num,
            size_per_head=self.size_per_head,
            num_layer=self.num_layer,
            bos_id=bos_token_id,
            eos_id=eos_token_id,
            temperature=temperature,
            use_fp16_decoding=self.use_fp16_decoding,
        )

        output_ids = output_ids[paddle.shape(input_ids)[-1] :, :]
        if forced_eos_token_id is not None:
            output_ids[:, -1] = forced_eos_token_id
        return output_ids


class InferUnifiedDecoding(nn.Layer):
    def __init__(
        self,
        model,
        decoding_lib=None,
        use_fp16_decoding=False,
        logits_mask=None,
        n_head=8,
        hidden_dims=512,
        size_per_head=64,
        n_layer=6,
        unk_id=0,
        mask_id=30000,
        normalize_before=True,
        hidden_act="gelu",
    ):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FastGeneration" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FastGeneration"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load(
                "FastGeneration" if get_ft_para_conf().no_para else "FasterTransformerParallel",
                verbose=True,
                need_parallel=not get_ft_para_conf().no_para,
            )

        super(InferUnifiedDecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in ["self"]:
                setattr(self, "_" + arg, value)

        params = convert_params(self, model, fuse_qkv=1, use_fp16=use_fp16_decoding, restore_data=True)
        params["word_emb"].append((model.embeddings.word_embeddings, "weight"))
        params["pos_emb"].append((model.embeddings.position_embeddings, "weight"))
        params["type_emb"].append((model.embeddings.token_type_embeddings, "weight"))
        if getattr(model.embeddings, "role_embeddings", None) is not None:
            params["role_emb"].append((model.embeddings.role_embeddings, "weight"))
        else:
            # inputs of custom op cannot be None
            params["role_emb"].append((paddle.zeros(shape=[1]), False, partial(setattr, self, "default_role_emb")))
        if not self._normalize_before:
            # pre-norm params has been converted in `convert_params`, and this
            # is only for post-norm such as UNIMO.
            params["decoder_ln_weight"].append((model.encoder_norm, "weight"))
            params["decoder_ln_bias"].append((model.encoder_norm, "bias"))
        params["trans_weight"].append((model.lm_head.transform, "weight"))
        params["trans_bias"].append((model.lm_head.transform, "bias"))
        params["lm_ln_weight"].append((model.lm_head.layer_norm, "weight"))
        params["lm_ln_bias"].append((model.lm_head.layer_norm, "bias"))
        # NOTE: newly created tensors should be layer attribute refered to be
        # able to convert to static graph.
        params["linear_weight"].append((model.lm_head.decoder_weight.t(), False, partial(setattr, self, "dec_weight")))
        params["linear_bias"].append(
            (paddle.assign(model.lm_head.decoder_bias), True, partial(setattr, self, "dec_bias"))
        )
        for k, v in params.items():
            setattr(self, k, v)

    def forward(
        self,
        input_ids,
        attn_mask,
        memory_seq_lens,
        type_id,
        decoder_type_id,
        role_id=None,
        decoder_role_id=None,
        position_id=None,
        decoder_position_id=None,
        beam_size=4,
        topk=4,
        topp=0.0,
        decoding_strategy="greedy_search",
        max_out_len=256,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        forced_eos_token_id=None,
        temperature=1.0,
        length_penalty=1.0,
        diversity_rate=0.0,
        pos_bias=True,
        rel_len=False,
        early_stopping=False,
        min_length=0,
    ):
        if role_id is None:
            role_id = paddle.zeros(shape=[0], dtype="int32")
            decoder_role_id = paddle.zeros(shape=[0], dtype="int32")
        if position_id is None:
            position_id = paddle.zeros(shape=[0], dtype="int32")
            decoder_position_id = paddle.zeros(shape=[0], dtype="int32")

        if decoding_strategy == "greedy_search":
            decoding_strategy = "topk_sampling"
            topk = 1
            topp = 0.0
        elif decoding_strategy in ["sampling", "topk_sampling", "topp_sampling"]:
            if topp == 1 and topk > 0:
                decoding_strategy = "topk_sampling"
                topp = 0.0
            elif topp > 0 and topk == 0:
                decoding_strategy = "topp_sampling"
            else:
                raise AttributeError(
                    "Only topk sampling or topp sampling are supported. "
                    "Topk sampling and topp sampling cannot be both applied in the fast version."
                )
        elif decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v3"

        output_ids, parent_ids, sequence_length, output_scores = infer_unified_decoding(
            input_ids=[input_ids],
            attn_mask=[attn_mask],
            memory_seq_lens=[memory_seq_lens],
            type_id=[type_id],
            decoder_type_id=[decoder_type_id],
            logits_mask=[self._logits_mask],
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
            trans_weight=self.trans_weight,
            trans_bias=self.trans_bias,
            lm_ln_weight=self.lm_ln_weight,
            lm_ln_bias=self.lm_ln_bias,
            linear_weight=self.linear_weight,
            linear_bias=self.linear_bias,
            pos_emb=self.pos_emb,
            type_emb=self.type_emb,
            role_id=[role_id],
            decoder_role_id=[decoder_role_id],
            role_emb=self.role_emb,
            position_id=[position_id],
            decoder_position_id=[decoder_position_id],
            _decoding_strategy=decoding_strategy,
            _beam_size=beam_size,
            _topk=topk,
            _topp=topp,
            _n_head=self._n_head,
            _size_per_head=self._size_per_head,
            _n_layer=self._n_layer,
            _bos_id=bos_token_id,
            _eos_id=eos_token_id,
            _max_out_len=max_out_len,
            _diversity_rate=-diversity_rate,
            _unk_id=self._unk_id,
            _mask_id=self._mask_id,
            _temperature=temperature,
            _len_penalty=length_penalty,
            _normalize_before=self._normalize_before,
            _pos_bias=pos_bias,
            _hidden_act=self._hidden_act,
            _rel_len=rel_len,
            _early_stopping=early_stopping,
            _min_length=min_length,
        )
        ids = finalize(
            beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            forced_eos_token_id=forced_eos_token_id,
            decoding_strategy=decoding_strategy,
        )
        return ids, output_scores


class InferMIRODecoding(nn.Layer):
    def __init__(
        self,
        model,
        decoding_lib=None,
        use_fp16_decoding=False,
        logits_mask=None,
        n_head=8,
        hidden_dims=512,
        size_per_head=64,
        n_layer=6,
        unk_id=0,
        mask_id=30000,
        normalize_before=True,
        hidden_act="relu",
    ):

        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FasterTransformer" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FasterTransformer"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load(
                "FasterTransformer" if get_ft_para_conf().no_para else "FasterTransformerParallel",
                verbose=True,
                need_parallel=not get_ft_para_conf().no_para,
            )

        super(InferMIRODecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in ["self"]:
                setattr(self, "_" + arg, value)

        params = convert_params(self, model, fuse_qkv=1, use_fp16=use_fp16_decoding, restore_data=True)
        params["word_emb"].append((model.embeddings.word_embeddings, "weight"))
        params["pos_emb"].append((model.embeddings.position_embeddings, "weight"))
        params["type_emb"].append((model.embeddings.token_type_embeddings, "weight"))
        if getattr(model.embeddings, "role_embeddings", None) is not None:
            params["role_emb"].append((model.embeddings.role_embeddings, "weight"))
        else:
            # inputs of custom op cannot be None
            params["role_emb"].append((paddle.zeros(shape=[1]), False, partial(setattr, self, "default_role_emb")))
        # if not self._normalize_before:
        #     # pre-norm params has been converted in `convert_params`, and this
        #     # is only for post-norm such as UNIMO.
        #     params["decoder_ln_weight"].append((model.encoder_norm, "weight"))
        #     params["decoder_ln_bias"].append((model.encoder_norm, "bias"))
        params["pre_decoder_ln_weight"].append((model.encoder_norm, "weight"))
        params["pre_decoder_ln_bias"].append((model.encoder_norm, "bias"))

        params["trans_weight"].append((model.lm_head.transform, "weight"))
        params["trans_bias"].append((model.lm_head.transform, "bias"))
        params["lm_ln_weight"].append((model.lm_head.layer_norm, "weight"))
        params["lm_ln_bias"].append((model.lm_head.layer_norm, "bias"))
        # NOTE: newly created tensors should be layer attribute refered to be
        # able to convert to static graph.
        params["linear_weight"].append((model.lm_head.decoder_weight.t(), False, partial(setattr, self, "dec_weight")))
        params["linear_bias"].append(
            (paddle.assign(model.lm_head.decoder_bias), True, partial(setattr, self, "dec_bias"))
        )
        for k, v in params.items():
            setattr(self, k, v)

    def forward(
        self,
        input_ids,
        attn_mask,
        memory_seq_lens,
        type_id,
        decoder_type_id,
        role_id=None,
        decoder_role_id=None,
        position_id=None,
        decoder_position_id=None,
        beam_size=4,
        topk=4,
        topp=0.0,
        decoding_strategy="greedy_search",
        max_out_len=256,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        forced_eos_token_id=None,
        temperature=1.0,
        length_penalty=1.0,
        diversity_rate=0.0,
        pos_bias=True,
        rel_len=False,
        early_stopping=False,
        min_length=0,
    ):
        if role_id is None:
            role_id = paddle.zeros(shape=[0], dtype="int32")
            decoder_role_id = paddle.zeros(shape=[0], dtype="int32")
        if position_id is None:
            position_id = paddle.zeros(shape=[0], dtype="int32")
            decoder_position_id = paddle.zeros(shape=[0], dtype="int32")

        if decoding_strategy == "greedy_search":
            decoding_strategy = "topk_sampling"
            topk = 1
            topp = 0.0
        elif decoding_strategy in ["sampling", "topk_sampling", "topp_sampling"]:
            if topp == 1 and topk > 0:
                decoding_strategy = "topk_sampling"
                topp = 0.0
            elif topp > 0 and topk == 0:
                decoding_strategy = "topp_sampling"
            else:
                raise AttributeError(
                    "Only topk sampling or topp sampling are supported. "
                    "Topk sampling and topp sampling cannot be both applied in the faster version."
                )
        elif decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v3"

        output_ids, parent_ids, sequence_length, output_scores = infer_miro_decoding(
            input_ids=[input_ids],
            attn_mask=[attn_mask],
            memory_seq_lens=[memory_seq_lens],
            type_id=[type_id],
            decoder_type_id=[decoder_type_id],
            logits_mask=[self._logits_mask],
            word_emb=self.word_emb,
            pre_decoder_ln_weight=self.pre_decoder_ln_weight,
            pre_decoder_ln_bias=self.pre_decoder_ln_bias,
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
            trans_weight=self.trans_weight,
            trans_bias=self.trans_bias,
            lm_ln_weight=self.lm_ln_weight,
            lm_ln_bias=self.lm_ln_bias,
            linear_weight=self.linear_weight,
            linear_bias=self.linear_bias,
            pos_emb=self.pos_emb,
            type_emb=self.type_emb,
            role_id=[role_id],
            decoder_role_id=[decoder_role_id],
            role_emb=self.role_emb,
            position_id=[position_id],
            decoder_position_id=[decoder_position_id],
            _decoding_strategy=decoding_strategy,
            _beam_size=beam_size,
            _topk=topk,
            _topp=topp,
            _n_head=self._n_head,
            _size_per_head=self._size_per_head,
            _n_layer=self._n_layer,
            _bos_id=bos_token_id,
            _eos_id=eos_token_id,
            _max_out_len=max_out_len,
            _diversity_rate=-diversity_rate,
            _unk_id=self._unk_id,
            _mask_id=self._mask_id,
            _temperature=temperature,
            _len_penalty=length_penalty,
            _normalize_before=self._normalize_before,
            _pos_bias=pos_bias,
            _hidden_act=self._hidden_act,
            _rel_len=rel_len,
            _early_stopping=early_stopping,
            _min_length=min_length,
        )

        ids = finalize(
            beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            forced_eos_token_id=forced_eos_token_id,
            decoding_strategy=decoding_strategy,
        )

        return ids, output_scores


class InferBartDecoding(nn.Layer):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FastGeneration" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FastGeneration"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load("FastGeneration", verbose=True)

        super(InferBartDecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in ["self", "model", "word_embedding", "positional_embedding", "linear"]:
                setattr(self, "_" + arg, value)
        self._num_decoder_layers = model.bart.config["decoder_layers"]
        self._n_head = model.bart.config["decoder_attention_heads"]
        self._d_model = model.bart.config["d_model"]

        params = convert_params(self, model.get_decoder(), fuse_qkv=2, use_fp16=use_fp16_decoding, restore_data=True)
        params["decoder_ln_weight"].append((model.decoder.decoder_layernorm_embedding, "weight"))
        params["decoder_ln_bias"].append((model.decoder.decoder_layernorm_embedding, "bias"))
        params["word_emb"].append((model.decoder.embed_tokens, "weight"))
        params["pos_emb"].append((model.decoder.decoder_embed_positions, "weight"))
        params["linear_weight"].append((model.lm_head_weight.t(), False, partial(setattr, self, "lm_head_weight_")))
        params["linear_bias"].append((model.final_logits_bias, True, partial(setattr, self, "lm_head_bias_")))
        for k, v in params.items():
            setattr(self, k, v)

    def forward(
        self,
        enc_output,
        memory_seq_lens,
        beam_size=4,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        decoding_strategy="beam_search_v3",
        max_out_len=256,
        min_out_len=256,
        diversity_rate=0.0,
        rel_len=False,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        forced_eos_token_id=None,
        alpha=0.6,
        early_stopping=False,
    ):
        # beam_search/beam_search_v2/beam_search_v3 should be corrected to beam_search_v3.
        if decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v3"
        elif decoding_strategy == "greedy_search":
            decoding_strategy = "topk_sampling"
            top_k = 1
            top_p = 0.0
        elif decoding_strategy in ["sampling", "topk_sampling", "topp_sampling"]:
            if top_p == 1 and top_k > 0:
                decoding_strategy = "topk_sampling"
                top_p = 0.0
            elif top_p > 0 and top_k == 0:
                decoding_strategy = "topp_sampling"
            else:
                raise AttributeError(
                    "Only topk sampling or topp sampling are supported. "
                    "Topk sampling and topp sampling cannot be both applied in the fast version. "
                )

        output_ids, parent_ids, sequence_length = infer_bart_decoding(
            [enc_output],
            [memory_seq_lens],
            self.word_emb,
            self.slf_ln_weight,
            self.slf_ln_bias,
            self.slf_q_weight,
            self.slf_q_bias,
            self.slf_k_weight,
            self.slf_k_bias,
            self.slf_v_weight,
            self.slf_v_bias,
            self.slf_out_weight,
            self.slf_out_bias,
            self.cross_ln_weight,
            self.cross_ln_bias,
            self.cross_q_weight,
            self.cross_q_bias,
            self.cross_k_weight,
            self.cross_k_bias,
            self.cross_v_weight,
            self.cross_v_bias,
            self.cross_out_weight,
            self.cross_out_bias,
            self.ffn_ln_weight,
            self.ffn_ln_bias,
            self.ffn_inter_weight,
            self.ffn_inter_bias,
            self.ffn_out_weight,
            self.ffn_out_bias,
            self.decoder_ln_weight,
            self.decoder_ln_bias,
            self.linear_weight,
            self.linear_bias,
            self.pos_emb,
            decoding_strategy,
            beam_size,
            top_k,
            top_p,
            temperature,
            self._n_head,
            int(self._d_model / self._n_head),
            self._num_decoder_layers,
            bos_token_id,
            eos_token_id,
            max_out_len,
            min_out_len,
            -diversity_rate,
            rel_len,
            alpha,
            early_stopping,
        )

        ids = finalize(
            beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            forced_eos_token_id=forced_eos_token_id,
            decoding_strategy=decoding_strategy,
        )
        return ids


class InferMBartDecoding(nn.Layer):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False, hidden_act="gelu"):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FastGeneration" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FastGeneration"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load("FastGeneration", verbose=True)

        super(InferMBartDecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in ["self", "model", "word_embedding", "positional_embedding", "linear"]:
                setattr(self, "_" + arg, value)
        self._num_decoder_layers = model.mbart.config["decoder_layers"]
        self._n_head = model.mbart.config["decoder_attention_heads"]
        self._d_model = model.mbart.config["d_model"]

        # process weights
        if use_fp16_decoding:
            for mod in model.mbart.decoder.decoder.layers:
                mod.norm1.weight = transfer_param(mod.norm1.weight, restore_data=True)
                mod.norm1.bias = transfer_param(mod.norm1.bias, is_bias=True, restore_data=True)
                mod.self_attn.q_proj.weight = transfer_param(mod.self_attn.q_proj.weight, restore_data=True)
                mod.self_attn.q_proj.bias = transfer_param(mod.self_attn.q_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.k_proj.weight = transfer_param(mod.self_attn.k_proj.weight, restore_data=True)
                mod.self_attn.k_proj.bias = transfer_param(mod.self_attn.k_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.v_proj.weight = transfer_param(mod.self_attn.v_proj.weight, restore_data=True)
                mod.self_attn.v_proj.bias = transfer_param(mod.self_attn.v_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.out_proj.weight = transfer_param(mod.self_attn.out_proj.weight, restore_data=True)
                mod.self_attn.out_proj.bias = transfer_param(
                    mod.self_attn.out_proj.bias, is_bias=True, restore_data=True
                )

                mod.norm2.weight = transfer_param(mod.norm2.weight, restore_data=True)
                mod.norm2.bias = transfer_param(mod.norm2.bias, is_bias=True, restore_data=True)
                mod.cross_attn.q_proj.weight = transfer_param(mod.cross_attn.q_proj.weight, restore_data=True)
                mod.cross_attn.q_proj.bias = transfer_param(
                    mod.cross_attn.q_proj.bias, is_bias=True, restore_data=True
                )
                mod.cross_attn.k_proj.weight = transfer_param(mod.cross_attn.k_proj.weight, restore_data=True)
                mod.cross_attn.k_proj.bias = transfer_param(
                    mod.cross_attn.k_proj.bias, is_bias=True, restore_data=True
                )
                mod.cross_attn.v_proj.weight = transfer_param(mod.cross_attn.v_proj.weight, restore_data=True)
                mod.cross_attn.v_proj.bias = transfer_param(
                    mod.cross_attn.v_proj.bias, is_bias=True, restore_data=True
                )
                mod.cross_attn.out_proj.weight = transfer_param(mod.cross_attn.out_proj.weight, restore_data=True)
                mod.cross_attn.out_proj.bias = transfer_param(
                    mod.cross_attn.out_proj.bias, is_bias=True, restore_data=True
                )

                mod.norm3.weight = transfer_param(mod.norm3.weight, restore_data=True)
                mod.norm3.bias = transfer_param(mod.norm3.bias, is_bias=True, restore_data=True)
                mod.linear1.weight = transfer_param(mod.linear1.weight, restore_data=True)
                mod.linear1.bias = transfer_param(mod.linear1.bias, is_bias=True, restore_data=True)
                mod.linear2.weight = transfer_param(mod.linear2.weight, restore_data=True)
                mod.linear2.bias = transfer_param(mod.linear2.bias, is_bias=True, restore_data=True)

            model.decoder.decoder_layernorm_embedding.weight = transfer_param(
                model.decoder.decoder_layernorm_embedding.weight, restore_data=True
            )
            model.decoder.decoder_layernorm_embedding.bias = transfer_param(
                model.decoder.decoder_layernorm_embedding.bias, is_bias=True, restore_data=True
            )

            model.decoder.decoder.norm.weight = transfer_param(model.decoder.decoder.norm.weight, restore_data=True)
            model.decoder.decoder.norm.bias = transfer_param(
                model.decoder.decoder.norm.bias, is_bias=True, restore_data=True
            )

            model.lm_head_weight = transfer_param(model.lm_head_weight, restore_data=True)
            model.final_logits_bias = transfer_param(model.final_logits_bias, is_bias=True, restore_data=True)

            model.decoder.decoder_embed_positions.weight = transfer_param(
                model.decoder.decoder_embed_positions.weight, restore_data=True
            )
            model.decoder.embed_tokens.weight = transfer_param(model.decoder.embed_tokens.weight, restore_data=True)

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

        for mod in model.mbart.decoder.decoder.layers:
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

        self.decoder_ln_weight = [model.decoder.decoder.norm.weight]
        self.decoder_ln_bias = [model.decoder.decoder.norm.bias]

        self.mbart_ln_weight = [model.decoder.decoder_layernorm_embedding.weight]
        self.mbart_ln_bias = [model.decoder.decoder_layernorm_embedding.bias]

        self.pos_emb = [model.decoder.decoder_embed_positions.weight]
        self.word_emb = [model.decoder.embed_tokens.weight]

        setattr(self, "lm_head_weight_", model.lm_head_weight.t())
        self.linear_weight = [getattr(self, "lm_head_weight_")]
        self.linear_bias = [model.final_logits_bias]

    def forward(
        self,
        enc_output,
        memory_seq_lens,
        trg_word=None,
        beam_size=4,
        top_k=1,
        top_p=0.0,
        decoding_strategy="beam_search_v3",
        max_out_len=256,
        diversity_rate=0.0,
        rel_len=False,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        alpha=0.6,
        temperature=1.0,
        early_stopping=False,
    ):
        # Beam_search/beam_search_v2/beam_search_v3 should be corrected to beam_search_v3.
        if decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v3"
        elif decoding_strategy == "greedy_search":
            decoding_strategy = "topk_sampling"
            top_k = 1
            top_p = 0.0
        elif decoding_strategy in ["sampling", "topk_sampling", "topp_sampling"]:
            if top_p == 1 and top_k > 0:
                decoding_strategy = "topk_sampling"
                top_p = 0.0
            elif top_p > 0 and top_k == 0:
                decoding_strategy = "topp_sampling"
            else:
                raise AttributeError(
                    "Only topk sampling or topp sampling are supported. "
                    "Topk sampling and topp sampling cannot be both applied in the fast version. "
                )
        output_ids, parent_ids, sequence_length = infer_mbart_decoding(
            [enc_output],
            [memory_seq_lens],
            self.word_emb,
            self.slf_ln_weight,
            self.slf_ln_bias,
            self.slf_q_weight,
            self.slf_q_bias,
            self.slf_k_weight,
            self.slf_k_bias,
            self.slf_v_weight,
            self.slf_v_bias,
            self.slf_out_weight,
            self.slf_out_bias,
            self.cross_ln_weight,
            self.cross_ln_bias,
            self.cross_q_weight,
            self.cross_q_bias,
            self.cross_k_weight,
            self.cross_k_bias,
            self.cross_v_weight,
            self.cross_v_bias,
            self.cross_out_weight,
            self.cross_out_bias,
            self.ffn_ln_weight,
            self.ffn_ln_bias,
            self.ffn_inter_weight,
            self.ffn_inter_bias,
            self.ffn_out_weight,
            self.ffn_out_bias,
            self.decoder_ln_weight,
            self.decoder_ln_bias,
            self.mbart_ln_weight,
            self.mbart_ln_bias,
            self.linear_weight,
            self.linear_bias,
            self.pos_emb,
            trg_word,
            decoding_strategy,
            beam_size,
            top_k,
            top_p,
            self._n_head,
            int(self._d_model / self._n_head),
            self._num_decoder_layers,
            bos_token_id,
            eos_token_id,
            max_out_len,
            -diversity_rate,
            rel_len,
            alpha,
            temperature,
            early_stopping,
            self._hidden_act,
        )

        ids = finalize(beam_size, output_ids, parent_ids, sequence_length, decoding_strategy=decoding_strategy)
        return ids


def convert_gptj_params(fast_model, model, fuse_qkv=1, use_fp16=False, restore_data=False, permutation=None):
    r"""
    Convert parameters included in Transformer layer  from original models
    to the format of faster models.

    Args:
        fast_model (Layer): The faster model object.
        model (Layer): The Transformer layer.
        fuse_qkv (int): 0 for nofuse, 1 for fuse, 2 for fuse and delete the
            unfused parameters. If environment variable `PPFG_QKV_MEM_OPT` is
            set and the weights of q/k/v is fused, it will try to delete the
            original unfused weights. Note the rollback to original model would
            not be guarantee anymore when the faster model failed if the original
            weights are deleted. Default to 1.
        use_fp16 (bool): Whether to use float16. Maybe we should use the default
            dtype as the highest priority later. Default to `False`.
        restore_data (bool): If `False`, need to reload the weight values. It
            should be `True` for weight loaded models. Default to `False`.

    Returns:
        defaultdict: Each value is a list including converted parameters in all
            layers. For other parameters not included in Transformer module to
            be converted, such as embeddings, you can achieve it by using the
            returned dict `params` though `params['word_emb'].append()` directly
            which would do CPU/GPU and fp32/fp16 transfer automatically.
    """
    if fuse_qkv == 1:
        fuse_qkv = 2 if os.getenv("PPFG_QKV_MEM_OPT", "0") == "1" else 1
    ft_para_conf = get_ft_para_conf()

    class _list(list):
        def append(self, item):
            def attr_handle_func(x):
                return x

            if isinstance(item[0], nn.Layer):
                # Axis is used for tensor slice in tensor parallel.
                # Use None to make no slice on the tensor.
                if len(item) == 2:
                    layer, attr = item
                    axis = None
                else:
                    layer, attr, axis = item
                param = getattr(layer, attr)
                if axis is not None and isinstance(layer, nn.Linear):
                    param = ft_para_conf.slice_weight(param, axis)
                param = transfer_param(
                    param,
                    is_bias=attr.endswith("bias"),
                    dtype="float16" if use_fp16 else "float32",
                    restore_data=restore_data,
                )
                # NOTE: Assignment to parameter 'weight' should be of type
                # Parameter or None, thus delete first in case of param is
                # a tensor.
                # TODO(guosheng): Make slice_weight use `output_param=True`
                # and remove delattr. Currently, if `param` is Tensor rather
                # than Parameter, it would not be in state_dict.
                delattr(layer, attr)
                setattr(layer, attr, param)
            else:
                # NOTE: Compared with if branch, there is no layer attribute
                # refered to the transfered param, thus we should set it as
                # the layer attribute to be able to convert to static graph.
                # Additionally, we suppose no need to process tensor parallel
                # here since the param passed in might have been processed.
                if len(item) == 2:
                    param, is_bias = item
                    attr_handle = attr_handle_func
                else:
                    param, is_bias, attr_handle = item
                param = transfer_param(
                    param, is_bias=is_bias, dtype="float16" if use_fp16 else "float32", restore_data=restore_data
                )
                attr_handle(param)
            return super().append(param)

    params = defaultdict(_list)

    def _convert(module):
        num_layer = len(module)
        for i, layer in enumerate(module):
            if not ft_para_conf.is_load(i, num_layer):
                continue
            # TODO(guosheng): Tensor with size 0 might be failed in
            # paddle develop, thus use tensor with size 1 instead
            # temporarily. Besides, we use 2D tensor since jit log
            # requires that on linear weight. While size 0 seems all
            # right in jit.to_static/jit.save.
            dummy_tensor = paddle.zeros([1, 1])
            if permutation is not None:
                qkv = layer.attn.qkv_proj.weight.numpy()
                qkv = qkv[:, permutation]
                if fuse_qkv == 2:
                    del layer.attn.qkv_proj.weight
                    setattr(layer.attn.qkv_proj, "weight", dummy_tensor)
                w = paddle.to_tensor(qkv)
            else:
                w = _convert_qkv(
                    layer.attn.q_proj,
                    layer.attn.k_proj,
                    layer.attn.v_proj,
                    attr="weight",
                    use_numpy=fuse_qkv == 2,
                    del_param=fuse_qkv == 2,
                    dummy_tensor=dummy_tensor,
                )
            params["slf_q_weight"].append((w, False))
            # NOTE: Use `params["slf_q_weight"][-1]` rather than `w`,
            # since the appended tensor might be a new transfered tensor.
            # Besides, to allow convert_params be called more than once,
            # we find a attr name not existing to avoid overwriting the
            # existing attr.
            attr = "slf_q_weight_" + str(i)
            while hasattr(fast_model, attr):
                attr += "_"
            setattr(fast_model, attr, params["slf_q_weight"][-1])

            params["slf_out_weight"].append((layer.attn.out_proj, "weight", 0))
            params["slf_ln_weight"].append((layer.ln_1, "weight"))
            params["slf_ln_bias"].append((layer.ln_1, "bias"))
            # Slice tensor when append according to axis(1 or 0) if parallel
            # is enable.
            params["ffn_inter_weight"].append((layer.mlp.fc_in, "weight", 1))
            params["ffn_inter_bias"].append((layer.mlp.fc_in, "bias", 1))
            params["ffn_out_weight"].append((layer.mlp.fc_out, "weight", 0))
            params["ffn_out_bias"].append((layer.mlp.fc_out, "bias"))

    _convert(model)
    return params


class InferGptJDecoding(nn.Layer):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False, transpose_qkv=False):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            if "FastGeneration" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FastGeneration"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load(
                "FastGeneration" if get_ft_para_conf().no_para else "FasterTransformerParallel",
                verbose=True,
                need_parallel=not get_ft_para_conf().no_para,
            )

        super(InferGptJDecoding, self).__init__()

        self.use_fp16_decoding = use_fp16_decoding
        self.model = model
        self.head_num = self.model.transformer.config["n_head"]
        self.size_per_head = int(self.model.transformer.config["n_embd"] / self.head_num)
        self.num_layer = self.model.transformer.config["n_layer"]
        self.rotary_embedding_dim = self.model.transformer.config["rotary_dim"]
        logger.info("Converting model weights, it will cost a few seconds.....")
        permutation = None
        if transpose_qkv:
            # GPTJ is different with CodeGen in attention project layer.
            local_dim = self.model.transformer.config["n_embd"] // 4
            base_permutation = [0, 3, 6, 9, 2, 5, 8, 11, 1, 4, 7, 10]
            permutation = np.concatenate([np.arange(i * local_dim, (i + 1) * local_dim) for i in base_permutation])
        params = convert_gptj_params(
            self,
            model.transformer.h,
            fuse_qkv=2,
            use_fp16=use_fp16_decoding,
            restore_data=True,
            permutation=permutation,
        )

        params["word_emb"].append((self.model.transformer.wte, "weight"))
        params["decoder_ln_weight"].append((self.model.transformer.ln_f, "weight"))
        params["decoder_ln_bias"].append((self.model.transformer.ln_f, "bias"))
        params["linear_weight"].append((self.model.lm_head.weight.t(), partial(setattr, self, "linear_weight_out")))
        params["linear_bias"].append((self.model.lm_head, "bias"))

        for k, v in params.items():
            setattr(self, k, v)
        logger.info("Already converted model weights.")

    def forward(
        self,
        input_ids,
        mem_seq_len,
        attention_mask=None,
        topk=4,
        topp=0.0,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        forced_eos_token_id=None,
        max_out_len=256,
        temperature=1,
        repetition_penalty=1.0,
        min_length=0,
    ):
        if attention_mask is None:
            batch_size, input_length = paddle.shape(input_ids)
            attention_mask = paddle.unsqueeze((input_ids != pad_token_id).astype("float32"), axis=[1])
            causal_mask = paddle.tril(paddle.ones([batch_size, input_length, input_length], dtype="float32"))
            attention_mask = paddle.logical_and(attention_mask, causal_mask)
            if not self.use_fp16_decoding:
                attention_mask = paddle.cast(attention_mask, dtype="float32")
            else:
                attention_mask = paddle.cast(attention_mask, dtype="float16")

        if self.use_fp16_decoding and attention_mask.dtype == paddle.float32:
            attention_mask = paddle.cast(attention_mask, dtype="float16")

        (output_ids,) = infer_gptj_decoding(
            input=[input_ids],
            attn_mask=[attention_mask],
            mem_seq_len=[mem_seq_len],
            word_emb=self.word_emb,
            slf_ln_weight=self.slf_ln_weight,
            slf_ln_bias=self.slf_ln_bias,
            slf_q_weight=self.slf_q_weight,
            slf_out_weight=self.slf_out_weight,
            ffn_inter_weight=self.ffn_inter_weight,
            ffn_inter_bias=self.ffn_inter_bias,
            ffn_out_weight=self.ffn_out_weight,
            ffn_out_bias=self.ffn_out_bias,
            decoder_ln_weight=self.decoder_ln_weight,
            decoder_ln_bias=self.decoder_ln_bias,
            linear_weight=self.linear_weight,
            linear_bias=self.linear_bias,
            topk=topk,
            topp=topp,
            max_out_len=max_out_len,
            head_num=self.head_num,
            size_per_head=self.size_per_head,
            num_layer=self.num_layer,
            bos_id=bos_token_id,
            eos_id=eos_token_id,
            temperature=temperature,
            rotary_embedding_dim=self.rotary_embedding_dim,
            repetition_penalty=repetition_penalty,
            min_length=min_length,
            use_fp16_decoding=self.use_fp16_decoding,
        )

        output_ids = output_ids[paddle.shape(input_ids)[-1] :, :]
        if forced_eos_token_id is not None:
            output_ids[:, -1] = forced_eos_token_id
        return output_ids


class InferPegasusDecoding(nn.Layer):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False, hidden_act="gelu"):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FastGeneration" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FastGeneration"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load("FastGeneration", verbose=True)

        super(InferPegasusDecoding, self).__init__()
        self._hidden_act = hidden_act
        self._num_decoder_layers = model.pegasus.config["num_decoder_layers"]
        self._n_head = model.pegasus.config["decoder_attention_heads"]
        self._d_model = model.pegasus.config["d_model"]

        params = convert_params(self, model.decoder.decoder, fuse_qkv=2, use_fp16=use_fp16_decoding, restore_data=True)

        self.decoder_ln_weight = [
            transfer_param(
                model.decoder.decoder_layernorm.weight,
                is_bias=False,
                dtype="float16" if use_fp16_decoding else "float32",
                restore_data=True,
            )
        ]
        self.decoder_ln_bias = [
            transfer_param(
                model.decoder.decoder_layernorm.bias,
                is_bias=True,
                dtype="float16" if use_fp16_decoding else "float32",
                restore_data=True,
            )
        ]

        self.pos_emb = [
            transfer_param(
                model.decoder.decoder_embed_positions.weight,
                is_bias=False,
                dtype="float16" if use_fp16_decoding else "float32",
                restore_data=True,
            )
        ]
        self.word_emb = [
            transfer_param(
                model.decoder.embed_tokens.weight,
                is_bias=False,
                dtype="float16" if use_fp16_decoding else "float32",
                restore_data=True,
            )
        ]
        setattr(
            self,
            "lm_head_weight_",
            transfer_param(
                model.lm_head_weight.t(),
                is_bias=False,
                dtype="float16" if use_fp16_decoding else "float32",
                restore_data=True,
            ),
        )
        self.linear_weight = [getattr(self, "lm_head_weight_")]
        self.linear_bias = [
            transfer_param(
                model.final_logits_bias,
                is_bias=True,
                dtype="float16" if use_fp16_decoding else "float32",
                restore_data=True,
            )
        ]
        for k, v in params.items():
            setattr(self, k, v)

    def forward(
        self,
        enc_output,
        memory_seq_lens,
        beam_size=4,
        top_k=1,
        top_p=0.0,
        decoding_strategy="beam_search_v3",
        max_out_len=256,
        min_out_len=256,
        diversity_rate=0.0,
        rel_len=False,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        alpha=0.6,
        temperature=1.0,
        early_stopping=False,
        forced_eos_token_id=None,
    ):
        # Beam_search/beam_search_v2/beam_search_v3 should be corrected to beam_search_v3.
        if decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v3"
        elif decoding_strategy == "greedy_search":
            decoding_strategy = "topk_sampling"
            top_k = 1
            top_p = 0.0
        elif decoding_strategy in ["sampling", "topk_sampling", "topp_sampling"]:
            if top_p == 1 and top_k > 0:
                decoding_strategy = "topk_sampling"
                top_p = 0.0
            elif top_p > 0 and top_k == 0:
                decoding_strategy = "topp_sampling"
            else:
                raise AttributeError(
                    "Only topk sampling or topp sampling are supported. "
                    "Topk sampling and topp sampling cannot be both applied in the fast version. "
                )
        output_ids, parent_ids, sequence_length = infer_pegasus_decoding(
            [enc_output],
            [memory_seq_lens],
            self.word_emb,
            self.slf_ln_weight,
            self.slf_ln_bias,
            self.slf_q_weight,
            self.slf_q_bias,
            self.slf_k_weight,
            self.slf_k_bias,
            self.slf_v_weight,
            self.slf_v_bias,
            self.slf_out_weight,
            self.slf_out_bias,
            self.cross_ln_weight,
            self.cross_ln_bias,
            self.cross_q_weight,
            self.cross_q_bias,
            self.cross_k_weight,
            self.cross_k_bias,
            self.cross_v_weight,
            self.cross_v_bias,
            self.cross_out_weight,
            self.cross_out_bias,
            self.ffn_ln_weight,
            self.ffn_ln_bias,
            self.ffn_inter_weight,
            self.ffn_inter_bias,
            self.ffn_out_weight,
            self.ffn_out_bias,
            self.decoder_ln_weight,
            self.decoder_ln_bias,
            self.linear_weight,
            self.linear_bias,
            self.pos_emb,
            decoding_strategy,
            beam_size,
            top_k,
            top_p,
            self._n_head,
            int(self._d_model / self._n_head),
            self._num_decoder_layers,
            bos_token_id,
            eos_token_id,
            max_out_len,
            min_out_len,
            diversity_rate,
            rel_len,
            alpha,
            temperature,
            early_stopping,
            self._hidden_act,
        )

        ids = finalize(
            beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            forced_eos_token_id=forced_eos_token_id,
            decoding_strategy=decoding_strategy,
        )
        return ids


class InferT5Decoding(InferBase):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):

        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FastGeneration" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(decoding_lib)
                LOADED_EXT["FastGeneration"] = ops
        else:
            if decoding_lib is not None:
                logger.warning("The specified decoding_lib does not exist, and it will be built automatically.")
            load("FastGeneration", verbose=True)

        super(InferT5Decoding, self).__init__(use_fp16_decoding)
        for arg, value in locals().items():
            if arg not in ["self", "model"]:
                setattr(self, "_" + arg, value)

        self._num_decoder_layers = model.config.num_decoder_layers
        self._n_head = model.config.num_heads
        self._d_model = model.config.d_model
        self._relative_attention_num_buckets = model.config.relative_attention_num_buckets
        self.tie_word_embeddings = model.config.tie_word_embeddings
        self.act = model.config.feed_forward_proj

        if "gelu" in self.act:
            self.act = "gelu"
        elif "relu" in self.act:
            self.act = "relu"
        else:
            raise ValueError("Only gelu and relu are available in Faster. ")

        # NOTE: using config when support.
        self._max_distance = 128

        params = convert_params(self, model.t5.decoder, fuse_qkv=2, use_fp16=use_fp16_decoding, restore_data=True)

        self.decoder_ln_weight = [
            transfer_param(
                model.t5.decoder.final_layer_norm.weight,
                is_bias=False,
                dtype="float16" if use_fp16_decoding else "float32",
                restore_data=True,
            )
        ]

        self.word_emb = [
            transfer_param(
                model.t5.decoder.embed_tokens.weight,
                is_bias=False,
                dtype="float16" if use_fp16_decoding else "float32",
                restore_data=True,
            )
        ]

        if self.tie_word_embeddings:
            setattr(
                self,
                "lm_head_weight_",
                transfer_param(
                    model.t5.decoder.embed_tokens.weight.t(),
                    is_bias=False,
                    dtype="float16" if use_fp16_decoding else "float32",
                    restore_data=True,
                ),
            )
        else:
            setattr(
                self,
                "lm_head_weight_",
                transfer_param(
                    paddle.assign(model.lm_head.weight),
                    is_bias=False,
                    dtype="float16" if use_fp16_decoding else "float32",
                    restore_data=True,
                ),
            )

        self.linear_weight = [getattr(self, "lm_head_weight_")]
        self.linear_bias = self.default_bias(self.linear_weight, 1)

        setattr(
            self,
            "relative_attn_bias_w",
            transfer_param(
                model.t5.decoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight,
                is_bias=False,
                dtype="float16" if use_fp16_decoding else "float32",
                restore_data=True,
            ),
        )
        self.relative_attention_bias_weight = [getattr(self, "relative_attn_bias_w")]
        for k, v in params.items():
            setattr(self, k, v)

        self.zeros_t = paddle.zeros(shape=[1, 1], dtype="float16" if use_fp16_decoding else "float32")
        if getattr(self, "slf_k_weight", None) is None:
            self.slf_k_weight = [self.zeros_t] * model.t5.config["num_decoder_layers"]
        if getattr(self, "slf_v_weight", None) is None:
            self.slf_v_weight = [self.zeros_t] * model.t5.config["num_decoder_layers"]

    def forward(
        self,
        enc_output,
        memory_seq_lens,
        beam_size=4,
        top_k=1,
        top_p=0.0,
        decoding_strategy="beam_search_v3",
        max_out_len=256,
        diversity_rate=0.0,
        rel_len=False,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        alpha=0.6,
        temperature=1.0,
        early_stopping=False,
    ):
        # Beam_search/beam_search_v2/beam_search_v3 should be corrected to beam_search_v3.
        if decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v3"
        elif decoding_strategy == "greedy_search":
            decoding_strategy = "topk_sampling"
            top_k = 1
            top_p = 0.0
        elif decoding_strategy in ["sampling", "topk_sampling", "topp_sampling"]:
            if top_p == 1 and top_k > 0:
                decoding_strategy = "topk_sampling"
                top_p = 0.0
            elif top_p > 0 and top_k == 0:
                decoding_strategy = "topp_sampling"
            else:
                raise AttributeError(
                    "Only topk sampling or topp sampling are supported. "
                    "Topk sampling and topp sampling cannot be both applied in the fast version. "
                )

        output_ids, parent_ids, sequence_length = infer_t5_decoding(
            enc_output=[enc_output],
            memory_seq_lens=[memory_seq_lens],
            word_emb=self.word_emb,
            slf_ln_weight=self.slf_ln_weight,
            slf_ln_bias=getattr(self, "slf_ln_bias", self.default_bias(self.slf_ln_weight, 0, True)),
            slf_q_weight=self.slf_q_weight,
            slf_q_bias=getattr(self, "slf_q_bias", self.default_bias(self.slf_q_weight, 1)),
            slf_k_weight=self.slf_k_weight,
            slf_k_bias=getattr(self, "slf_k_bias", self.default_bias(self.slf_k_weight, 1)),
            slf_v_weight=self.slf_v_weight,
            slf_v_bias=getattr(self, "slf_v_bias", self.default_bias(self.slf_v_weight, 1)),
            slf_out_weight=self.slf_out_weight,
            slf_out_bias=getattr(self, "slf_out_bias", self.default_bias(self.slf_out_weight, 1)),
            relative_attention_bias_weight=self.relative_attention_bias_weight,
            cross_ln_weight=self.cross_ln_weight,
            cross_ln_bias=getattr(self, "cross_ln_bias", self.default_bias(self.cross_ln_weight, 0, True)),
            cross_q_weight=self.cross_q_weight,
            cross_q_bias=getattr(self, "cross_q_bias", self.default_bias(self.cross_q_weight, 1)),
            cross_k_weight=self.cross_k_weight,
            cross_k_bias=getattr(self, "cross_k_bias", self.default_bias(self.cross_k_weight, 1)),
            cross_v_weight=self.cross_v_weight,
            cross_v_bias=getattr(self, "cross_v_bias", self.default_bias(self.cross_v_weight, 1)),
            cross_out_weight=self.cross_out_weight,
            cross_out_bias=getattr(self, "cross_out_bias", self.default_bias(self.cross_out_weight, 1)),
            ffn_ln_weight=self.ffn_ln_weight,
            ffn_ln_bias=getattr(self, "ffn_ln_bias", self.default_bias(self.ffn_ln_weight, 0, True)),
            ffn_inter_weight_0=self.ffn_inter_weight_0,
            ffn_inter_bias_0=getattr(self, "ffn_inter_bias_0", self.default_bias(self.ffn_inter_weight_0, 1)),
            ffn_inter_weight_1=getattr(
                self, "ffn_inter_weight_1", self.default_bias(self.ffn_inter_weight_0, 1, True)
            ),
            ffn_inter_bias_1=getattr(self, "ffn_inter_bias_1", self.default_bias(self.ffn_inter_weight_1, 1))
            if hasattr(self, "ffn_inter_weight_1")
            else getattr(self, "ffn_inter_bias_1", self.default_bias(self.ffn_inter_weight_0, 1, True)),
            ffn_out_weight=self.ffn_out_weight,
            ffn_out_bias=getattr(self, "ffn_out_bias", self.default_bias(self.ffn_out_weight, 1)),
            decoder_ln_weight=self.decoder_ln_weight,
            decoder_ln_bias=getattr(self, "decoder_ln_bias", self.default_bias(self.decoder_ln_weight, 0, True)),
            linear_weight=self.linear_weight,
            linear_bias=getattr(self, "linear_bias", self.default_bias(self.linear_weight, 1)),
            decoding_strategy=decoding_strategy,
            beam_size=beam_size,
            top_k=top_k,
            top_p=top_p,
            head_num=self._n_head,
            size_per_head=int(self._d_model / self._n_head),
            num_decoder_layers=self._num_decoder_layers,
            start_id=bos_token_id,
            end_id=eos_token_id,
            max_out_len=max_out_len,
            diversity_rate=-diversity_rate,
            rel_len=rel_len,
            alpha=alpha,
            temperature=temperature,
            early_stopping=early_stopping,
            max_distance=self._max_distance,
            relative_attention_num_buckets=self._relative_attention_num_buckets,
            tie_word_embeddings=self.tie_word_embeddings,
            act=self.act,
        )

        ids = finalize(beam_size, output_ids, parent_ids, sequence_length, decoding_strategy=decoding_strategy)

        return ids
