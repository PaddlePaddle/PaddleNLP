import os
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.fluid.layer_helper import LayerHelper


def infer_transformer_decoder(
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
        'SelfLayernormWeight': slf_ln_weight,
        'SelfLayernormBias': slf_ln_bias,
        'SelfQueryWeight': slf_q_weight,
        'SelfQueryBias': slf_q_bias,
        'SelfKeyWeight': slf_k_weight,
        'SelfKeyBias': slf_k_bias,
        'SelfValueWeight': slf_v_weight,
        'SelfValueBias': slf_v_bias,
        'SelfOutWeight': slf_out_weight,
        'SelfOutBias': slf_out_bias,
        'CrossLayernormWeight': cross_ln_weight,
        'CrossLayernormBias': cross_ln_bias,
        'CrossQueryWeight': cross_q_weight,
        'CrossQueryBias': cross_q_bias,
        'CrossKeyWeight': cross_k_weight,
        'CrossKeyBias': cross_k_bias,
        'CrossValueWeight': cross_v_weight,
        'CrossValueBias': cross_v_bias,
        'CrossOutWeight': cross_out_weight,
        'CrossOutBias': cross_out_bias,
        'FFNLayernormWeight': ffn_ln_weight,
        'FFNLayernormBias': ffn_ln_bias,
        'FFNInterWeight': ffn_inter_weight,
        'FFNInterBias': ffn_inter_bias,
        'FFNOutWeight': ffn_out_weight,
        'FFNOutBias': ffn_out_bias,
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

    output_ids = helper.create_variable_for_type_inference("int32")
    parent_ids = helper.create_variable_for_type_inference("int32")
    sequence_length = helper.create_variable_for_type_inference("int32")

    outputs = {
        'OutputIds': output_ids,
        'ParentIds': parent_ids,
        'SequenceLength': sequence_length
    }

    helper.append_op(
        type='fusion_decoding', inputs=inputs, outputs=outputs, attrs=attrs)

    return output_ids, parent_ids, sequence_length


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


def transfer_param(p, is_bias=False):
    param_shape = p.shape
    del p
    return paddle.create_parameter(
        shape=param_shape, dtype="float16", is_bias=is_bias)


class InferTransformerDecoding(nn.Layer):
    def __init__(self,
                 decoder,
                 word_embedding,
                 positional_embedding,
                 linear,
                 max_length,
                 n_layer,
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
        if decoding_lib is None:
            raise ValueError(
                "The args decoding_lib must be set to use Faster Transformer. ")
        elif not os.path.exists(decoding_lib):
            raise ValueError("The path to decoding lib is not exist.")

        super(InferTransformerDecoding, self).__init__()
        paddle.utils.load_op_library(decoding_lib)
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

        output_ids, parent_ids, sequence_length = infer_transformer_decoder(
            [enc_output], [memory_seq_lens],
            [paddle.cast(
                self.word_emb[0], dtype="float16")]
            if self._use_fp16_decoding else self.word_emb,
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
            self.linear_bias, [paddle.cast(
                self.pos_emb[0], dtype="float16")]
            if self._use_fp16_decoding else self.pos_emb,
            self._decoding_strategy,
            self._beam_size,
            self._topk,
            self._topp,
            self._n_head,
            int(self._d_model / self._n_head),
            self._n_layer,
            self._bos_id,
            self._eos_id,
            self._max_out_len,
            self._beam_search_diversity_rate)

        ids = finalize(
            self._beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            decoding_strategy=self._decoding_strategy)

        return ids
