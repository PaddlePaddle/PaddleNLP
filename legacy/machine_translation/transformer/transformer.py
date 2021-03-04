# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers.utils import map_structure

from desc import *

# Set seed for CE or debug
dropout_seed = None


def wrap_layer_with_block(layer, block_idx):
    """
    Make layer define support indicating block, by which we can add layers
    to other blocks within current block. This will make it easy to define
    cache among while loop.
    """

    class BlockGuard(object):
        """
        BlockGuard class.

        BlockGuard class is used to switch to the given block in a program by
        using the Python `with` keyword.
        """

        def __init__(self, block_idx=None, main_program=None):
            self.main_program = fluid.default_main_program(
            ) if main_program is None else main_program
            self.old_block_idx = self.main_program.current_block().idx
            self.new_block_idx = block_idx

        def __enter__(self):
            self.main_program.current_block_idx = self.new_block_idx

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.main_program.current_block_idx = self.old_block_idx
            if exc_type is not None:
                return False  # re-raise exception
            return True

    def layer_wrapper(*args, **kwargs):
        with BlockGuard(block_idx):
            return layer(*args, **kwargs)

    return layer_wrapper


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(np.arange(
        num_timescales)) * -log_timescale_increment
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                               0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype("float32")


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         static_kv=False):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      bias_attr=False,
                      num_flatten_dims=2)
        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        fc_layer = wrap_layer_with_block(
            layers.fc, fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None and static_kv else layers.fc
        k = fc_layer(
            input=keys,
            size=d_key * n_head,
            bias_attr=False,
            num_flatten_dims=2)
        v = fc_layer(
            input=values,
            size=d_value * n_head,
            bias_attr=False,
            num_flatten_dims=2)
        return q, k, v

    def __split_heads_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Reshape input tensors at the last dimension to split multi-heads 
        and then transpose. Specifically, transform the input tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped_q = layers.reshape(
            x=queries, shape=[0, 0, n_head, d_key], inplace=True)
        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        q = layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        reshape_layer = wrap_layer_with_block(
            layers.reshape,
            fluid.default_main_program().current_block(
            ).parent_idx) if cache is not None and static_kv else layers.reshape
        transpose_layer = wrap_layer_with_block(
            layers.transpose,
            fluid.default_main_program().current_block().
            parent_idx) if cache is not None and static_kv else layers.transpose
        reshaped_k = reshape_layer(
            x=keys, shape=[0, 0, n_head, d_key], inplace=True)
        k = transpose_layer(x=reshaped_k, perm=[0, 2, 1, 3])
        reshaped_v = reshape_layer(
            x=values, shape=[0, 0, n_head, d_value], inplace=True)
        v = transpose_layer(x=reshaped_v, perm=[0, 2, 1, 3])

        if cache is not None:  # only for faster inference
            cache_, i = cache
            if static_kv:  # For encoder-decoder attention in inference
                cache_k, cache_v = cache_["static_k"], cache_["static_v"]
                # To init the static_k and static_v in global block.
                static_cache_init = wrap_layer_with_block(
                    layers.assign,
                    fluid.default_main_program().current_block().parent_idx)
                static_cache_init(
                    k,
                    fluid.default_main_program().global_block().var(
                        "static_k_%d" % i))
                static_cache_init(
                    v,
                    fluid.default_main_program().global_block().var(
                        "static_v_%d" % i))
                k, v = cache_k, cache_v
            else:  # For decoder self-attention in inference
                # use cache and concat time steps.
                cache_k, cache_v = cache_["k"], cache_["v"]
                k = layers.concat([cache_k, k], axis=2)
                v = layers.concat([cache_v, v], axis=2)
                cache_["k"], cache_["v"] = (k, v)
        return q, k, v

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        product = layers.matmul(x=q, y=k, transpose_y=True, alpha=d_key**-0.5)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                seed=dropout_seed,
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
    q, k, v = __split_heads_qkv(q, k, v, n_head, d_key, d_value)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_model,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         bias_attr=False,
                         num_flatten_dims=2)
    return proj_out


def positionwise_feed_forward(x, d_inner_hid, d_hid, dropout_rate):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act="relu")
    if dropout_rate:
        hidden = layers.dropout(
            hidden, dropout_prob=dropout_rate, seed=dropout_seed, is_test=False)
    out = layers.fc(input=hidden, size=d_hid, num_flatten_dims=2)
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0.):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.initializer.Constant(1.),
                bias_attr=fluid.initializer.Constant(0.))
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    seed=dropout_seed,
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def prepare_encoder_decoder(src_word,
                            src_pos,
                            src_vocab_size,
                            src_emb_dim,
                            src_max_len,
                            dropout_rate=0.,
                            bos_idx=0,
                            word_emb_param_name=None,
                            pos_enc_param_name=None):
    """Add word embeddings and position encodings.
    The output tensor has a shape of:
    [batch_size, max_src_length_in_batch, d_model].
    This module is used at the bottom of the encoder stacks.
    """
    src_word_emb = fluid.embedding(
        src_word,
        size=[src_vocab_size, src_emb_dim],
        padding_idx=bos_idx,  # set embedding of bos to 0
        param_attr=fluid.ParamAttr(
            name=word_emb_param_name,
            initializer=fluid.initializer.Normal(0., src_emb_dim**-0.5)))

    src_word_emb = layers.scale(x=src_word_emb, scale=src_emb_dim**0.5)
    src_pos_enc = fluid.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        param_attr=fluid.ParamAttr(
            name=pos_enc_param_name, trainable=False))
    src_pos_enc.stop_gradient = True
    enc_input = src_word_emb + src_pos_enc
    return layers.dropout(
        enc_input, dropout_prob=dropout_rate, seed=dropout_seed,
        is_test=False) if dropout_rate else enc_input


prepare_encoder = partial(
    prepare_encoder_decoder, pos_enc_param_name=pos_enc_param_names[0])
prepare_decoder = partial(
    prepare_encoder_decoder, pos_enc_param_name=pos_enc_param_names[1])


def encoder_layer(enc_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  preprocess_cmd="n",
                  postprocess_cmd="da"):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    attn_output = multi_head_attention(
        pre_process_layer(enc_input, preprocess_cmd,
                          prepostprocess_dropout), None, None, attn_bias, d_key,
        d_value, d_model, n_head, attention_dropout)
    attn_output = post_process_layer(enc_input, attn_output, postprocess_cmd,
                                     prepostprocess_dropout)
    ffd_output = positionwise_feed_forward(
        pre_process_layer(attn_output, preprocess_cmd, prepostprocess_dropout),
        d_inner_hid, d_model, relu_dropout)
    return post_process_layer(attn_output, ffd_output, postprocess_cmd,
                              prepostprocess_dropout)


def encoder(enc_input,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd="n",
            postprocess_cmd="da"):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    for i in range(n_layer):
        enc_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd, )
        enc_input = enc_output
    enc_output = pre_process_layer(enc_output, preprocess_cmd,
                                   prepostprocess_dropout)
    return enc_output


def decoder_layer(dec_input,
                  enc_output,
                  slf_attn_bias,
                  dec_enc_attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  preprocess_cmd,
                  postprocess_cmd,
                  cache=None):
    """ The layer to be stacked in decoder part.
    The structure of this module is similar to that in the encoder part except
    a multi-head attention is added to implement encoder-decoder attention.
    """
    slf_attn_output = multi_head_attention(
        pre_process_layer(dec_input, preprocess_cmd, prepostprocess_dropout),
        None,
        None,
        slf_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        cache=cache)
    slf_attn_output = post_process_layer(
        dec_input,
        slf_attn_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    enc_attn_output = multi_head_attention(
        pre_process_layer(slf_attn_output, preprocess_cmd,
                          prepostprocess_dropout),
        enc_output,
        enc_output,
        dec_enc_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        cache=cache,
        static_kv=True)
    enc_attn_output = post_process_layer(
        slf_attn_output,
        enc_attn_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    ffd_output = positionwise_feed_forward(
        pre_process_layer(enc_attn_output, preprocess_cmd,
                          prepostprocess_dropout),
        d_inner_hid,
        d_model,
        relu_dropout, )
    dec_output = post_process_layer(
        enc_attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    return dec_output


def decoder(dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
            caches=None):
    """
    The decoder is composed of a stack of identical decoder_layer layers.
    """
    for i in range(n_layer):
        dec_output = decoder_layer(
            dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
            cache=None if caches is None else (caches[i], i))
        dec_input = dec_output
    dec_output = pre_process_layer(dec_output, preprocess_cmd,
                                   prepostprocess_dropout)
    return dec_output


def transformer(model_input,
                src_vocab_size,
                trg_vocab_size,
                max_length,
                n_layer,
                n_head,
                d_key,
                d_value,
                d_model,
                d_inner_hid,
                prepostprocess_dropout,
                attention_dropout,
                relu_dropout,
                preprocess_cmd,
                postprocess_cmd,
                weight_sharing,
                label_smooth_eps,
                bos_idx=0,
                is_test=False):
    if weight_sharing:
        assert src_vocab_size == trg_vocab_size, (
            "Vocabularies in source and target should be same for weight sharing."
        )

    enc_inputs = (model_input.src_word, model_input.src_pos,
                  model_input.src_slf_attn_bias)
    dec_inputs = (model_input.trg_word, model_input.trg_pos,
                  model_input.trg_slf_attn_bias, model_input.trg_src_attn_bias)
    label = model_input.lbl_word
    weights = model_input.lbl_weight

    enc_output = wrap_encoder(
        enc_inputs,
        src_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        weight_sharing,
        bos_idx=bos_idx)

    predict = wrap_decoder(
        dec_inputs,
        trg_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        weight_sharing,
        enc_output=enc_output)

    # Padding index do not contribute to the total loss. The weights is used to
    # cancel padding index in calculating the loss.
    if label_smooth_eps:
        # TODO: use fluid.input.one_hot after softmax_with_cross_entropy removing
        # the enforcement that the last dimension of label must be 1.
        label = layers.label_smooth(
            label=layers.one_hot(
                input=label, depth=trg_vocab_size),
            epsilon=label_smooth_eps)

    cost = layers.softmax_with_cross_entropy(
        logits=predict,
        label=label,
        soft_label=True if label_smooth_eps else False)
    weighted_cost = layers.elementwise_mul(x=cost, y=weights, axis=0)
    sum_cost = layers.reduce_sum(weighted_cost)
    token_num = layers.reduce_sum(weights)
    token_num.stop_gradient = True
    avg_cost = sum_cost / token_num
    return sum_cost, avg_cost, predict, token_num


def wrap_encoder(enc_inputs,
                 src_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 bos_idx=0):
    """
    The wrapper assembles together all needed layers for the encoder.
    """
    src_word, src_pos, src_slf_attn_bias = enc_inputs
    enc_input = prepare_encoder(
        src_word,
        src_pos,
        src_vocab_size,
        d_model,
        max_length,
        prepostprocess_dropout,
        bos_idx=bos_idx,
        word_emb_param_name=word_emb_param_names[0])
    enc_output = encoder(
        enc_input,
        src_slf_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd, )
    return enc_output


def wrap_decoder(dec_inputs,
                 trg_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 enc_output=None,
                 caches=None,
                 bos_idx=0):
    """
    The wrapper assembles together all needed layers for the decoder.
    """
    trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias = dec_inputs

    dec_input = prepare_decoder(
        trg_word,
        trg_pos,
        trg_vocab_size,
        d_model,
        max_length,
        prepostprocess_dropout,
        bos_idx=bos_idx,
        word_emb_param_name=word_emb_param_names[0]
        if weight_sharing else word_emb_param_names[1])
    dec_output = decoder(
        dec_input,
        enc_output,
        trg_slf_attn_bias,
        trg_src_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        caches=caches)
    # Reshape to 2D tensor to use GEMM instead of BatchedGEMM
    dec_output = layers.reshape(
        dec_output, shape=[-1, dec_output.shape[-1]], inplace=True)
    if weight_sharing:
        predict = layers.matmul(
            x=dec_output,
            y=fluid.default_main_program().global_block().var(
                word_emb_param_names[0]),
            transpose_y=True)
    else:
        predict = layers.fc(input=dec_output,
                            size=trg_vocab_size,
                            bias_attr=False)
    if dec_inputs is None:
        # Return probs for independent decoder program.
        predict = layers.softmax(predict)
    return predict


def fast_decode(model_input, src_vocab_size, trg_vocab_size, max_in_len,
                n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
                prepostprocess_dropout, attention_dropout, relu_dropout,
                preprocess_cmd, postprocess_cmd, weight_sharing, beam_size,
                max_out_len, bos_idx, eos_idx):
    """
    Use beam search to decode. Caches will be used to store states of history
    steps which can make the decoding faster.
    """
    enc_inputs = (model_input.src_word, model_input.src_pos,
                  model_input.src_slf_attn_bias)
    dec_inputs = (model_input.trg_word, model_input.init_score,
                  model_input.init_idx, model_input.trg_src_attn_bias)

    enc_output = wrap_encoder(
        enc_inputs,
        src_vocab_size,
        max_in_len,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        weight_sharing,
        bos_idx=bos_idx)
    start_tokens, init_scores, parent_idx, trg_src_attn_bias = dec_inputs

    def beam_search():
        max_len = layers.fill_constant(
            shape=[1],
            dtype=start_tokens.dtype,
            value=max_out_len,
            force_cpu=True)
        step_idx = layers.fill_constant(
            shape=[1], dtype=start_tokens.dtype, value=0, force_cpu=True)
        # array states will be stored for each step.
        ids = layers.array_write(
            layers.reshape(start_tokens, (-1, 1)), step_idx)
        scores = layers.array_write(init_scores, step_idx)
        # cell states will be overwrited at each step.
        # caches contains states of history steps in decoder self-attention
        # and static encoder output projections in encoder-decoder attention
        # to reduce redundant computation.
        batch_size = layers.shape(start_tokens)[0]
        caches = [
            {
                "k":  # for self attention
                layers.fill_constant(
                    shape=[batch_size, n_head, 0, d_key],
                    dtype=enc_output.dtype,
                    value=0),
                "v":  # for self attention
                layers.fill_constant(
                    shape=[batch_size, n_head, 0, d_value],
                    dtype=enc_output.dtype,
                    value=0),
                "static_k":  # for encoder-decoder attention
                fluid.data(
                    shape=[None, n_head, 0, d_key],
                    dtype=enc_output.dtype,
                    name=("static_k_%d" % i)),
                "static_v":  # for encoder-decoder attention
                fluid.data(
                    shape=[None, n_head, 0, d_value],
                    dtype=enc_output.dtype,
                    name=("static_v_%d" % i)),
            } for i in range(n_layer)
        ]

        def cond_func(step_idx, selected_ids, selected_scores, gather_idx,
                      caches, trg_src_attn_bias):
            length_cond = layers.less_than(x=step_idx, y=max_len)
            finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
            return layers.logical_and(x=length_cond, y=finish_cond)

        def body_func(step_idx, pre_ids, pre_scores, gather_idx, caches,
                      trg_src_attn_bias):
            # gather cell states corresponding to selected parent
            pre_caches = map_structure(
                lambda x: layers.gather(x, index=gather_idx), caches)
            pre_src_attn_bias = layers.gather(
                trg_src_attn_bias, index=gather_idx)
            bias_batch_size = layers.shape(pre_src_attn_bias)[0]
            pre_pos = layers.elementwise_mul(
                x=layers.fill_constant(
                    value=1, shape=[bias_batch_size, 1], dtype=pre_ids.dtype),
                y=step_idx,
                axis=0)
            logits = wrap_decoder(
                (pre_ids, pre_pos, None, pre_src_attn_bias),
                trg_vocab_size,
                max_in_len,
                n_layer,
                n_head,
                d_key,
                d_value,
                d_model,
                d_inner_hid,
                prepostprocess_dropout,
                attention_dropout,
                relu_dropout,
                preprocess_cmd,
                postprocess_cmd,
                weight_sharing,
                enc_output=enc_output,
                caches=pre_caches,
                bos_idx=bos_idx)
            # intra-beam topK
            topk_scores, topk_indices = layers.topk(
                input=layers.softmax(logits), k=beam_size)
            accu_scores = layers.elementwise_add(
                x=layers.log(topk_scores), y=pre_scores, axis=0)
            # beam_search op uses lod to differentiate branches.
            accu_scores = layers.lod_reset(accu_scores, pre_ids)
            # topK reduction across beams, also contain special handle of
            # end beams and end sentences(batch reduction)
            selected_ids, selected_scores, gather_idx = layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=beam_size,
                end_id=eos_idx,
                return_parent_idx=True)
            step_idx = layers.increment(x=step_idx, value=1.0, in_place=False)
            layers.array_write(selected_ids, i=step_idx, array=ids)
            layers.array_write(selected_scores, i=step_idx, array=scores)
            return (step_idx, selected_ids, selected_scores, gather_idx,
                    pre_caches, pre_src_attn_bias)

        _ = layers.while_loop(
            cond=cond_func,
            body=body_func,
            loop_vars=[
                step_idx, start_tokens, init_scores, parent_idx, caches,
                trg_src_attn_bias
            ],
            is_test=True)

        finished_ids, finished_scores = layers.beam_search_decode(
            ids, scores, beam_size=beam_size, end_id=eos_idx)
        return finished_ids, finished_scores

    finished_ids, finished_scores = beam_search()
    return finished_ids, finished_scores


def create_net(is_training, model_input, args):
    if is_training:
        sum_cost, avg_cost, _, token_num = transformer(
            model_input, args.src_vocab_size, args.trg_vocab_size,
            args.max_length + 1, args.n_layer, args.n_head, args.d_key,
            args.d_value, args.d_model, args.d_inner_hid,
            args.prepostprocess_dropout, args.attention_dropout,
            args.relu_dropout, args.preprocess_cmd, args.postprocess_cmd,
            args.weight_sharing, args.label_smooth_eps, args.bos_idx)
        return sum_cost, avg_cost, token_num
    else:
        out_ids, out_scores = fast_decode(
            model_input, args.src_vocab_size, args.trg_vocab_size,
            args.max_length + 1, args.n_layer, args.n_head, args.d_key,
            args.d_value, args.d_model, args.d_inner_hid,
            args.prepostprocess_dropout, args.attention_dropout,
            args.relu_dropout, args.preprocess_cmd, args.postprocess_cmd,
            args.weight_sharing, args.beam_size, args.max_out_len, args.bos_idx,
            args.eos_idx)
        return out_ids, out_scores
