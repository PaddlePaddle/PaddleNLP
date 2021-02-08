#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import re
import numpy as np
import paddle.fluid as fluid


def log_softmax(logits, axis=-1):
    logsoftmax = logits - fluid.layers.log(
        fluid.layers.reduce_sum(fluid.layers.exp(logits), axis))
    return logsoftmax


def einsum4x4(equation, x, y):
    idx_x, idx_y, idx_z = re.split(",|->", equation)
    repeated_idx = list(set(idx_x + idx_y) - set(idx_z))

    unique_idx_x = list(set(idx_x) - set(idx_y))
    unique_idx_y = list(set(idx_y) - set(idx_x))
    common_idx = list(set(idx_x) & set(idx_y) - set(repeated_idx))

    new_idx_x = common_idx + unique_idx_x + repeated_idx
    new_idx_y = common_idx + unique_idx_y + repeated_idx
    new_idx_z = common_idx + unique_idx_x + unique_idx_y

    perm_x = [idx_x.index(i) for i in new_idx_x]
    perm_y = [idx_y.index(i) for i in new_idx_y]
    perm_z = [new_idx_z.index(i) for i in idx_z]

    x = fluid.layers.transpose(x, perm=perm_x)
    y = fluid.layers.transpose(y, perm=perm_y)
    z = fluid.layers.matmul(x=x, y=y, transpose_y=True)
    z = fluid.layers.transpose(z, perm=perm_z)
    return z


def positional_embedding(pos_seq, inv_freq, bsz=None):
    pos_seq = fluid.layers.reshape(pos_seq, [-1, 1])
    inv_freq = fluid.layers.reshape(inv_freq, [1, -1])
    sinusoid_inp = fluid.layers.matmul(pos_seq, inv_freq)
    pos_emb = fluid.layers.concat(
        input=[fluid.layers.sin(sinusoid_inp), fluid.layers.cos(sinusoid_inp)],
        axis=-1)
    pos_emb = fluid.layers.unsqueeze(pos_emb, [1])
    if bsz is not None:
        pos_emb = fluid.layers.expand(pos_emb, [1, bsz, 1])

    return pos_emb


def positionwise_ffn(inp,
                     d_model,
                     d_inner,
                     dropout_prob,
                     param_initializer=None,
                     act_type='relu',
                     name='ff'):
    """Position-wise Feed-forward Network."""
    if act_type not in ['relu', 'gelu']:
        raise ValueError('Unsupported activation type {}'.format(act_type))

    output = fluid.layers.fc(input=inp,
                             size=d_inner,
                             act=act_type,
                             num_flatten_dims=2,
                             param_attr=fluid.ParamAttr(
                                 name=name + '_layer_1_weight',
                                 initializer=param_initializer),
                             bias_attr=name + '_layer_1_bias')
    output = fluid.layers.dropout(
        output,
        dropout_prob=dropout_prob,
        dropout_implementation="upscale_in_train",
        is_test=False)
    output = fluid.layers.fc(output,
                             size=d_model,
                             num_flatten_dims=2,
                             param_attr=fluid.ParamAttr(
                                 name=name + '_layer_2_weight',
                                 initializer=param_initializer),
                             bias_attr=name + '_layer_2_bias')
    output = fluid.layers.dropout(
        output,
        dropout_prob=dropout_prob,
        dropout_implementation="upscale_in_train",
        is_test=False)
    output = fluid.layers.layer_norm(
        output + inp,
        begin_norm_axis=len(output.shape) - 1,
        epsilon=1e-12,
        param_attr=fluid.ParamAttr(
            name=name + '_layer_norm_scale',
            initializer=fluid.initializer.Constant(1.)),
        bias_attr=fluid.ParamAttr(
            name + '_layer_norm_bias',
            initializer=fluid.initializer.Constant(0.)))
    return output


def head_projection(h, d_model, n_head, d_head, param_initializer, name=''):
    """Project hidden states to a specific head with a 4D-shape."""
    proj_weight = fluid.layers.create_parameter(
        shape=[d_model, n_head, d_head],
        dtype=h.dtype,
        attr=fluid.ParamAttr(
            name=name + '_weight', initializer=param_initializer),
        is_bias=False)
    # ibh,hnd->ibnd 
    head = fluid.layers.mul(x=h,
                            y=proj_weight,
                            x_num_col_dims=2,
                            y_num_col_dims=1)
    return head


def post_attention(h,
                   attn_vec,
                   d_model,
                   n_head,
                   d_head,
                   dropout,
                   param_initializer,
                   residual=True,
                   name=''):
    """Post-attention processing."""
    # post-attention projection (back to `d_model`)
    proj_o = fluid.layers.create_parameter(
        shape=[d_model, n_head, d_head],
        dtype=h.dtype,
        attr=fluid.ParamAttr(
            name=name + '_o_weight', initializer=param_initializer),
        is_bias=False)
    # ibnd,hnd->ibh
    proj_o = fluid.layers.transpose(proj_o, perm=[1, 2, 0])
    attn_out = fluid.layers.mul(x=attn_vec,
                                y=proj_o,
                                x_num_col_dims=2,
                                y_num_col_dims=2)

    attn_out = fluid.layers.dropout(
        attn_out,
        dropout_prob=dropout,
        dropout_implementation="upscale_in_train",
        is_test=False)

    if residual:
        output = fluid.layers.layer_norm(
            attn_out + h,
            begin_norm_axis=len(attn_out.shape) - 1,
            epsilon=1e-12,
            param_attr=fluid.ParamAttr(
                name=name + '_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name + '_layer_norm_bias',
                initializer=fluid.initializer.Constant(0.)))
    else:
        output = fluid.layers.layer_norm(
            attn_out,
            begin_norm_axis=len(attn_out.shape) - 1,
            epsilon=1e-12,
            param_attr=fluid.ParamAttr(
                name=name + '_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name + '_layer_norm_bias',
                initializer=fluid.initializer.Constant(0.)))

    return output


def abs_attn_core(q_head, k_head, v_head, attn_mask, dropatt, scale):
    """Core absolute positional attention operations."""

    attn_score = einsum4x4('ibnd,jbnd->ijbn', q_head, k_head)

    attn_score *= scale
    if attn_mask is not None:
        attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = fluid.layers.softmax(attn_score, axis=1)
    attn_prob = fluid.layers.dropout(
        attn_prob,
        dropout_prob=dropatt,
        dropout_implementation="upscale_in_train",
        is_test=False)

    # attention output
    attn_vec = einsum4x4('ijbn,jbnd->ibnd', attn_prob, v_head)

    return attn_vec


def rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                  r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt, scale,
                  name):
    """Core relative positional attention operations."""
    ## content based attention score
    ac = einsum4x4('ibnd,jbnd->ijbn',
                   fluid.layers.elementwise_add(q_head, r_w_bias, 2), k_head_h)

    # position based attention score
    bd = einsum4x4('ibnd,jbnd->ijbn',
                   fluid.layers.elementwise_add(q_head, r_r_bias, 2), k_head_r)

    bd = rel_shift(bd, klen=ac.shape[1])

    # segment based attention score
    if seg_mat is None:
        ef = 0
    else:
        seg_embed = fluid.layers.stack([seg_embed] * q_head.shape[0], axis=0)

        ef = einsum4x4('ibnd,isnd->ibns',
                       fluid.layers.elementwise_add(q_head, r_s_bias, 2),
                       seg_embed)
        ef = einsum4x4('ijbs,ibns->ijbn', seg_mat, ef)

    attn_score = (ac + bd + ef) * scale

    if attn_mask is not None:
        # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
        attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = fluid.layers.softmax(attn_score, axis=1)
    attn_prob = fluid.layers.dropout(
        attn_prob, dropatt, dropout_implementation="upscale_in_train")

    # attention output
    attn_vec = einsum4x4('ijbn,jbnd->ibnd', attn_prob, v_head_h)
    return attn_vec


def rel_shift(x, klen=-1):
    """perform relative shift to form the relative attention score."""
    x_size = x.shape
    x = fluid.layers.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
    x = fluid.layers.slice(x, axes=[0], starts=[1], ends=[x_size[1]])
    x = fluid.layers.reshape(x,
                             [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
    x = fluid.layers.slice(x, axes=[1], starts=[0], ends=[klen])

    return x


def _cache_mem(curr_out, prev_mem, mem_len, reuse_len=None):
    """cache hidden states into memory."""
    if mem_len is None or mem_len == 0:
        return None
    else:
        if reuse_len is not None and reuse_len > 0:
            curr_out = curr_out[:reuse_len]

        if prev_mem is None:
            new_mem = curr_out[-mem_len:]
        else:
            new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]

    new_mem.stop_gradient = True
    return new_mem


def relative_positional_encoding(qlen,
                                 klen,
                                 d_model,
                                 clamp_len,
                                 attn_type,
                                 bi_data,
                                 bsz=None,
                                 dtype=None):
    """create relative positional encoding."""
    freq_seq = fluid.layers.range(0, d_model, 2.0, 'float32')
    if dtype is not None and dtype != 'float32':
        freq_seq = tf.cast(freq_seq, dtype=dtype)
    inv_freq = 1 / (10000**(freq_seq / d_model))

    if attn_type == 'bi':
        beg, end = klen, -qlen
    elif attn_type == 'uni':
        beg, end = klen, -1
    else:
        raise ValueError('Unknown `attn_type` {}.'.format(attn_type))

    if bi_data:
        fwd_pos_seq = fluid.layers.range(beg, end, -1.0, 'float32')
        bwd_pos_seq = fluid.layers.range(-beg, -end, 1.0, 'float32')

        if dtype is not None and dtype != 'float32':
            fwd_pos_seq = fluid.layers.cast(fwd_pos_seq, dtype='float32')
            bwd_pos_seq = fluid.layers.cast(bwd_pos_seq, dtype='float32')

        if clamp_len > 0:
            fwd_pos_seq = fluid.layers.clip(fwd_pos_seq, -clamp_len, clamp_len)
            bwd_pos_seq = fluid.layers.clip(bwd_pos_seq, -clamp_len, clamp_len)

        if bsz is not None:
            # With bi_data, the batch size should be divisible by 2.
            assert bsz % 2 == 0
            fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
            bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
        else:
            fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq)

        pos_emb = fluid.layers.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
    else:
        fwd_pos_seq = fluid.layers.range(beg, end, -1.0, 'float32')
        if dtype is not None and dtype != 'float32':
            fwd_pos_seq = fluid.layers.cast(fwd_pos_seq, dtype=dtype)
        if clamp_len > 0:
            fwd_pos_seq = fluid.layers.clip(fwd_pos_seq, -clamp_len, clamp_len)
        pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz)
        fluid.layers.reshape(pos_emb, [2 * qlen, -1, d_model], inplace=True)
    return pos_emb


def multihead_attn(q,
                   k,
                   v,
                   attn_mask,
                   d_model,
                   n_head,
                   d_head,
                   dropout,
                   dropatt,
                   is_training,
                   kernel_initializer,
                   residual=True,
                   scope='abs_attn',
                   reuse=None):
    """Standard multi-head attention with absolute positional embedding."""

    scale = 1 / (d_head**0.5)
    with tf.variable_scope(scope, reuse=reuse):
        # attention heads
        q_head_h = head_projection(
            h, d_model, n_head, d_head, initializer, name=name + '_rel_attn_q')

        q_head = head_projection(q, d_model, n_head, d_head, kernel_initializer,
                                 'q')
        k_head = head_projection(k, d_model, n_head, d_head, kernel_initializer,
                                 'k')
        v_head = head_projection(v, d_model, n_head, d_head, kernel_initializer,
                                 'v')

        # attention vector
        attn_vec = abs_attn_core(q_head, k_head, v_head, attn_mask, dropatt,
                                 is_training, scale)

        # post processing
        output = post_attention(v, attn_vec, d_model, n_head, d_head, dropout,
                                is_training, kernel_initializer, residual)

    return output


def rel_multihead_attn(h,
                       r,
                       r_w_bias,
                       r_r_bias,
                       seg_mat,
                       r_s_bias,
                       seg_embed,
                       attn_mask,
                       mems,
                       d_model,
                       n_head,
                       d_head,
                       dropout,
                       dropatt,
                       initializer,
                       name=''):
    """Multi-head attention with relative positional encoding."""

    scale = 1 / (d_head**0.5)
    if mems is not None and len(mems.shape) > 1:
        cat = fluid.layers.concat([mems, h], 0)
    else:
        cat = h

    # content heads
    q_head_h = head_projection(
        h, d_model, n_head, d_head, initializer, name=name + '_rel_attn_q')
    k_head_h = head_projection(
        cat, d_model, n_head, d_head, initializer, name=name + '_rel_attn_k')
    v_head_h = head_projection(
        cat, d_model, n_head, d_head, initializer, name=name + '_rel_attn_v')

    # positional heads
    k_head_r = head_projection(
        r, d_model, n_head, d_head, initializer, name=name + '_rel_attn_r')

    # core attention ops
    attn_vec = rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_embed,
                             seg_mat, r_w_bias, r_r_bias, r_s_bias, attn_mask,
                             dropatt, scale, name)

    # post processing
    output = post_attention(
        h,
        attn_vec,
        d_model,
        n_head,
        d_head,
        dropout,
        initializer,
        name=name + '_rel_attn')

    return output


def transformer_xl(inp_k,
                   n_token,
                   n_layer,
                   d_model,
                   n_head,
                   d_head,
                   d_inner,
                   dropout,
                   dropatt,
                   attn_type,
                   bi_data,
                   initializer,
                   mem_len=None,
                   inp_q=None,
                   mems=None,
                   same_length=False,
                   clamp_len=-1,
                   untie_r=False,
                   input_mask=None,
                   perm_mask=None,
                   seg_id=None,
                   reuse_len=None,
                   ff_activation='relu',
                   target_mapping=None,
                   use_fp16=False,
                   name='',
                   **kwargs):
    """
    Defines a Transformer-XL computation graph with additional
	support for XLNet.

    Args:
	inp_k: int64 Tensor in shape [len, bsz], the input token IDs.
	seg_id: int64 Tensor in shape [len, bsz], the input segment IDs.
	input_mask: float32 Tensor in shape [len, bsz], the input mask.
	  0 for real tokens and 1 for padding.
	mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
	  from previous batches. The length of the list equals n_layer.
	  If None, no memory is used.
	perm_mask: float32 Tensor in shape [len, len, bsz].
	  If perm_mask[i, j, k] = 0, i attend to j in batch k;
	  if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
	  If None, each position attends to all the others.
	target_mapping: float32 Tensor in shape [num_predict, len, bsz].
	  If target_mapping[i, j, k] = 1, the i-th predict in batch k is
	  on the j-th token.
	  Only used during pretraining for partial prediction.
	  Set to None during finetuning.
	inp_q: float32 Tensor in shape [len, bsz].
	  1 for tokens with losses and 0 for tokens without losses.
	  Only used during pretraining for two-stream attention.
	  Set to None during finetuning.
	n_layer: int, the number of layers.
	d_model: int, the hidden size.
	n_head: int, the number of attention heads.
	d_head: int, the dimension size of each attention head.
	d_inner: int, the hidden size in feed-forward layers.
	ff_activation: str, "relu" or "gelu".
	untie_r: bool, whether to untie the biases in attention.
	n_token: int, the vocab size.
	is_training: bool, whether in training mode.
	use_tpu: bool, whether TPUs are used.
	use_fp16: bool, use bfloat16 instead of float32.
	dropout: float, dropout rate.
	dropatt: float, dropout rate on attention probabilities.
	init: str, the initialization scheme, either "normal" or "uniform".
	init_range: float, initialize the parameters with a uniform distribution
	  in [-init_range, init_range]. Only effective when init="uniform".
	init_std: float, initialize the parameters with a normal distribution
	  with mean 0 and stddev init_std. Only effective when init="normal".
	mem_len: int, the number of tokens to cache.
	reuse_len: int, the number of tokens in the currect batch to be cached
	  and reused in the future.
	bi_data: bool, whether to use bidirectional input pipeline.
	  Usually set to True during pretraining and False during finetuning.
	clamp_len: int, clamp all relative distances larger than clamp_len.
	  -1 means no clamping.
	same_length: bool, whether to use the same attention length for each token.
	summary_type: str, "last", "first", "mean", or "attn". The method
	  to pool the input to get a vector representation.
    """
    print('memory input {}'.format(mems))
    data_type = "float16" if use_fp16 else "float32"
    print('Use float type {}'.format(data_type))

    qlen = inp_k.shape[0]
    mlen = mems[0].shape[0] if mems is not None else 0
    klen = mlen + qlen
    bsz = fluid.layers.slice(
        fluid.layers.shape(inp_k), axes=[0], starts=[1], ends=[2])

    ##### Attention mask
    # causal attention mask
    if attn_type == 'uni':
        attn_mask = fluid.layers.create_global_var(
            name='attn_mask',
            shape=[qlen, klen, 1, 1],
            value=0.0,
            dtype=data_type,
            persistable=True)
    elif attn_type == 'bi':
        attn_mask = None
    else:
        raise ValueError('Unsupported attention type: {}'.format(attn_type))

    # data mask: input mask & perm mask
    if input_mask is not None and perm_mask is not None:
        data_mask = fluid.layers.unsqueeze(input_mask, [0]) + perm_mask
    elif input_mask is not None and perm_mask is None:
        data_mask = fluid.layers.unsqueeze(input_mask, [0])
    elif input_mask is None and perm_mask is not None:
        data_mask = perm_mask
    else:
        data_mask = None

    if data_mask is not None:
        # all mems can be attended to
        mems_mask = fluid.layers.zeros(
            shape=[data_mask.shape[0], mlen, 1], dtype='float32')
        mems_mask = fluid.layers.expand(mems_mask, [1, 1, bsz])
        data_mask = fluid.layers.concat([mems_mask, data_mask], 1)
        if attn_mask is None:
            attn_mask = fluid.layers.unsqueeze(data_mask, [-1])
        else:
            attn_mask += fluid.layers.unsqueeze(data_mask, [-1])
    if attn_mask is not None:
        attn_mask = fluid.layers.cast(attn_mask > 0, dtype=data_type)

    if attn_mask is not None:
        non_tgt_mask = fluid.layers.diag(
            np.array([-1] * qlen).astype(data_type))
        non_tgt_mask = fluid.layers.concat(
            [fluid.layers.zeros(
                [qlen, mlen], dtype=data_type), non_tgt_mask],
            axis=-1)

        attn_mask = fluid.layers.expand(attn_mask, [qlen, 1, 1, 1])
        non_tgt_mask = fluid.layers.unsqueeze(non_tgt_mask, axes=[2, 3])
        non_tgt_mask = fluid.layers.expand(non_tgt_mask, [1, 1, bsz, 1])
        non_tgt_mask = fluid.layers.cast(
            (attn_mask + non_tgt_mask) > 0, dtype=data_type)
        non_tgt_mask.stop_gradient = True
    else:
        non_tgt_mask = None

    if untie_r:
        r_w_bias = fluid.layers.create_parameter(
            shape=[n_layer, n_head, d_head],
            dtype=data_type,
            attr=fluid.ParamAttr(
                name=name + '_r_w_bias', initializer=initializer),
            is_bias=True)
        r_w_bias = [
            fluid.layers.slice(
                r_w_bias, axes=[0], starts=[i], ends=[i + 1])
            for i in range(n_layer)
        ]
        r_w_bias = [
            fluid.layers.squeeze(
                r_w_bias[i], axes=[0]) for i in range(n_layer)
        ]
        r_r_bias = fluid.layers.create_parameter(
            shape=[n_layer, n_head, d_head],
            dtype=data_type,
            attr=fluid.ParamAttr(
                name=name + '_r_r_bias', initializer=initializer),
            is_bias=True)
        r_r_bias = [
            fluid.layers.slice(
                r_r_bias, axes=[0], starts=[i], ends=[i + 1])
            for i in range(n_layer)
        ]
        r_r_bias = [
            fluid.layers.squeeze(
                r_r_bias[i], axes=[0]) for i in range(n_layer)
        ]
    else:
        r_w_bias = fluid.layers.create_parameter(
            shape=[n_head, d_head],
            dtype=data_type,
            attr=fluid.ParamAttr(
                name=name + '_r_w_bias', initializer=initializer),
            is_bias=True)
        r_r_bias = fluid.layers.create_parameter(
            shape=[n_head, d_head],
            dtype=data_type,
            attr=fluid.ParamAttr(
                name=name + '_r_r_bias', initializer=initializer),
            is_bias=True)

    lookup_table = fluid.layers.create_parameter(
        shape=[n_token, d_model],
        dtype=data_type,
        attr=fluid.ParamAttr(
            name=name + '_word_embedding', initializer=initializer),
        is_bias=False)
    word_emb_k = fluid.layers.embedding(
        input=inp_k,
        size=[n_token, d_model],
        dtype=data_type,
        param_attr=fluid.ParamAttr(
            name=name + '_word_embedding', initializer=initializer))

    if inp_q is not None:
        pass

    output_h = fluid.layers.dropout(
        word_emb_k,
        dropout_prob=dropout,
        dropout_implementation="upscale_in_train")

    if inp_q is not None:
        pass

    if seg_id is not None:
        if untie_r:
            r_s_bias = fluid.layers.create_parameter(
                shape=[n_layer, n_head, d_head],
                dtype=data_type,
                attr=fluid.ParamAttr(
                    name=name + '_r_s_bias', initializer=initializer),
                is_bias=True)
            r_s_bias = [
                fluid.layers.slice(
                    r_s_bias, axes=[0], starts=[i], ends=[i + 1])
                for i in range(n_layer)
            ]
            r_s_bias = [
                fluid.layers.squeeze(
                    r_s_bias[i], axes=[0]) for i in range(n_layer)
            ]
        else:
            r_s_bias = fluid.layers.create_parameter(
                shape=[n_head, d_head],
                dtype=data_type,
                attr=fluid.ParamAttr(
                    name=name + '_r_s_bias', initializer=initializer),
                is_bias=True)

        seg_embed = fluid.layers.create_parameter(
            shape=[n_layer, 2, n_head, d_head],
            dtype=data_type,
            attr=fluid.ParamAttr(
                name=name + '_seg_embed', initializer=initializer))
        seg_embed = [
            fluid.layers.slice(
                seg_embed, axes=[0], starts=[i], ends=[i + 1])
            for i in range(n_layer)
        ]
        seg_embed = [
            fluid.layers.squeeze(
                seg_embed[i], axes=[0]) for i in range(n_layer)
        ]

        # COnver `seg_id` to one-hot seg_mat
        # seg_id: [bsz, qlen, 1]
        mem_pad = fluid.layers.fill_constant_batch_size_like(
            input=seg_id, shape=[-1, mlen], value=0, dtype='int64')
        # cat_ids: [bsz, klen, 1]
        cat_ids = fluid.layers.concat(input=[mem_pad, seg_id], axis=1)
        seg_id = fluid.layers.stack([seg_id] * klen, axis=2)
        cat_ids = fluid.layers.stack([cat_ids] * qlen, axis=2)
        cat_ids = fluid.layers.transpose(cat_ids, perm=[0, 2, 1])

        # seg_mat: [bsz, qlen, klen]
        seg_mat = fluid.layers.cast(
            fluid.layers.logical_not(fluid.layers.equal(seg_id, cat_ids)),
            dtype='int64')

        seg_mat = fluid.layers.transpose(seg_mat, perm=[1, 2, 0])
        seg_mat = fluid.layers.unsqueeze(seg_mat, [-1])
        seg_mat = fluid.layers.one_hot(seg_mat, 2)
        seg_mat.stop_gradient = True
    else:
        seg_mat = None

    pos_emb = relative_positional_encoding(
        qlen,
        klen,
        d_model,
        clamp_len,
        attn_type,
        bi_data,
        bsz=bsz,
        dtype=data_type)
    pos_emb = fluid.layers.dropout(
        pos_emb, dropout, dropout_implementation="upscale_in_train")
    pos_emb.stop_gradient = True
    ##### Attention layers
    if mems is None:
        mems = [None] * n_layer

    for i in range(n_layer):
        # cache new mems
        #new_mems.append(_cache_mem(output_h, mems[i], mem_len, reuse_len)) 

        # segment bias
        if seg_id is None:
            r_s_bias_i = None
            seg_embed_i = None
        else:
            r_s_bias_i = r_s_bias if not untie_r else r_s_bias[i]
            seg_embed_i = seg_embed[i]

        if inp_q is not None:
            pass
        else:
            output_h = rel_multihead_attn(
                h=output_h,
                r=pos_emb,
                r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                seg_mat=seg_mat,
                r_s_bias=r_s_bias_i,
                seg_embed=seg_embed_i,
                attn_mask=non_tgt_mask,
                mems=mems[i],
                d_model=d_model,
                n_head=n_head,
                d_head=d_head,
                dropout=dropout,
                dropatt=dropatt,
                initializer=initializer,
                name=name + '_layer_{}'.format(i))

        if inp_q is not None:
            pass
        output_h = positionwise_ffn(
            inp=output_h,
            d_model=d_model,
            d_inner=d_inner,
            dropout_prob=dropout,
            param_initializer=initializer,
            act_type=ff_activation,
            name=name + '_layer_{}_ff'.format(i))
    if inp_q is not None:
        output = fluid.layers.dropout(
            output_g, dropout, dropout_implementation="upscale_in_train")
    else:
        output = fluid.layers.dropout(
            output_h, dropout, dropout_implementation="upscale_in_train")
    new_mems = None
    return output, new_mems, lookup_table


def lm_loss(hidden,
            target,
            n_token,
            d_model,
            initializer,
            lookup_table=None,
            tie_weight=False,
            bi_data=True):

    if tie_weight:
        assert lookup_table is not None, \
          'lookup_table cannot be None for tie_weight'
        softmax_w = lookup_table
    else:
        softmax_w = fluid.layers.create_parameter(
            shape=[n_token, d_model],
            dtype=hidden.dtype,
            attr=fluid.ParamAttr(
                name='model_loss_weight', initializer=initializer),
            is_bias=False)

    softmax_b = fluid.layers.create_parameter(
        shape=[n_token],
        dtype=hidden.dtype,
        attr=fluid.ParamAttr(
            name='model_lm_loss_bias', initializer=initializer),
        is_bias=False)

    logits = fluid.layers.matmul(
        x=hidden, y=softmax_w, transpose_y=True) + softmax_b

    loss = fluid.layers.softmax_cross_entropy_with_logits(
        input=logits, label=target)

    return loss


def summarize_sequence(summary_type,
                       hidden,
                       d_model,
                       n_head,
                       d_head,
                       dropout,
                       dropatt,
                       input_mask,
                       initializer,
                       scope=None,
                       reuse=None,
                       use_proj=True,
                       name=''):
    """
      Different classification tasks may not may not share the same parameters
      to summarize the sequence features.
      If shared, one can keep the `scope` to the default value `None`.
      Otherwise, one should specify a different `scope` for each task.
  """
    if summary_type == 'last':
        summary = hidden[-1]
    elif summary_type == 'first':
        summary = hidden[0]
    elif summary_type == 'mean':
        summary = fluid.layers.reduce_mean(hidden, axis=0)
    elif summary_type == 'attn':
        bsz = fluid.layers.slice(
            fluid.layers.shape(hidden), axes=[0], starts=[1], ends=[2])

        summary_bias = tf.get_variable(
            'summary_bias', [d_model],
            dtype=hidden.dtype,
            initializer=initializer)
        summary_bias = tf.tile(summary_bias[None, None], [1, bsz, 1])

        if input_mask is not None:
            input_mask = input_mask[None, :, :, None]

        summary = multihead_attn(
            summary_bias,
            hidden,
            hidden,
            input_mask,
            d_model,
            n_head,
            d_head,
            dropout,
            dropatt,
            is_training,
            initializer,
            residual=False)
        summary = summary[0]
    else:
        raise ValueError('Unsupported summary type {}'.format(summary_type))

    # use another projection as in BERT
    if use_proj:
        summary = fluid.layers.fc(input=summary,
                                  size=d_model,
                                  act='tanh',
                                  param_attr=fluid.ParamAttr(
                                      name=name + '_summary_weight',
                                      initializer=initializer),
                                  bias_attr=name + '_summary_bias')

    summary = fluid.layers.dropout(
        summary,
        dropout_prob=dropout,
        dropout_implementation="upscale_in_train")

    return summary


def classification_loss(hidden,
                        labels,
                        n_class,
                        initializer,
                        name,
                        reuse=None,
                        return_logits=False):
    """
      Different classification tasks should use different parameter names to ensure
      different dense layers (parameters) are used to produce the logits.
      An exception will be in transfer learning, where one hopes to transfer
      the classification weights.
    """

    logits = fluid.layers.fc(input=hidden,
                             size=n_class,
                             param_attr=fluid.ParamAttr(
                                 name=name + '_logit_weight',
                                 initializer=initializer),
                             bias_attr=name + '_logit_bias')

    one_hot_target = fluid.layers.one_hot(labels, depth=n_class)
    loss = -1.0 * fluid.layers.reduce_sum(
        log_softmax(logits) * one_hot_target, dim=-1)

    if return_logits:
        return loss, logits

    return loss


def regression_loss(hidden, labels, initializer, name, return_logits=False):

    logits = fluid.layers.fc(input=hidden,
                             size=1,
                             param_attr=fluid.ParamAttr(
                                 name=name + '_logits_weight',
                                 initializer=initializer),
                             bias_attr=name + '_logits_bias')

    loss = fluid.layers.square(logits - labels)

    if return_logits:
        return loss, logits

    return loss
