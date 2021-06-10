#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Transformer encoder."""

from functools import partial

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers


def gelu(x):
    """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
    cdf = 0.5 * (1.0 + fluid.layers.tanh(
        (np.sqrt(2.0 / np.pi) * (x + 0.044715 * fluid.layers.pow(x, 3.0)))))
    return x * cdf


def _build_linear_column_parallel(x, n_in, n_out, name, init, mp):
    return paddle.distributed.split(
        x,
        size=(n_in, n_out),
        operation='linear',
        axis=1,
        gather_out=False,
        num_partitions=mp,
        weight_attr=fluid.ParamAttr(
            name='%s.w_0' % name if name is not None else None,
            initializer=init),
        bias_attr='%s.b_0' % name if name is not None else None, )


def _build_linear_row_parallel(x, n_in, n_out, name, init, mp):
    return paddle.distributed.split(
        x,
        size=(n_in, n_out),
        operation='linear',
        axis=0,
        gather_out=True,
        num_partitions=mp,
        weight_attr=fluid.ParamAttr(
            name='%s.w_0' % name if name is not None else None,
            initializer=init),
        bias_attr='%s.b_0' % name if name is not None else None, )


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
                         param_initializer=None,
                         name='multi_head_att',
                         topo=None):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    #if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
    #    raise ValueError(
    #        "Inputs: quries, keys and values should all be 3-D tensors. but {} v.s. {} v.s. {}"\
    #                .format(queries.shape, keys.shape, values.shape))

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        if topo is None or topo.mp.size == 1:
            q = layers.fc(input=queries,
                          size=d_key * n_head,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=name + '_query_fc.w_0',
                              initializer=param_initializer),
                          bias_attr=name + '_query_fc.b_0')
            k = layers.fc(input=keys,
                          size=d_key * n_head,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=name + '_key_fc.w_0',
                              initializer=param_initializer),
                          bias_attr=name + '_key_fc.b_0')
            v = layers.fc(input=values,
                          size=d_value * n_head,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=name + '_value_fc.w_0',
                              initializer=param_initializer),
                          bias_attr=name + '_value_fc.b_0')
        else:
            q = _build_linear_column_parallel(
                queries, d_model, d_model,
                name + '_query_fc_' + str(topo.mp.rank), param_initializer,
                topo.mp.size)
            k = _build_linear_column_parallel(
                keys, d_model, d_model, name + '_key_fc_' + str(topo.mp.rank),
                param_initializer, topo.mp.size)
            v = _build_linear_column_parallel(
                values, d_model, d_model,
                name + '_value_fc_' + str(topo.mp.rank), param_initializer,
                topo.mp.size)
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
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
        scaled_q = layers.scale(x=q, scale=d_key**-0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        k = cache["k"] = layers.concat(
            [layers.reshape(
                cache["k"], shape=[0, 0, d_model]), k], axis=1)
        v = cache["v"] = layers.concat(
            [layers.reshape(
                cache["v"], shape=[0, 0, d_model]), v], axis=1)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    if topo is None or topo.mp.size == 1:
        proj_out = layers.fc(input=out,
                             size=d_model,
                             num_flatten_dims=2,
                             param_attr=fluid.ParamAttr(
                                 name=name + '_output_fc.w_0',
                                 initializer=param_initializer),
                             bias_attr=name + '_output_fc.b_0')
    else:
        proj_out = _build_linear_row_parallel(
            out, d_model, d_model, name + '_output_fc_' + str(topo.mp.rank),
            param_initializer, topo.mp.size)
    return proj_out


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_initializer=None,
                              name='ffn',
                              topo=None):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """

    assert hidden_act == 'gelu.approximate'
    if topo is None or topo.mp.size == 1:
        hidden = layers.fc(input=x,
                           size=d_inner_hid,
                           num_flatten_dims=2,
                           act=None,
                           param_attr=fluid.ParamAttr(
                               name=name + '_fc_0.w_0',
                               initializer=param_initializer),
                           bias_attr=name + '_fc_0.b_0')
    else:
        hidden = _build_linear_column_parallel(
            x, d_hid, d_inner_hid, name + '_fc_0_' + str(topo.mp.rank),
            param_initializer, topo.mp.size)
    hidden = gelu(hidden)

    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)

    if topo is None or topo.mp.size == 1:
        out = layers.fc(input=hidden,
                        size=d_hid,
                        num_flatten_dims=2,
                        param_attr=fluid.ParamAttr(
                            name=name + '_fc_1.w_0',
                            initializer=param_initializer),
                        bias_attr=name + '_fc_1.b_0')
    else:
        out = _build_linear_row_parallel(hidden, d_inner_hid, d_hid,
                                         name + '_fc_1_' + str(topo.mp.rank),
                                         param_initializer, topo.mp.size)
    return out


def pre_post_process_layer(prev_out,
                           out,
                           process_cmd,
                           dropout_rate=0.,
                           epsilon=1e-12,
                           name=''):
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
                param_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_scale',
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_bias',
                    initializer=fluid.initializer.Constant(0.)),
                epsilon=epsilon)
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


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
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name='',
                  epsilon=1e-12,
                  topo=None):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """

    attn_output = multi_head_attention(
        enc_input,
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att',
        topo=topo)

    attn_output = post_process_layer(
        enc_input,
        attn_output,
        'an',
        prepostprocess_dropout,
        name=name + '_post_att',
        epsilon=epsilon)

    ffd_output = positionwise_feed_forward(
        attn_output,
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn',
        topo=topo)

    return post_process_layer(
        attn_output,
        ffd_output,
        'an',
        prepostprocess_dropout,
        name=name + '_post_ffn',
        epsilon=epsilon)


def encoder_layer_preln(enc_input,
                        attn_bias,
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        hidden_act,
                        preprocess_cmd="n",
                        postprocess_cmd="da",
                        param_initializer=None,
                        name='',
                        epsilon=1e-12,
                        topo=None):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """

    enc_ln_output = post_process_layer(
        None,
        enc_input,
        'n',
        prepostprocess_dropout,
        name=name + '_post_att',
        epsilon=epsilon)

    attn_output = multi_head_attention(
        enc_ln_output,
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att',
        topo=topo)

    attn_output = post_process_layer(
        enc_input,
        attn_output,
        'a',
        prepostprocess_dropout,
        name=name + '_post_att',
        epsilon=epsilon)

    attn_ln_output = post_process_layer(
        None,
        attn_output,
        'n',
        prepostprocess_dropout,
        name=name + '_post_ffn',
        epsilon=epsilon)

    ffd_output = positionwise_feed_forward(
        attn_ln_output,
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn',
        topo=topo)

    return post_process_layer(
        attn_output,
        ffd_output,
        'a',
        prepostprocess_dropout,
        name=name + '_post_ffn',
        epsilon=epsilon)


def encoder_inner_share(enc_input,
                        attn_bias,
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        hidden_act,
                        preprocess_cmd,
                        postprocess_cmd,
                        epsilon,
                        param_initializer=None,
                        name='',
                        n_layer_per_block=1,
                        topo=None):
    """
       The encoder_inner_share is composed of n_layer_per_block layers returned by calling
       encoder_layer.
    """
    _checkpoints = []
    for i in range(n_layer_per_block):
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
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i),
            epsilon=epsilon,
            topo=topo)
        _checkpoints.append(enc_output.name)
        enc_input = enc_output

    return enc_output, _checkpoints


def encoder_outer_share(enc_input,
                        attn_bias,
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        hidden_act,
                        preprocess_cmd,
                        postprocess_cmd,
                        epsilon,
                        param_initializer=None,
                        name='',
                        n_layer_per_block=1,
                        topo=None,
                        preln=False):
    """
       The encoder_outer_share is composed of n_layer_per_block layers returned by calling
       encoder_layer.
    """
    enc_fn = encoder_layer_preln if preln else encoder_layer
    _checkpoints = []
    for i in range(n_layer_per_block):
        enc_output = enc_fn(
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
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            name=name,
            epsilon=epsilon,
            topo=topo)
        _checkpoints.append(enc_output.name)
        enc_input = enc_output

    return enc_output, _checkpoints


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
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            epsilon,
            n_layer_per_block,
            param_initializer=None,
            name='',
            param_share=None,
            topo=None,
            preln=False):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer .
    """
    checkpoints = []
    # http://wiki.baidu.com/pages/viewpage.action?pageId=978216217
    # for outer_share it will share same param in one block,
    # and for inner_share it will share param across blocks, rather than in one same block
    #
    # outer-share   inner-block
    #    [1]           [1]      ----\ 1st block
    #    [1]           [2]      ----/
    #    [2]           [1]      ----\ 2nd block
    #    [2]           [2]      ----/

    if param_share == "normal" or param_share == 'outer_share':
        #n_layer_per_block = 24 for bert-large
        enc_fn = encoder_outer_share
        name_fn = lambda i: name + '_layer_' + str(i)
    elif param_share == "inner_share":
        #n_layer_per_block = 2
        enc_fn = encoder_inner_share
        name_fn = lambda i: name
    else:
        raise ValueError('unsupported param share mode')

    layer_per_stage = n_layer // topo.pp.size

    for i in range(n_layer // n_layer_per_block):
        with fluid.device_guard(f'gpu:{i//layer_per_stage}'):
            attn_bias.stop_gradient = True
            attn_bias.persistable = True
            enc_output, cp = enc_fn(
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
                hidden_act,
                preprocess_cmd,
                postprocess_cmd,
                param_initializer=param_initializer,
                name=name_fn(i),
                n_layer_per_block=n_layer_per_block,
                epsilon=epsilon,
                topo=topo,
                preln=preln)
            if i % layer_per_stage == 0:
                checkpoints.extend(cp)
            enc_input = enc_output

    if preln:
        with fluid.device_guard(f'gpu:{topo.pp.size-1}'):
            enc_output = post_process_layer(
                None,
                enc_output,
                'n',
                prepostprocess_dropout,
                name='post_encoder',
                epsilon=epsilon)
    with fluid.device_guard(f'gpu:{topo.pp.size-1}'):
        enc_output = pre_process_layer(
            enc_output,
            preprocess_cmd,
            prepostprocess_dropout,
            name="post_encoder",
            epsilon=epsilon)

    return enc_output, checkpoints
