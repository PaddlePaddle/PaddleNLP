# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.fluid import layers
from paddle.fluid.data_feeder import convert_dtype
from paddle.nn import Dropout, LayerNorm, Linear


def _convert_param_attr_to_list(param_attr, n):
    """
    If `param_attr` is a list or tuple, convert every element in it to a
    ParamAttr instance. Otherwise, repeat `param_attr` `n` times to
    construct a list, and rename every one by appending a increasing index
    suffix to avoid having same names when `param_attr` contains a name.
    Parameters:
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`.
        n (int): The times to repeat to construct a list when `param_attr`
            is not a list or tuple.
    Returns:
        list: A list composed of each including cell's `param_attr`.
    """
    if isinstance(param_attr, (list, tuple)):
        assert len(param_attr) == n, "length of param_attr should be %d when it is a list/tuple" % n
        param_attrs = []
        for attr in param_attr:
            if isinstance(attr, bool):
                if attr:
                    param_attrs.append(ParamAttr._to_attr(None))
                else:
                    param_attrs.append(False)
            else:
                param_attrs.append(ParamAttr._to_attr(attr))
        # param_attrs = [ParamAttr._to_attr(attr) for attr in param_attr]
    elif isinstance(param_attr, bool):
        param_attrs = []
        if param_attr:
            param_attrs = [ParamAttr._to_attr(None) for i in range(n)]
        else:
            param_attrs = [False] * n
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs


def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.
    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.
    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == "bool" or "int" in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.
    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.
        weight_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr` .
        bias_attr (ParamAttr|bool, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr` .
    Examples:
        .. code-block:: python
            import paddle
            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
        weight_attr=None,
        bias_attr=None,
    ):
        super().__init__()

        assert embed_dim > 0, "Expected embed_dim to be greater than 0, " "but received {}".format(embed_dim)
        assert num_heads > 0, "Expected num_heads to be greater than 0, " "but received {}".format(num_heads)

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = Linear(self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = Linear(self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = Linear(embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.
        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`.
            value (Tensor): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`.
            cache (MultiHeadAttention.Cache|MultiHeadAttention.StaticCache, optional):
                It is a namedtuple with `k` and `v` as fields, and stores tensors
                shaped `[batch_size, num_heads, length, embed_dim]` which are results
                of linear projection, reshape and transpose calculations in
                MultiHeadAttention. If is an instance of `Cache`, `k` and `v`
                fields reserve intermediate results of previous positions, which
                mostly used for decoder self attention. If it is an instance of
                `StaticCache`, `key` and `value` args would be ignored, `k` and
                `v` fields would be used as calculated results on `key` and
                `value`, which mostly used for decoder-encoder cross attention.
                It is only used for inference and should be None for training.
                Default None.
        Returns:
            tuple: A tuple including linear projected keys and values. These two \
                tensors have shapes `[batch_size, n_head, sequence_length, d_key]` \
                and `[batch_size, n_head, sequence_length, d_value]` separately, \
                and their data types are same as inputs.
        """
        q = self.q_proj(query)
        q = paddle.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = paddle.concat([cache.k, k], axis=2)
            v = paddle.concat([cache.v, v], axis=2)
            cache = self.Cache(k, v)

        return (q, k, v) if cache is None else (q, k, v, cache)

    def compute_kv(self, key, value):
        r"""
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.
        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.
        Parameters:
            key (Tensor): The keys for multi-head attention. It is a tensor
                with shape `[batch_size, sequence_length, kdim]`. The data type
                should be float32 or float64.
            value (Tensor): The values for multi-head attention. It is a tensor
                with shape `[batch_size, sequence_length, vdim]`. The data type
                should be float32 or float64.
        Returns:
            tuple: A tuple including transformed keys and values. Their shapes \
                both are `[batch_size, num_heads, sequence_length, embed_dim // num_heads]`, \
                and their data types are same as inputs.
        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = paddle.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = paddle.transpose(x=k, perm=[0, 2, 1, 3])
        v = paddle.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = paddle.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        `Cache` or `StaticCache` is namedtuple with `k` and `v` as fields,
        and it stores tensors shaped `[batch_size, num_heads, length, embed_dim]`
        which are results of linear projection, reshape and transpose calculations
        in MultiHeadAttention.
        If the generated cache is an instance of `Cache`, `k` and `v` fields
        reserve intermediate result tensors of previous positions, and the tensors
        are incremental among decoding steps, which mostly are used for decoder
        decoder self attention.
        If the generated cache is an instance of `StaticCache`, `k` and `v` fields
        would be used as calculated result tensors on keys an values in `forward`,
        and the tensors keep unchanged among decoding steps, which are mostly used
        for decoder-encoder cross attention.
        The cache is generated as follows:
        1. If `type` is `StaticCache`, apply `compute_kv(key, value)` and use the
        results to create an instance of `StaticCache`.
        2. If `type` is `Cache` and `value` is None, generate empty tensors shaped
        `[batch_size, num_heads, 0, embed_dim // num_heads]` and use the results
        to create an instance of `Cache`, where `batch_size` is from the first
        dimension of `key`.
        3. If `type` is `Cache` and `value` is not None, use `key`, `value` to create
        an instance of `Cache`.
        Parameters:
            key (Tensor): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If `value` is None,
                it is only for batch size and data type reference.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, `key` is only
                for batch size reference. Default None.
            type (type): It should be `MultiHeadAttention.StaticCache` or
                `MultiHeadAttention.Cache` to indicate the cache type to generate.
        Returns:
            namedtuple: an instance of `Cache` or `StaticCache` accordingly.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0,
            )
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0,
            )
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
            cache (MultiHeadAttention.Cache|MultiHeadAttention.StaticCache, optional):
                It is a namedtuple with `k` and `v` as fields, and stores tensors
                shaped `[batch_size, num_heads, length, embed_dim]` which are results
                of linear projection, reshape and transpose calculations in
                MultiHeadAttention. If it is an instance of `Cache`, `k` and `v`
                fields reserve intermediate results of previous positions, which
                mostly used for decoder self attention. If it is an instance of
                `StaticCache`, `key` and `value` args would be ignored, `k` and
                `v` fields would be used as calculated results on `key` and
                `value`, which mostly used for decoder-encoder cross attention.
                It is only used for inference and should be None for training.
                Default None.
        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)

        # scale dot product attention
        product = paddle.matmul(x=q * (self.head_dim**-0.5), y=k, transpose_y=True)
        if attn_mask is not None:
            # Support bool or int mask
            # attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask

        weights = F.softmax(product)
        # breakpoint()
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train",
            )

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerDecoderLayer(nn.Layer):
    """
    TransformerDecoderLayer is composed of three sub-layers which are decoder
    self (multi-head) attention, decoder-encoder cross attention and feedforward
    network. Before and after each sub-layer, pre-process and post-precess would
    be applied on the input and output accordingly. If `normalize_before` is True,
    pre-process is layer normalization and post-precess includes dropout, residual
    connection. Otherwise, no pre-process and post-precess includes dropout, residual
    connection, layer normalization.
    Parameters:
        d_model (int): The expected feature size in the input and output.
        nhead (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, `weight_attr[0]` would be used as `weight_attr` for
            self attention, `weight_attr[1]` would be used as `weight_attr` for
            cross attention, and `weight_attr[2]` would be used as `weight_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `weight_attr` to create parameters. Default: None, which means the
            default weight parameter property is used. See usage for details
            in :ref:`api_paddle_fluid_param_attr_ParamAttr` .
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, `bias_attr[0]` would be used as `bias_attr` for
            self attention, `bias_attr[1]` would be used as `bias_attr` for
            cross attention, and `bias_attr[2]` would be used as `bias_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `bias_attr` to create parameters. The `False` value means the
            corresponding layer would not have trainable bias parameter. See
            usage for details in :code:`ParamAttr` . Default: None,which means
            the default bias parameter property is used.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.nn import TransformerDecoderLayer
            # decoder input: [batch_size, tgt_len, d_model]
            dec_input = paddle.rand((2, 4, 128))
            # encoder output: [batch_size, src_len, d_model]
            enc_output = paddle.rand((2, 6, 128))
            # self attention mask: [batch_size, n_head, tgt_len, tgt_len]
            self_attn_mask = paddle.rand((2, 2, 4, 4))
            # cross attention mask: [batch_size, n_head, tgt_len, src_len]
            cross_attn_mask = paddle.rand((2, 2, 4, 6))
            decoder_layer = TransformerDecoderLayer(128, 2, 512)
            output = decoder_layer(dec_input,
                                   enc_output,
                                   self_attn_mask,
                                   cross_attn_mask)  # [2, 4, 128]
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="gelu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=False,
        weight_attr=None,
        bias_attr=None,
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()

        assert d_model > 0, "Expected d_model to be greater than 0, " "but received {}".format(d_model)
        assert nhead > 0, "Expected nhead to be greater than 0, " "but received {}".format(nhead)
        assert dim_feedforward > 0, "Expected dim_feedforward to be greater than 0, " "but received {}".format(
            dim_feedforward
        )

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model, nhead, dropout=attn_dropout, weight_attr=weight_attrs[0], bias_attr=bias_attrs[0]
        )
        self.cross_attn = MultiHeadAttention(
            d_model, nhead, dropout=attn_dropout, weight_attr=weight_attrs[1], bias_attr=bias_attrs[1]
        )
        self.linear1 = Linear(d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])
        self.norm1 = LayerNorm(d_model, epsilon=1e-12)
        self.norm2 = LayerNorm(d_model, epsilon=1e-12)
        self.norm3 = LayerNorm(d_model, epsilon=1e-12)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        r"""
        Applies a Transformer decoder layer on the input.
        Parameters:
            tgt (Tensor): The input of Transformer decoder layer. It is a tensor
                with shape `[batch_size, target_length, d_model]`. The data type
                should be float32 or float64.
            memory (Tensor): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            tgt_mask (Tensor, optional): A tensor used in self attention
                to prevents attention to some unwanted positions, usually the
                the subsequent positions. It is a tensor with shape broadcasted
                to `[batch_size, n_head, target_length, target_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
            memory_mask (Tensor, optional): A tensor used in decoder-encoder
                cross attention to prevents attention to some unwanted positions,
                usually the paddings. It is a tensor with shape broadcasted to
                `[batch_size, n_head, target_length, source_length]`. When the
                data type is bool, the unwanted positions have `False` values
                and the others have `True` values. When the data type is int,
                the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
            cache (tuple, optional): It is a tuple( :code:`(incremental_cache, static_cache)` ),
                `incremental_cache` is an instance of `MultiHeadAttention.Cache`,
                `static_cache` is an instance of `MultiHeadAttention.StaticCache.
                See `TransformerDecoderLayer.gen_cache` for more details. It is
                only used for inference and should be None for training. Default
                None.
        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `tgt`, representing the output of Transformer decoder layer. \
                Or a tuple if `cache` is not None, except for decoder layer output, \
                the tuple includes the new cache which is same as input `cache` \
                argument but `incremental_cache` in it has an incremental length. \
                See `MultiHeadAttention.gen_cache` and `MultiHeadAttention.forward` \
                for more details.
        """
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask, cache=None)
        else:
            tgt, incremental_cache = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask, cache=cache[0])
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)
        # breakpoint()
        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if cache is None:
            tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else:
            tgt, static_cache = self.cross_attn(tgt, memory, memory, memory_mask, cache[1])
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)
        # breakpoint()
        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)

        if not self.normalize_before:
            tgt = self.norm3(tgt)
        # breakpoint()
        return tgt if cache is None else (tgt, (incremental_cache, static_cache))

    def gen_cache(self, memory):
        r"""
        Generates cache for `forward` usage. The generated cache is a tuple
        composed of an instance of `MultiHeadAttention.Cache` and an instance
        of `MultiHeadAttention.StaticCache`.
        Parameters:
            memory (Tensor): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
        Returns:
            tuple: It is a tuple( :code:`(incremental_cache, static_cache)` ). \
                `incremental_cache` is an instance of `MultiHeadAttention.Cache` \
                produced by `self_attn.gen_cache(memory, MultiHeadAttention.Cache)`, \
                it reserves two tensors shaped `[batch_size, nhead, 0, d_model // nhead]`. \
                `static_cache` is an instance of `MultiHeadAttention.StaticCache` \
                produced by `cross_attn.gen_cache(memory, MultiHeadAttention.StaticCache)`, \
                it reserves two tensors shaped `[batch_size, nhead, source_length, d_model // nhead]`.
                See `MultiHeadAttention.gen_cache` and `MultiHeadAttention.forward` \
                for more details.
        """
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        static_cache = self.cross_attn.gen_cache(memory, memory, type=self.cross_attn.StaticCache)
        return incremental_cache, static_cache
