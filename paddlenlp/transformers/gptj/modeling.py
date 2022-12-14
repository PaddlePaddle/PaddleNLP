# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The EleutherAI Authors and The HuggingFace Inc. team
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer, Embedding

from ..nezha.modeling import ACT2FN
from .. import PretrainedModel, register_base_model

__all__ = [
    "GPTJModel",
    "GPTJPretrainedModel",
    "GPTJForCausalLM",
    "GPTJForSequenceClassification",
    "GPTJForQuestionAnswering",
]


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2) / dim))
    sinusoid_inp = paddle.einsum("i , j -> i j", paddle.arange(seq_len, dtype="float32"), inv_freq)
    return paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = paddle.stack((-x2, x1), axis=-1)
    # In einsum notation: rearrange(x, '... d j -> ... (d j)')
    return x.flatten(-2)


def duplicate_interleave(m):
    return paddle.repeat_interleave(m, 2, axis=1)


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :], sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class GPTJAttention(Layer):
    def __init__(self, embed_dim, rotary_dim, num_attention_heads, max_positions, attn_pdrop, resid_pdrop):
        super().__init__()

        self.register_buffer(
            "causal_mask",
            paddle.tril(paddle.ones((max_positions, max_positions), dtype=paddle.get_default_dtype())).reshape(
                (1, 1, max_positions, max_positions)
            ),
        )

        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = paddle.sqrt(paddle.to_tensor(self.head_dim, dtype="float32"))
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.rotary_dim = rotary_dim

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        new_shape = tensor.shape[:-1] + [num_attention_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            # (batch, blocks, head, block_length, head_features)
            return tensor.transpose([0, 1, 3, 2, 4])
        elif len(tensor.shape) == 4:
            # (batch, head, seq_length, head_features)
            return tensor.transpose([0, 2, 1, 3])
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.transpose([0, 1, 3, 2, 4])
        elif len(tensor.shape) == 4:
            tensor = tensor.transpose([0, 2, 1, 3])
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.shape[:-2] + [num_attention_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def _attn(self, query, key, value, attention_mask=None):

        # compute causal mask from causal mask buffer
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = paddle.cast(query, "float32")
        key = paddle.cast(key, "float32")
        attn_weights = paddle.matmul(query, key, transpose_y=True)

        mask_value = paddle.to_tensor(-1e9, dtype=attn_weights.dtype)
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        attn_weights = paddle.where(causal_mask, attn_weights, mask_value)
        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1, dtype=value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = paddle.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        use_cache=False,
        cache=None,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

        seq_len = key.shape[1]
        offset = 0

        if cache is not None:
            offset = cache[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = paddle.concat([k_rot, k_pass], axis=-1)
            query = paddle.concat([q_rot, q_pass], axis=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.transpose([0, 2, 1, 3])
        query = query.transpose([0, 2, 1, 3])

        if cache is not None:
            past_key = cache[0]
            past_value = cache[1]
            key = paddle.concat((past_key, key), axis=-2)
            value = paddle.concat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present


class GPTJMLP(Layer):
    def __init__(self, embed_dim, inner_dim, activation_function, resid_pdrop):
        super().__init__()

        self.fc_in = nn.Linear(embed_dim, inner_dim)
        self.fc_out = nn.Linear(inner_dim, embed_dim)

        self.act = ACT2FN[activation_function]
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTJBlock(Layer):
    def __init__(
        self,
        embed_dim,
        rotary_dim,
        n_head,
        n_positions,
        attn_pdrop,
        resid_pdrop,
        activation_function,
        layer_norm_epsilon,
    ):
        super().__init__()
        inner_dim = 4 * embed_dim
        self.ln_1 = nn.LayerNorm(embed_dim, epsilon=layer_norm_epsilon)
        self.attn = GPTJAttention(embed_dim, rotary_dim, n_head, n_positions, attn_pdrop, resid_pdrop)
        self.mlp = GPTJMLP(embed_dim, inner_dim, activation_function, resid_pdrop)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        use_cache=False,
        cache=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states, attention_mask=attention_mask, cache=cache, use_cache=use_cache)
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class GPTJPretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    pretrained_init_configuration = {}
    pretrained_resource_files_map = {"model_state": {}}

    base_model_prefix = "transformer"

    def init_weights(self, layer):
        """Initialize the weights."""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if isinstance(layer.weight, paddle.Tensor) and paddle.get_default_dtype() == "float32":
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range")
                        else self.transformer.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))
            layer._epsilon = getattr(self, "layer_norm_epsilon", 1e-05)
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            layer.bias.set_value(paddle.zeros_like(layer.bias))


@register_base_model
class GPTJModel(GPTJPretrainedModel):
    r"""
    The bare GPTJ Model outputting raw hidden-states.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `GPTJModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `GPTJModel`.
        bos_token_id (int, optional):
            The beginning of sequence token that was used during pretraining. Can be
            used a sequence classifier token.
            Defaults to `0`.
        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `50256`.
        eos_toke_idn (int, optional):
            A special token representing the end of a sequence that was used during pretraining.
            Defaults to `2`.
        n_embed (int, optional):
            Dimensionality of the embedding layer, decoder layer. Defaults to `1024`.
        n_layer (int, optional):
            Number of hidden layers. Defaults to `20`.
        n_head (int, optional):
            Number of attention heads for each attention layer in the Transformer decoder.
            Defaults to `16`.
        n_positions (int, optional):
            The maximum sequence length that this model might ever be used with.
            Defaults to `2048`.
        attn_pdrop (float, optional):
            The dropout probability used in MultiHeadAttention in all decoder layers to drop some attention target.
            Defaults to `0.0`.
        resid_pdrop (float, optional):
            The dropout probability for all residual layers in the decoder.
            Defaults to `0.0`.
        embd_pdrop (float, optional):
            The dropout probability used in embedding layers. Defaults to `0.0`.
        rotary_dim (int, optional):
            Dimensionality of rotay position embeddings.
            Defaults to `32`.
        activation_function (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions are supported.
            Defaults to `"gelu_new"`.
        layer_norm_epsilon (float, optional):
            The epsilon to use in the layer normalization layers.
            Defaults to `1e-05`.
        initializer_range (float, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Default to `0.02`.
    """

    def __init__(
        self,
        vocab_size,
        bos_token_id=50256,
        pad_token_id=50256,
        eos_token_id=50256,
        n_embd=4096,
        n_layer=28,
        n_head=16,
        n_positions=2048,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        rotary_dim=64,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.embed_dim = n_embd
        self.initializer_range = initializer_range
        self.wte = nn.Embedding(vocab_size, self.embed_dim)
        self.drop = nn.Dropout(embd_pdrop)
        self.h = nn.LayerList(
            [
                GPTJBlock(
                    n_embd,
                    rotary_dim,
                    n_head,
                    n_positions,
                    attn_pdrop,
                    resid_pdrop,
                    activation_function,
                    layer_norm_epsilon,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, epsilon=layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_cache=False,
        cache=None,
    ):
        r"""
        The GPTJModel forward method, overrides the `__call__()` special method.
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
            use_cache (bool, optional):
                 Whether or not to use cache. Defaults to `False`. If set to `True`, key value states will be returned and
                 can be used to speed up decoding.
            cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)`.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.
        Returns:
            Tensor: Returns tensor `decoder_output`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import GPTJModel, GPTJTokenizer
                tokenizer = GPTJTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
                model = GPTJModel.from_pretrained('EleutherAI/gpt-j-6B')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """

        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape(shape=(-1, input_shape[-1]))
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("You have to specify input_ids")

        if cache is None:
            past_length = 0
            cache = tuple([None] * len(self.h))
        else:
            past_length = cache[0][0].shape[-2]

        # Attention mask.
        if attention_mask is None:
            assert input_ids is not None, "input_ids should be " "specified when generating attention_mask"
            attention_mask = (
                paddle.cast(input_ids == self.pad_token_id, dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
            )
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
            attention_mask.stop_gradient = True

        inputs_embeds = self.wte(input_ids)

        hidden_states = self.drop(inputs_embeds)
        output_shape = input_shape[:] + [hidden_states.shape[-1]]

        presents = () if use_cache else None
        for i, (block, old_cache) in enumerate(zip(self.h, cache)):
            outputs = block(hidden_states, attention_mask=attention_mask, use_cache=use_cache, cache=old_cache)

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.reshape(shape=output_shape)

        last_hidden_state = hidden_states
        new_cache = presents

        return last_hidden_state, new_cache


class GPTJForCausalLM(GPTJPretrainedModel):
    r"""
    GPTJ Model with a `language modeling` head on top.
    Args:
        GPTJ (:class:`GPTJModel`):
            An instance of GPTJModel.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.lm_head = nn.Linear(self.transformer.config["n_embd"], self.transformer.config["vocab_size"])

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_faster_entry(self, kwargs):
        from paddlenlp.ops import FasterGPTJ

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decoding_lib = kwargs.get("decoding_lib", None)
        decode_strategy = kwargs.get("decode_strategy")
        if decode_strategy == "beam_search":
            raise AttributeError("'beam_search' is not supported yet in the faster version of GPTJ")
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.transformer.config["n_embd"] // self.transformer.config["n_head"]
        if size_per_head not in [32, 64, 80, 96, 128, 160, 192, 224, 256]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the faster version of GPTJ" % size_per_head
            )
        if kwargs["forced_bos_token_id"] is not None:
            # not support for min_length yet in the faster version
            raise AttributeError("'forced_bos_token_id != None' is not supported yet in the faster version")
        self._faster_entry = FasterGPTJ(self, decoding_lib=decoding_lib, use_fp16_decoding=use_fp16_decoding).forward
        return self._faster_entry

    def prepare_inputs_for_generation(self, input_ids, cache=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if cache:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, :, -1:, :]

        return {
            "input_ids": input_ids,
            "cache": cache,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, cache=None):
        r"""
        The GPTJForCausalLM forward method, overrides the __call__() special method.
        Args:
            input_ids (Tensor):
                See :class:`GPTJModel`.
            attention_mask (Tensor, optional):
                See :class:`GPTJModel`.
            use_cache (bool, optional):
                See :class:`GPTJModel`.
            cache (Tensor, optional):
                See :class:`GPTJModel`.
        Returns:
            Tensor or tuple: Returns Tensor `lm_logits` if `use_cache` is `False`, otherwise, returns tuple (`lm_logits`, `cache`).
            With the fields:
            - `lm_logits` (Tensor):
                The generated sentence of the model.
                Its data type should be float32 and has a shape of [batch_size, sequence_length, vocab_size].
            - `cache` (Tensor):
                See :class:`GPTJModel`.
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import GPTJForCausalLM, GPTJTokenizer
                tokenizer = GPTJTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
                model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
        """

        transformer_outputs = self.transformer(
            input_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )

        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = paddle.cast(self.lm_head(hidden_states), "float32")
        past_key_values = transformer_outputs[1]

        return lm_logits, past_key_values

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e


class GPTJForSequenceClassification(GPTJPretrainedModel):
    r"""
    GPTJ Model with a linear layer on top of the pooled output,
    designed for sequence classification/regression tasks like GLUE tasks.
    Since it does classification on the last token, it requires to know the
    position of the last token. If a `pad_token_id` is defined in the configuration,
    it finds the last token that is not a padding token in each row. If no `pad_token_id`
    is defined, it simply takes the last value in each row of the batch.

    Args:
        GPTJ (:class:`GPTJModel`):
            An instance of GPTJModel.
        num_labels (int, optional):
            The number of different labels. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of GPTJ.
            If None, use the same value as `hidden_dropout_prob` of `GPTJModel`
            instance `GPTJ`. Defaults to None.
    """

    def __init__(self, transformer, num_labels=2):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(self.transformer.config["n_embd"], num_labels, bias_attr=False)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_cache=False,
        cache=None,
    ):
        r"""
        The GPTJForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`GPTJModel`.
            attention_mask (Tensor, optional):
                See :class:`GPTJModel`.
            use_cache (bool, optional):
                See :class:`GPTJModel`.
            cache (Tensor, optional):
                See :class:`GPTJModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_labels]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import GPTJForSequenceClassification, GPTJTokenizer
                tokenizer = GPTJTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
                model = GPTJForSequenceClassification.from_pretrained('EleutherAI/gpt-j-6B')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """
        transformer_outputs = self.transformer(
            input_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )

        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)
        batch_size = input_ids.shape[0]

        if self.transformer.config.get("pad_token_id", None) is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.transformer.config.get("pad_token_id", None) is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (input_ids != self.transformer.config["pad_token_id"]).sum(-1) - 1

        pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]

        return pooled_logits


class GPTJForQuestionAnswering(GPTJPretrainedModel):
    r"""
    GPTJ Model with a linear layer on top of the hidden-states output to
    compute `span_start_logits` and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        GPTJ (:class:`GPTJModel`):
            An instance of GPTJModel.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(self.transformer.config["n_embd"], 2)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_cache=False,
        cache=None,
    ):
        r"""
        The GPTJForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`GPTJModel`.
            attention_mask (Tensor, optional):
                See :class:`GPTJModel`.
            use_cache (bool, optional):
                See :class:`GPTJModel`.
            cache (Tensor, optional):
                See :class:`GPTJModel`.

        Returns:
            tuple: Returns tuple (`start_logits`, `end_logits`).

            With the fields:

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import GPTJForQuestionAnswering, GPTJTokenizer

                tokenizer = GPTJTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
                model = GPTJForQuestionAnswering.from_pretrained('EleutherAI/gpt-j-6B')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_token_type_ids=False)
                outputs = model(**inputs)
                start_logits = outputs[0]
                end_logits  =outputs[1]
        """
        transformer_outputs = self.transformer(
            input_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )

        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        return start_logits, end_logits
