def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    sequence_parallel=False,
    reshard_layer=None,
    attn_dropout_prob=0.0,
    trainer_mode=False,
    sliding_window=None,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    if hasattr(config, "attn_logit_softcapping") and config.attn_logit_softcapping > 0:
        attn_logit_softcapping = config.attn_logit_softcapping
    else:
        attn_logit_softcapping = None

    if config.use_flash_attention and flash_attention:

        version = paddle.version.full_version
        if version != "0.0.0" and version <= "2.5.2":
            if alibi is not None:
                raise ValueError("Flash Attention doesn't support alibi")
            attn_output, attn_weights = flash_attention(
                query_states,
                key_states,
                value_states,
                causal=True,
                dropout=attn_dropout_prob,
                return_softmax=output_attentions,
            )
        else:
            if alibi is not None:
                alibi = alibi.reshape([bsz, num_heads, 1, -1])
                attention_mask = attention_mask.cast(alibi.dtype) + alibi
            
            if sliding_window is not None and sliding_window > 0:
                window_mask = paddle.ones((q_len, kv_seq_len), dtype=query_states.dtype)
                window_mask = paddle.triu(window_mask, diagonal=1-sliding_window)
                window_mask = paddle.tril(window_mask, diagonal=0)
                window_mask = 1.0 - window_mask
                window_mask = window_mask * paddle.finfo(query_states.dtype).min
                window_mask = window_mask.unsqueeze([0, 1])  # [1, 1, q_len, kv_seq_len]
                
                if attention_mask is not None:
                    attention_mask = attention_mask + window_mask
                else:
                    attention_mask = window_mask
            
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
            )
            attn_weights = None

        if reshard_layer is not None:
            attn_output = reshard_layer(
                attn_output,
                split_axis=1,
                concat_axis=2,
            )
            assert (
                config.sep_parallel_degree > 1 and q_len % config.sep_parallel_degree == 0
            ), f"q_len:{q_len}, config.sep_parallel_degree:{config.sep_parallel_degree}"
            q_len = q_len // config.sep_parallel_degree
            num_heads = num_heads * config.sep_parallel_degree

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output
    else:
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        attn_weights = paddle.matmul(query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2]))
        if alibi is not None:
            alibi = alibi.reshape([bsz, num_heads, 1, -1])
            attn_weights = attn_weights + alibi

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if reshard_layer is not None:
            attention_mask = None

        if sliding_window is not None and sliding_window > 0:
            window_mask = paddle.ones((q_len, kv_seq_len), dtype=attn_weights.dtype)
            window_mask = paddle.triu(window_mask, diagonal=1-sliding_window)
            window_mask = paddle.tril(window_mask, diagonal=0)
            window_mask = 1.0 - window_mask
            window_mask = window_mask * paddle.finfo(attn_weights.dtype).min
            window_mask = window_mask.unsqueeze([0, 1])  # [1, 1, q_len, kv_seq_len]
            
            if attention_mask is not None:
                attention_mask = attention_mask + window_mask
            else:
                attention_mask = window_mask

        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)
        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attn_weights + attention_mask
        
        if attn_logit_softcapping is not None:
            attn_weights = attn_logit_softcapping * F.tanh(attn_weights / attn_logit_softcapping)
            
        if not paddle.in_dynamic_mode():
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        else:
            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        attn_weights = F.dropout(attn_weights, attn_dropout_prob, training=trainer_mode)
        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        if reshard_layer is not None:
            attn_output = reshard_layer(
                attn_output,
                split_axis=1,
                concat_axis=2,
            )
            q_len = q_len // config.sep_parallel_degree
            num_heads = num_heads * config.sep_parallel_degree

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        mask = paddle.concat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


class Gemma2RMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if hasattr(config, "sequence_parallel") and config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x):
        if hasattr(self.config, "use_fused_rms_norm") and self.config.use_fused_rms_norm:
            return rms_norm_fused(x, self.weight, self.variance_epsilon)

        output = self._norm(x.astype(paddle.float32)).astype(x.dtype)
        return output * self.weight


class Gemma2RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))

    def forward(self, x, seq_len=None):
        t = paddle.arange(seq_len, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat([freqs, freqs], axis=-1)
        return (emb.cos()[None, :, None, :].cast(dtype=x.dtype), emb.sin()[None, :, None, :].cast(dtype=x.dtype))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):

    if position_ids is None:
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Gemma2MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tensor_parallel_degree = getattr(config, "tensor_parallel_degree", 1)

        if hasattr(config, "sequence_parallel") and config.sequence_parallel:
            self.gate_proj = Linear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
                name="gate_proj",
            )
            self.up_proj = Linear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
                name="up_proj",
            )
            self.down_proj = Linear(
                self.intermediate_size,
                self.hidden_size,
                scatter_input=False,
                has_bias=False,
                name="down_proj",
            )
        else:
            self.gate_proj = Linear(
                self.hidden_size,
                self.intermediate_size,
                parallel="col",
                has_bias=False,
                name="gate_proj",
            )
            self.up_proj = Linear(
                self.hidden_size,
                self.intermediate_size,
                parallel="col",
                has_bias=False,
                name="up_proj",
            )
            self.down_proj = Linear(
                self.intermediate_size,
                self.hidden_size,
                parallel="row",
                has_bias=False,
                name="down_proj",
            )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
