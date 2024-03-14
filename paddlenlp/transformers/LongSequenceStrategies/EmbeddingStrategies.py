from paddle import Tensor, nn
import paddle

__all__ = [
    "RotaryEmbedding",
    "LinearScalingRotaryEmbedding",
    "NTKScalingRotaryEmbedding",
    "DynamicNTKScalingRotaryEmbedding"
]

class RotaryEmbedding(nn.Layer):
    def __init__(self, **init_args):
        super().__init__()
        self.dim = init_args['dim']
        self.max_position_embeddings = init_args['max_position_embeddings']
        self.base = init_args['base']
        self.position_encoding_2d =  init_args['position_encoding_2d'] if 'position_encoding_2d' in init_args else False
        if self.position_encoding_2d:
            # [dim / 4]# 2D--Embedding
            self.dim = self.dim / 2
            inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype=paddle.float32) / self.dim))
        else:
            # [dim / 2]
            inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype=paddle.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq)       
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype=paddle.float32)
        # [seq_len, dim/2]
        with paddle.amp.auto_cast(enable=False):
            freqs = paddle.outer(t.astype(self.inv_freq.dtype), self.inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[:,:]
        self.sin_cached = emb.sin()[:,:]
    def forward(self, seq_len=None ,ntk_alpha=None):

            return self.cos_cached[:seq_len,:],self.sin_cached[:seq_len,:]

class LinearScalingRotaryEmbedding(RotaryEmbedding):
    def __init__(self, **init_args):
        self.scaling_factor = init_args['scaling_factor'] 
        print("I am here LinearScalingRotaryEmbedding")
        super().__init__(**init_args)
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype=paddle.float32)
        t = t / self.scaling_factor
        # [seq_len, dim/2]
        with paddle.amp.auto_cast(enable=False):
            freqs = paddle.outer(t.astype(self.inv_freq.dtype), self.inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[:,:]
        self.sin_cached = emb.sin()[:,:]

class NTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/"""

    def __init__(self, **init_args):
        init_args['base'] = init_args['base'] * init_args['scaling_factor'] ** (init_args['dim'] / (init_args['dim'] - 2))
        print("I am here NTKScalingRotaryEmbedding",init_args['base'])
        super().__init__(**init_args)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/"""

    def __init__(self, **init_args):
        self.scaling_factor = init_args['scaling_factor']
        self._seq_len_cached = 0
        super().__init__(**init_args)

    def _scale_cos_sin(self, seq_len , ntk_alpha = None):
        # [seq_len]
        t = paddle.arange(seq_len, dtype=paddle.float32)
        if ntk_alpha == None:
            ntk_alpha = (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
        # [seq_len, dim/2]
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype=paddle.float32) / self.dim))
        with paddle.amp.auto_cast(enable=False):
            freqs = paddle.outer(t.astype(inv_freq.dtype), inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)          
        self.cos_cached = emb.cos()[:, :]
        self.sin_cached = emb.sin()[:, :]

    def forward(self, seq_len=None , ntk_alpha = None):

        if seq_len > self.max_position_embeddings:
            print("I am here DynamicNTKScalingRotaryEmbedding")
            self._scale_cos_sin(seq_len=seq_len,ntk_alpha=ntk_alpha)
            
        return super().forward(seq_len)