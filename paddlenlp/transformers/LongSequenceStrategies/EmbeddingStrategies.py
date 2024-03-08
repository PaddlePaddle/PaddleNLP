from paddle import Tensor, nn
import paddle



class RotaryEmbedding(nn.Layer):
    def __init__(self, **init_args):
        '''
        **init_args:dim, max_position_embeddings=2048, base=10000
        '''
        super().__init__()
        self.dim = init_args['dim']
        self.max_position_embeddings = init_args['max_position_embeddings']
        self.base = init_args['base']
        self.model_type = init_args['model_type'] if 'model_type' in init_args else None
        self.position_encoding_2d =  init_args['position_encoding_2d'] if 'position_encoding_2d' in init_args else False
        self.scaling_factor = init_args['scaling_factor'] if 'scaling_factor' in init_args else 1
        if self.position_encoding_2d:
            # [dim / 4]# chatglm1 2D-Pos-Embedding
            self.dim = self.dim / 2
            self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        else:
            # [dim / 2]
            self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        t = t / self.scaling_factor
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[:,:]
        self.sin_cached = emb.sin()[:,:]
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
            return self.cos_cached,self.sin_cached

class LinearScalingRotaryEmbedding(RotaryEmbedding):
    def __init__(self, **init_args):
        '''
        **init_args:dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0
        '''
        self.scaling_factor = init_args['scaling_factor']
        super().__init__(**init_args)

class NTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/"""

    def __init__(self, **init_args):
        '''
         **init_args: dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0
        '''
        init_args['base'] = init_args['base'] * init_args['scaling_factor'] ** (init_args['dim'] / (init_args['dim'] - 2))
        print("NTKScalingRotaryEmbedding:init_args['base'] is " , init_args['base'])
        self.scaling_factor = init_args['scaling_factor'] 
        super().__init__(**init_args)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/"""

    def __init__(self, **init_args):
        '''
        **init_args: dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0
        '''
        self.scaling_factor = init_args['scaling_factor']
        super().__init__(**init_args)

    def _scale_cos_sin(self, seq_len):
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        alpha = (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        base = self.base * alpha ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        freqs = paddle.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [seqlen, 1, dim]            
        scale_cos = emb.cos()[:, :]
        scale_sin = emb.sin()[:, :]

        return scale_cos, scale_sin

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        print("DynamicNTKScalingRotaryEmbedding:seq_len is " ,seq_len)

        if seq_len > self.max_position_embeddings:
            print("self._scale_cos_sin(seq_len=seq_len)")
            scale_cos, scale_sin = self._scale_cos_sin(seq_len=seq_len)
            self.cos_cached, self.sin_cached = scale_cos, scale_sin
        return super().forward(x,seq_len)