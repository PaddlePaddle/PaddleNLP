from paddle import Tensor, nn
import paddle
import math
__all__ = [
    "AttentionWithLinearBias"
]
class AttentionWithLinearBias(nn.Layer):
    '''
    init_args:bool_attention_mask,num_heads,dtype,tensor_parallel_degree
    '''
    def __init__(self,**init_args):
        '''
        **init_args:...
        '''
        super().__init__()
    def _get_interleave(self,n):
        def _get_interleave_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                _get_interleave_power_of_2(closest_power_of_2)
                + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )
    def forward(self,
                bool_attention_mask: Tensor,
                num_heads: int, 
                dtype: paddle.dtype, 
                tensor_parallel_degree=1      
    ):
        attention_mask = bool_attention_mask.astype("float32")
        batch_size, seq_length = attention_mask.shape[0], attention_mask.shape[-1]
        slopes = paddle.to_tensor(self._get_interleave(num_heads), dtype="float32")
        alibi = slopes.unsqueeze(axis=[1, 2]) * paddle.arange(seq_length, dtype="float32").unsqueeze(axis=[0, 1]).expand(
            [num_heads, -1, -1]
        )
        alibi = alibi.reshape(shape=(1, num_heads, 1, seq_length)).expand([batch_size, -1, -1, -1])
        return paddle.cast(alibi, dtype)     
           


        