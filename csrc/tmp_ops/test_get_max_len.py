from tmp_ops import get_max_len 
import paddle 
import os 


seq_lens_encoder = paddle.ones([4], dtype='int32')
seq_lens_decoder = paddle.zeros([4], dtype='int32')

get_max_len(seq_lens_encoder, seq_lens_decoder)
