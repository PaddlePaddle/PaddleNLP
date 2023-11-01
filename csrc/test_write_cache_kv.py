import paddle
from paddlenlp_ops import (
    encode_rotary_qk,
    qkv_transpose_split,
    rebuild_padding,
    transpose_remove_padding,
    write_cache_kv,
)

max_batch = 2
num_head = 20
cache_max_seq = 1288
x = 8
head_dim = 128
cache_kv = paddle.zeros([2, max_batch, num_head, cache_max_seq, head_dim],dtype="float16")

k_out = paddle.randn([700, num_head, head_dim], dtype="float16")
v_out = paddle.randn([700, num_head, head_dim], dtype="float16")
seq_lens = paddle.to_tensor([[57],[643]], dtype="int32")



new_k_out = paddle.zeros([max_batch, cache_max_seq, num_head, head_dim], dtype="float16")
new_k_out[0,:57] = k_out[:57,:,:]
new_k_out[1,:643] = k_out[57:,:,:]
new_v_out = paddle.zeros([max_batch, cache_max_seq, num_head, head_dim], dtype="float16")
new_v_out[0,:57] = v_out[:57,:,:]
new_v_out[1,:643] = v_out[57:,:,:]



new_k_out = new_k_out.reshape([max_batch, cache_max_seq, num_head, head_dim // x, x])
new_k_out = new_k_out.transpose([0, 2, 3, 1, 4])
new_k_out = new_k_out.reshape([max_batch, num_head, cache_max_seq, head_dim])

new_v_out = new_v_out.reshape([max_batch, cache_max_seq, num_head, head_dim // x, x])
new_v_out = new_v_out.transpose([0, 2, 1, 3, 4])
new_v_out = new_v_out.reshape([max_batch, num_head, cache_max_seq, head_dim])

write_cache_kv(k_out, v_out, cache_kv, seq_lens)

print(paddle.max(cache_kv[0] - new_k_out))
print(paddle.max(cache_kv[1] - new_v_out))


