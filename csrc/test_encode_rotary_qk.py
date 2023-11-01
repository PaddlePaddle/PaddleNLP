import paddle
from paddlenlp_ops import (
    encode_rotary_qk,
    qkv_transpose_split,
    rebuild_padding,
    transpose_remove_padding,
    write_cache_kv,
)

batch_size = 2
num_head = 2
head_dim = 128
seq = 643
# rotary_embs is [2, batch_size, seq_length, self.head_size]
rotary_embs = paddle.randn([2, batch_size, seq, head_dim // 2], dtype="float32")
rotary_embs = rotary_embs.unsqueeze(-1).tile([1, 1, 1, 1, 2]).reshape([2, batch_size, seq, head_dim])

q_out = paddle.randn([700, num_head, head_dim], dtype="float16")
k_out = paddle.randn([700, num_head, head_dim], dtype="float16")
seq_lens = paddle.to_tensor([[57],[643]], dtype="int32")

new_q_out = paddle.zeros([batch_size, seq, num_head, head_dim], dtype="float16")
new_q_out[0,:57] = q_out[:57,:,:]
new_q_out[1,:643] = q_out[57:,:,:]

new_k_out = paddle.zeros([batch_size, seq, num_head, head_dim], dtype="float16")
new_k_out[0,:57] = k_out[:57,:,:]
new_k_out[1,:643] = k_out[57:,:,:]

cos = rotary_embs[0]
sin = rotary_embs[1]
cos = cos.reshape([batch_size, seq, 1, head_dim // 2, 2])
sin = sin.reshape([batch_size, seq, 1, head_dim // 2, 2])
cos0 = cos[:,:,:,:,0]
cos1 = cos[:,:,:,:,1]
sin0 = sin[:,:,:,:,0]
sin1 = sin[:,:,:,:,1]

new_q_out = new_q_out.reshape([batch_size, seq, num_head, head_dim // 2, 2]).cast("float32")
new_q_out0 = new_q_out[:,:,:,:,0]
new_q_out1 = new_q_out[:,:,:,:,1]

new_q_out00 = cos0 * new_q_out0 - sin0 * new_q_out1
new_q_out11 = cos1 * new_q_out1 + sin1 * new_q_out0
new_q_out00 = new_q_out00.unsqueeze(-1)
new_q_out11 = new_q_out11.unsqueeze(-1)
new_q_out = paddle.concat([new_q_out00, new_q_out11], axis = -1)
new_q_out = new_q_out.reshape([batch_size, seq, num_head , head_dim])
new_q_out = paddle.concat([new_q_out[0,:57], new_q_out[1,:643]],axis=0)



encode_rotary_qk(
    q_out,
    k_out,
    rotary_embs,
    seq_lens,
    rotary_emb_dims=1,
    use_neox=False,
)

#print(paddle.max(new_q_out.cast("float32") - q_out.cast("float32")))

print(new_q_out[0,0,:5])
print(q_out[0,0,:5])



