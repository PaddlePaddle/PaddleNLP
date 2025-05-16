import paddle
import FusedQuantOps as FQO


from paddle.incubate.nn.functional import swiglu

# o1.shape = [ 8 * 4096, 2048 * 2], dtype = "bfloat16"
# prob.shape = [8 * 4096, 1], dtype = "float32"
# do2_p.shape = [8 * 4096, 2048], 

seq_len = 4096
topk = 8
moe_intermediate_size = 2025

o1 = paddle.rand( [seq_len * topk, moe_intermediate_size * 2], dtype="bfloat16")

unzipped_probs = paddle.rand( [ seq_len * topk, 1], dtype="float32")

do2_s = paddle.rand( [seq_len * topk, moe_intermediate_size], dtype="bfloat16")


def swiglu_grad(x, dz):
    xs = paddle.split(x, 2, axis=-1)
    lhs = xs[0] 
    rhs = xs[1]
    
    sig = paddle.nn.functional.sigmoid(lhs)
    tmp = sig * lhs
    x0_grad = dz * rhs * sig * (1 + lhs - tmp)
    x1_grad = dz * tmp
    
    x_grad = paddle.concat([x0_grad, x1_grad], axis=-1)
    
    return x_grad

def fn_gold():
    # do2: 前向从bfloat16-->float32，反向从float32-->bfloat16,do2 需要保持 bfloat16（因为 o2 是 bfloat16)
    o2 = swiglu(o1)
    o2_s = (o2 * unzipped_probs)
    do2 = (do2_s.cast(paddle.float32) * unzipped_probs)
    do2 = do2.cast(paddle.bfloat16)
    
    # 展开swiglu_grad(o1, do2)部分
    xs = paddle.split(o1, 2, axis=-1)
    lhs = xs[0]
    rhs = xs[1]
    sig = paddle.nn.functional.sigmoid(lhs)
    tmp = sig * lhs
    x0_grad = do2 * rhs * sig * (1 + lhs - tmp)
    x1_grad = do2 * tmp
    do1 = paddle.concat([x0_grad, x1_grad], axis=-1)
    
    probs_grad = (do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))).sum(axis=-1)
    return do1, probs_grad
def fn_fused():
    return FQO.fused_swiglu_probs_bwd(o1, do2_s, unzipped_probs)

#input: o1 unzipped_probs do2
def fn(): 
    # do2: 前向从bfloat16-->float32，反向从float32-->bfloat16,do2 需要保持 bfloat16（因为 o2 是 bfloat16)
    o2 = swiglu(o1)
    o2_s = (o2 * unzipped_probs)
    do2 = (do2_s.cast(paddle.float32) * unzipped_probs)
    do2 = do2.cast(paddle.bfloat16)
    do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
    probs_grad = (do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))).sum(axis=-1)
    return do1, probs_grad


do1_gold, pg_gold = fn_gold()
do1, pg= fn_fused()
print("do1_gold", do1_gold.astype("float32").numpy())
print("pg_gold", pg_gold.astype("float32").numpy())
print("do1", do1.astype("float32").numpy())
print("pg", pg.astype("float32").numpy())