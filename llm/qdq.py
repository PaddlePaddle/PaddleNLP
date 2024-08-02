import paddle
from safetensors import safe_open

def cal_abs_max_channel(inputs, quant_axis=1):
    reduce_axis = tuple(
        [i for i in range(len(inputs.shape)) if i != quant_axis])
    abs_max_values = paddle.max(paddle.abs(inputs), axis=reduce_axis)
    abs_max_values = paddle.where(
        abs_max_values == paddle.to_tensor(0, dtype=inputs.dtype),
        paddle.to_tensor(1e-8, dtype=inputs.dtype), abs_max_values)
    return abs_max_values

def qdq_weight(x, quant_bit=8, quant_axis=1, scales=None, dequant=False):
    if scales is None:
        scales = cal_abs_max_channel(x)
    bnt = (1 << (quant_bit - 1)) - 1
    if not dequant:
        # quant
        quant_x = paddle.clip(paddle.round(x / scales * bnt), -bnt - 1, bnt)
        return quant_x, scales
    else:
        quant_x = x
        # dequant
        qdq_x = quant_x / bnt * scales
        return qdq_x, scales

#aa = safe_open('PaddleNLP/llm/checkpoints/ckpt_quant/model-00001-of-00008.safetensors', framework='np')
aaa = paddle.randn([500, 1000])
bbb, scale = qdq_weight(aaa)
ccc, _ = qdq_weight(bbb, scales=scale, dequant=True)
print(aaa - ccc)
import pdb; pdb.set_trace()
