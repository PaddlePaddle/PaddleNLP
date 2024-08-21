import paddle
from safetensors import safe_open
import numpy as np

def cal_abs_max_channel(inputs, quant_axis=1):
    reduce_axis = tuple(
        [i for i in range(len(inputs.shape)) if i != quant_axis])
    abs_max_values = paddle.max(paddle.abs(inputs), axis=reduce_axis)
    abs_max_values = paddle.where(
        abs_max_values == paddle.to_tensor(0, dtype=inputs.dtype),
        paddle.to_tensor(1e-8, dtype=inputs.dtype), abs_max_values)
    return abs_max_values

def qdq_weight(x, quant_bit=8, quant_axis=-1, scales=None, dequant=False, rank=-1, world_size=1, peek=False):
    if scales is None:
        scales = cal_abs_max_channel(x)
    bnt = (1 << (quant_bit - 1)) - 1
    if not dequant:
        # quant
        quant_x = np.clip(np.round(x / scales * bnt), -bnt - 1, bnt)
        return quant_x.astype(np.int8), scales
    else:
        quant_x = x
        # dequant
        if not peek:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                qdq_x = quant_x / bnt * scales
            else:
                qdq_x = quant_x / bnt * scales[rank * scales.shape[0] // world_size: (rank + 1) * scales.shape[0] // world_size]
            return qdq_x.astype(np.float32), scales
        else:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                qdq_x = quant_x / bnt * scales.unsqueeze(0).expand(quant_x.shape)
            else:
                qdq_x = quant_x / bnt * scales[rank * scales.shape[0] // world_size: (rank + 1) * scales.shape[0] // world_size].unsqueeze(0).expand(quant_x.shape)
            return qdq_x.astype(paddle.float32), scales

def cal_abs_min_max_channel(inputs, quant_axis=1):
    reduce_axis = tuple(
        [i for i in range(len(inputs.shape)) if i != quant_axis])
    abs_max_values = paddle.max(inputs, axis=reduce_axis)
    abs_min_values = paddle.min(inputs, axis=reduce_axis)
    abs_max_values = paddle.where(
        abs_max_values == paddle.to_tensor(0, dtype=inputs.dtype),
        paddle.to_tensor(1e-8, dtype=inputs.dtype), abs_max_values)
    abs_min_values = paddle.where(
        abs_min_values == paddle.to_tensor(0, dtype=inputs.dtype),
        paddle.to_tensor(1e-8, dtype=inputs.dtype), abs_min_values)
    return abs_max_values, abs_min_values

def asymmetry_qdq_weight(x, quant_bit=8, quant_axis=-1, mins=None, maxs=None, dequant=False, rank=-1, world_size=1, peek=False):
    if mins is None:
        maxs, mins = cal_abs_min_max_channel(x)
    bnt = (1 << (quant_bit)) - 1
    scales = maxs - mins
    if not dequant:
        # quant
        quant_x = np.clip(np.round((x - mins) / scales * bnt), 0, bnt)
        return quant_x.astype(np.uint8), mins, maxs
    else:
        quant_x = x
        # dequant
        if not peek:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                qdq_x = (quant_x / bnt * scales) + mins
            else:
                qdq_x = (quant_x / bnt * scales[rank * scales.shape[0] // world_size: (rank + 1) * scales.shape[0] // world_size]) + mins[rank * mins.shape[0] // world_size: (rank + 1) * mins.shape[0] // world_size]
            return qdq_x.astype(np.float32), scales
        else:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                qdq_x = (quant_x / bnt * scales.unsqueeze(0).expand(quant_x.shape)) + mins
            else:
                qdq_x = (quant_x / bnt * scales[rank * scales.shape[0] // world_size: (rank + 1) * scales.shape[0] // world_size].unsqueeze(0).expand(quant_x.shape)) + mins[rank * mins.shape[0] // world_size: (rank + 1) * mins.shape[0] // world_size]
            return qdq_x.astype(paddle.float32), scales

#aa = safe_open('PaddleNLP/llm/checkpoints/ckpt_quant/model-00001-of-00008.safetensors', framework='np')
aaa = paddle.randn([500, 1000])
aaa = aaa + aaa.abs().max() + 1
bbb, mins, maxs = asymmetry_qdq_weight(aaa)
ccc, _ = asymmetry_qdq_weight(bbb, mins=mins, maxs=maxs, dequant=True, peek=True)
print(aaa - ccc)
import pdb; pdb.set_trace()
