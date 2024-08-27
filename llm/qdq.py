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

def group_wise_quant_dequant(inputs, mins=None, maxs=None, quant_bits=4, group_size=128, quant=True, use_pd=False):
    qmax = (1 << (quant_bits)) - 1
    qmin = 0
    shape = inputs.shape

    if quant:
        inputs_processed = inputs.reshape([shape[0] // group_size, group_size, shape[1]])
        # scales: [shape[0] // group_size, shape[1]]
        maxs = np.max(inputs_processed, axis=1)
        mins = np.min(inputs_processed, axis=1)
        scales = maxs - mins
        # new_scales: [shape[0], shape[1]]
        new_scales = np.repeat(scales, repeats=group_size, axis=0)
        new_mins = np.repeat(mins, repeats=group_size, axis=0)
        quant_tensor = np.clip(np.round((inputs - new_mins) / new_scales * qmax), qmin, qmax)
        return quant_tensor.astype('uint8'), mins, maxs
    else:
        scales = maxs - mins
        if use_pd:
            new_scales = paddle.repeat_interleave(scales, group_size, 0)
            new_mins = paddle.repeat_interleave(mins, group_size, 0)
        else:
            new_scales = np.repeat(scales, repeats=group_size, axis=0)
            new_mins = np.repeat(mins, repeats=group_size, axis=0)

        if len(new_scales.shape) == 0 or inputs.shape[-1] == new_scales.shape[-1]:
            dequant_tensor = inputs * new_scales / qmax + new_mins
        else:
            dequant_tensor = (inputs / qmax * new_scales[rank * new_scales.shape[0] // world_size: (rank + 1) * new_scales.shape[0] // world_size]) + new_mins[rank * new_mins.shape[0] // world_size: (rank + 1) * new_mins.shape[0] // world_size]
        return dequant_tensor

def merge_int4(x, y):
    offset = 2 ** 4
    res = x * offset + y
    return res

def split_int8(z):
    offset = 2 ** 4
    x, y = z // offset, z % offset
    return x, y


#aa = safe_open('PaddleNLP/llm/checkpoints/ckpt_quant/model-00001-of-00008.safetensors', framework='np')
aaa = paddle.randn([1024, 1024])
aaa = aaa + aaa.abs().max() + 1
bbb, mins, maxs = asymmetry_qdq_weight(aaa)
ccc, _ = asymmetry_qdq_weight(bbb, mins=mins, maxs=maxs, dequant=True, peek=True)
print(aaa - ccc)
aaa = np.random.randn(1024, 1024)
aaa, bbb = np.random.randn(1024, 1024), np.random.randn(1024, 1024)
ccc, cmins, cmaxs = group_wise_quant_dequant(aaa)
ddd, dmins, dmaxs = group_wise_quant_dequant(bbb)

eee = merge_int4(ccc, ddd)
fff, ggg = split_int8(eee)

hhh = group_wise_quant_dequant(fff, mins=cmins, maxs=cmaxs, quant=False)
iii = group_wise_quant_dequant(ggg, mins=dmins, maxs=dmaxs, quant=False)
print(aaa - hhh)
print(bbb - iii)
import pdb; pdb.set_trace()
