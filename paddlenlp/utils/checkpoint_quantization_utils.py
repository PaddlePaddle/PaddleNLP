# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import paddle


def quant_unified_optimizer(state_dict, state_dict_type, ckpt_quant_stage):
    quant = False
    if ckpt_quant_stage != "O0":
        quant = True
    del_key = []
    if quant and state_dict_type == "optimizer_weight":
        codebook_dict = {}
        opt_keys = state_dict.keys()
        all_bits, quant_bits = paddle.to_tensor(0.0), paddle.to_tensor(0.0)
        for k in opt_keys:
            momentum1 = k.endswith(MOMENT1_KEYNAME)
            momentum2 = k.endswith(MOMENT2_KEYNAME)
            k_size = state_dict[k].size
            if momentum1 or momentum2:
                all_bits += k_size * 4

            quant_weight = None

            if ckpt_quant_stage == "O1":
                # m1: wint8, 1/(sqrt(m2)+eps): wint8
                if momentum2:
                    # m1: m1_quant_weight, m2: ratio
                    m1_key = k.split("/")[0] + "/" + MOMENT1_KEYNAME
                    ratio = cal_ratio(state_dict[m1_key], state_dict[k])
                    m1_quant, codebook = qdq_weight(state_dict[m1_key], quant_bit=8)
                    quant_weight, mins, maxs = asymmetry_qdq_weight(ratio, quant_bit=8)
                    state_dict[m1_key] = m1_quant
                    codebook_dict[m1_key + SYMMETRY_QUANT_SCALE] = codebook
                    codebook_dict[k + ASYMMETRY_QUANT_SCALE_MIN] = mins
                    codebook_dict[k + ASYMMETRY_QUANT_SCALE_MAX] = maxs
                elif not momentum1:
                    quant_weight = state_dict[k]
            elif ckpt_quant_stage == "O2":
                # m1: bw-wint4, 1/(sqrt(m2)+eps): bw-wint4
                if momentum2:
                    if len(state_dict[k].shape) < 2:
                        continue
                    # m1: m1_quant_weight, m2: ratio
                    m1_key = k.split("/")[0] + "/" + MOMENT1_KEYNAME
                    ratio = cal_ratio(state_dict[m1_key], state_dict[k])
                    m1_quant, m1_codebook = group_wise_quant_dequant(state_dict[m1_key], quant_bits=4, symetry=True)
                    quant_weight, r_mins, r_maxs = group_wise_quant_dequant(ratio, quant_bits=4)
                    quant_weight = merge_int4(m1_quant, quant_weight)
                    codebook_dict[m1_key + SYMMETRY_QUANT_SCALE] = m1_codebook
                    codebook_dict[k + ASYMMETRY_QUANT_SCALE_MIN] = r_mins
                    codebook_dict[k + ASYMMETRY_QUANT_SCALE_MAX] = r_maxs
                    del_key.append(m1_key)
                elif not momentum1:
                    quant_weight = state_dict[k]

            if quant_weight is not None:
                state_dict[k] = quant_weight

        for k in del_key:
            state_dict.pop(k, None)

        state_dict.update(codebook_dict)

        if paddle.distributed.get_world_size() > 1:
            dist.all_reduce(all_bits)
            dist.all_reduce(quant_bits)

        model_numel = all_bits / 4
        all_bits = model_numel * 7.0
        quant_bits_mw = quant_bits + model_numel * 6.0
        quant_bits = quant_bits + model_numel * 2.0
        logger.info(
            f"all bits: {all_bits.item()}, quant bits: {quant_bits.item()}, quant bits mw: {quant_bits_mw.item()}"
        )
        logger.info(f"quant ratio (w/o Master Weight): {(all_bits.item() - quant_bits.item()) / all_bits.item()}")
        logger.info(f"quant ratio (w/ Master Weight): {(all_bits.item() - quant_bits_mw.item()) / all_bits.item()}")

    return state_dict


# cal adam update ratio
def cal_ratio(m, v, eps=1e-8):
    return 1 / (np.sqrt(v) + eps)


# group-wise quantization (support symmetry, asymmetry)
def group_wise_quant_dequant(
    inputs,
    mins=None,
    maxs=None,
    quant_bits=4,
    group_size=32,
    quant=True,
    rank=-1,
    world_size=1,
    use_pd=False,
    symetry=False,
):
    qmax = (1 << (quant_bits)) - 1
    qmin = 0
    shape = inputs.shape

    if quant:
        inputs_processed = inputs.reshape([shape[0] // group_size, group_size, shape[1]])
        if symetry:
            bnt = (1 << (quant_bits - 1)) - 1
            scales = np.max(np.abs(inputs_processed), axis=1)
            new_scales = np.repeat(scales, repeats=group_size, axis=0)
            quant_tensor = np.clip(np.round(inputs / new_scales * bnt), -bnt - 1, bnt)
            return quant_tensor.astype("int8"), scales

        # scales: [shape[0] // group_size, shape[1]]
        maxs = np.max(inputs_processed, axis=1)
        mins = np.min(inputs_processed, axis=1)
        scales = maxs - mins
        # new_scales: [shape[0], shape[1]]
        new_scales = np.repeat(scales, repeats=group_size, axis=0)
        new_mins = np.repeat(mins, repeats=group_size, axis=0)
        # add eps to avoid devide zero
        quant_tensor = np.clip(np.round((inputs - new_mins) / (new_scales) * qmax), qmin, qmax)
        quant_tensor = np.nan_to_num(quant_tensor)
        return quant_tensor.astype("uint8"), mins, maxs
    else:
        if symetry:
            scales = mins
            bnt = (1 << (quant_bits - 1)) - 1
            if use_pd:
                new_scales = paddle.repeat_interleave(scales, group_size, 0)
            else:
                new_scales = np.repeat(scales, repeats=group_size, axis=0)

            if rank == -1:
                dequant_tensor = inputs.astype("float32") * new_scales / bnt
            elif len(new_scales.shape) == 0 or inputs.shape[-1] == new_scales.shape[-1]:
                dequant_tensor = (
                    inputs.astype("float32")
                    * new_scales[
                        rank * new_scales.shape[0] // world_size : (rank + 1) * new_scales.shape[0] // world_size
                    ]
                    / bnt
                )
            else:
                dequant_tensor = (
                    inputs.astype("float32")
                    * new_scales[
                        :, rank * new_scales.shape[-1] // world_size : (rank + 1) * new_scales.shape[-1] // world_size
                    ]
                    / bnt
                )
            return dequant_tensor

        scales = maxs - mins
        if use_pd:
            new_scales = paddle.repeat_interleave(scales, group_size, 0)
            new_mins = paddle.repeat_interleave(mins, group_size, 0)
        else:
            new_scales = np.repeat(scales, repeats=group_size, axis=0)
            new_mins = np.repeat(mins, repeats=group_size, axis=0)

        if rank == -1:
            dequant_tensor = (inputs.astype("float32") / qmax * new_scales) + new_mins
        elif len(new_scales.shape) == 0 or inputs.shape[-1] == new_scales.shape[-1]:
            dequant_tensor = (
                inputs.astype("float32")
                / qmax
                * new_scales[rank * new_scales.shape[0] // world_size : (rank + 1) * new_scales.shape[0] // world_size]
            ) + new_mins[rank * new_mins.shape[0] // world_size : (rank + 1) * new_mins.shape[0] // world_size]
        else:
            dequant_tensor = (
                inputs.astype("float32")
                / qmax
                * new_scales[
                    :, rank * new_scales.shape[-1] // world_size : (rank + 1) * new_scales.shape[-1] // world_size
                ]
            ) + new_mins[:, rank * new_mins.shape[-1] // world_size : (rank + 1) * new_mins.shape[-1] // world_size]
        return dequant_tensor


# merge 2 signed int4 to 1 int8
def merge_int4(x, y):
    int4_high = x << 4
    int4_low = y & 0x0F
    final = int4_high | int4_low
    return final.astype("int8")


# split an int8 to 2 int4 elems
def split_int8(final):
    int4_high = final >> 4
    int4_low = final & 0x0F

    int4_high = np.where(int4_high > 8, int4_high - 16, int4_high)

    high_tensor = paddle.Tensor(int4_high, zero_copy=True)
    low_tensor = paddle.Tensor(int4_low, zero_copy=True)

    return high_tensor, low_tensor


# channel-wise min max scales calculation
def cal_abs_min_max_channel(inputs, quant_axis=1):
    reduce_axis = tuple([i for i in range(len(inputs.shape)) if i != quant_axis])
    abs_max_values = np.max(inputs, axis=reduce_axis)
    abs_min_values = np.min(inputs, axis=reduce_axis)
    abs_max_values = np.where(
        abs_max_values == np.array(0, dtype=inputs.dtype), np.array(1e-8, dtype=inputs.dtype), abs_max_values
    )
    abs_min_values = np.where(
        abs_min_values == np.array(0, dtype=inputs.dtype), np.array(1e-8, dtype=inputs.dtype), abs_min_values
    )
    return abs_max_values, abs_min_values


# channel-wise asymmetry quantization
def asymmetry_qdq_weight(
    x, quant_bit=8, quant_axis=-1, mins=None, maxs=None, dequant=False, rank=-1, world_size=1, peek=False
):
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
                qdq_x = (
                    quant_x
                    / bnt
                    * scales[rank * scales.shape[0] // world_size : (rank + 1) * scales.shape[0] // world_size]
                ) + mins[rank * mins.shape[0] // world_size : (rank + 1) * mins.shape[0] // world_size]
            return qdq_x.astype(np.float32), scales
        else:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                qdq_x = (quant_x / bnt * scales.unsqueeze(0).expand(quant_x.shape)) + mins
            else:
                qdq_x = (
                    quant_x
                    / bnt
                    * scales[rank * scales.shape[0] // world_size : (rank + 1) * scales.shape[0] // world_size]
                    .unsqueeze(0)
                    .expand(quant_x.shape)
                ) + mins[rank * mins.shape[0] // world_size : (rank + 1) * mins.shape[0] // world_size]
            return qdq_x.astype(paddle.float32), scales


# channel-wise abs max calculation
def cal_abs_max_channel(inputs, quant_axis=1):
    reduce_axis = tuple([i for i in range(len(inputs.shape)) if i != quant_axis])
    abs_max_values = np.max(np.abs(inputs), axis=reduce_axis)
    abs_max_values = np.where(
        abs_max_values == np.array(0, dtype=inputs.dtype), np.array(1e-8, dtype=inputs.dtype), abs_max_values
    )
    return abs_max_values


# channel-wise symmetry quantization
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
                qdq_x = (
                    quant_x
                    / bnt
                    * scales[rank * scales.shape[0] // world_size : (rank + 1) * scales.shape[0] // world_size]
                )
            # fp32 , int8, int, fp32 or fp64
            return qdq_x.astype(np.float32), scales
        else:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                qdq_x = quant_x / bnt * scales.unsqueeze(0).expand(quant_x.shape)
            else:
                qdq_x = (
                    quant_x
                    / bnt
                    * scales[rank * scales.shape[0] // world_size : (rank + 1) * scales.shape[0] // world_size]
                    .unsqueeze(0)
                    .expand(quant_x.shape)
                )
            # fp32 , int8, int, fp32 or fp64
            return qdq_x.astype(paddle.float32), scales
