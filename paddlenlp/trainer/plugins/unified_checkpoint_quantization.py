# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed as dist

from paddlenlp.utils.checkpoint_quantization_utils import (
    asymmetry_qdq_weight,
    cal_ratio,
    group_wise_quant_dequant,
    merge_int4,
    qdq_weight,
)
from paddlenlp.utils.env import (
    ASYMMETRY_QUANT_SCALE_MAX,
    ASYMMETRY_QUANT_SCALE_MIN,
    MOMENT1_KEYNAME,
    MOMENT2_KEYNAME,
    SYMMETRY_QUANT_SCALE,
)
from paddlenlp.utils.log import logger


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
                    m1_quant, m1_codebook = group_wise_quant_dequant(state_dict[m1_key], quant_bits=4, symmetry=True)
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
