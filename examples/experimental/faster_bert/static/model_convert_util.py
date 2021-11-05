# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

def fused_weight(weight, num_head):
    if paddle.in_dynamic_mode():
        a = paddle.transpose(weight, perm=[1, 0])
        return paddle.reshape(a, shape=[1, num_head, int(a.shape[0]/num_head), a.shape[1]])
    else:
        a = weight.transpose(1, 0)
        return a.reshape((1, num_head, int(a.shape[0]/num_head), a.shape[1]))

def fused_qkv(qkv_weight, num_head):
    q = qkv_weight['q']
    k = qkv_weight['k']
    v = qkv_weight['v']

    fq = fused_weight(q, num_head)
    fk = fused_weight(k, num_head)
    fv = fused_weight(v, num_head)
    if paddle.in_dynamic_mode():
        return paddle.concat(x=[fq, fk, fv], axis=0)
    else:
        return np.concatenate((fq, fk, fv), axis=0)

def convert_base_to_fused(state_to_load):
    base_to_fused = dict()
    base_to_fused["weight"] = "scale"
    base_to_fused["bias"] = "bias"

    fused_state_to_load = dict()
    qkv_weight = dict()
    qkv_bias = dict()
    qkv_count = 0
    num_head = 16
    layer_index = 0
    for key, tensor_value in state_to_load.items():
        array = key.split('.')
        fused_array = list(array)
        if paddle.in_dynamic_mode():
            value = tensor_value
        else:
            value = np.array(tensor_value)

        if len(array) == 6:#linear or layer_norm
            if 'linear' in array[4]:
                #linear1.weight -> ffn._linear1_weight
                #linear1.bias -> ffn._linear1_bias
                fused_array[5] = "_" + array[4] + "_" + array[5]
                fused_array[4] = "ffn"
                fused_key = '.'.join(fused_array)
                fused_state_to_load[fused_key] = value
            elif 'norm' in array[4]:
                if array[4][-1] == '1':
                    #norm1.weight -> fused_atten.pre_ln_scale
                    #norm2.weight -> fused_atten.ln_scale
                    fused_array[4] = "fused_attn"
                    fused_array[5] = "ln_" + base_to_fused[array[5]]
                    fused_key = '.'.join(fused_array)
                    fused_state_to_load[fused_key] = value
                else:
                    #norm1.weight -> ffn._ln1_scale
                    fused_array[4] = "ffn"
                    fused_array[5] = "_ln" + array[4][-1] + "_" + base_to_fused[array[5]]
                    fused_key = '.'.join(fused_array)
                    fused_state_to_load[fused_key] = value
        elif len(array) == 7:#self_atten
            if 'q' in array[5]:
                if array[6] == "weight":
                    qkv_weight['q'] = value
                else:
                    qkv_bias['q'] = value
                qkv_count += 1
            elif 'k' in array[5]:
                if array[6] == "weight":
                    qkv_weight['k'] = value
                else:
                    qkv_bias['k'] = value
                qkv_count += 1
            elif 'v' in array[5]:
                if array[6] == "weight":
                    qkv_weight['v'] = value
                else:
                    qkv_bias['v'] = value
                qkv_count += 1
            else:
                fused_array.pop()
                fused_array[4] = "fused_attn"
                if array[6] == "weight":
                    fused_array[5] = "linear_weight"
                else:
                    fused_array[5] = "linear_bias"
                fused_key = '.'.join(fused_array)
                fused_state_to_load[fused_key] = value
            if qkv_count == 6:
                qkv_count = 0
                fused_array.pop()

                fused_array[4] = "fused_attn"
                fused_array[5] = "qkv_weight"
                fused_key = '.'.join(fused_array)
                fused_state_to_load[fused_key] = fused_qkv(qkv_weight, num_head)

                fused_array[4] = "fused_attn"
                fused_array[5] = "qkv_bias"
                fused_key = '.'.join(fused_array)
                if paddle.in_dynamic_mode():
                    a = paddle.concat(x=[qkv_bias['q'], qkv_bias['k'], qkv_bias['v']], axis=0)
                    tmp_bias = paddle.reshape(a, shape=[3, num_head, int(a.shape[0]/3/num_head)])
                    fused_state_to_load[fused_key] = tmp_bias
                else:
                    a = np.concatenate((qkv_bias['q'], qkv_bias['k'], qkv_bias['v']), axis=0)
                    fused_state_to_load[fused_key] = a.reshape((3, num_head, int(a.shape[0]/3/num_head)))
        else:
            fused_state_to_load[key] = value
    return fused_state_to_load
