# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

sd = paddle.load("old/model_state.pdparams")

layers = 28
for l in range(layers):
    # qkv spilt --> fuse
    qkv_weight_name, qkv_bias_name = (
        f"encoder.layers.{l}.self_attention.query_key_value.weight",
        f"encoder.layers.{l}.self_attention.query_key_value.bias",
    )
    q_weight_name, q_bias_name = (
        f"encoder.layers.{l}.self_attention.query.weight",
        f"encoder.layers.{l}.self_attention.query.bias",
    )
    k_weight_name, k_bias_name = (
        f"encoder.layers.{l}.self_attention.key.weight",
        f"encoder.layers.{l}.self_attention.key.bias",
    )
    v_weight_name, v_bias_name = (
        f"encoder.layers.{l}.self_attention.value.weight",
        f"encoder.layers.{l}.self_attention.value.bias",
    )
    sd[qkv_weight_name] = paddle.concat([sd[q_weight_name], sd[k_weight_name], sd[v_weight_name]], axis=1)
    sd[qkv_bias_name] = paddle.concat([sd[q_bias_name], sd[k_bias_name], sd[v_bias_name]], axis=0)
    sd.pop(q_weight_name)
    sd.pop(q_bias_name)
    sd.pop(k_weight_name)
    sd.pop(k_bias_name)
    sd.pop(v_weight_name)
    sd.pop(v_bias_name)

    # MLP
    mlp_weight_name = f"encoder.layers.{l}.mlp.dense_h_to_4h.weight"
    sd[mlp_weight_name] = sd[mlp_weight_name].reshape([4096, -1, 2]).transpose([0, 2, 1]).reshape([4096, -1])

paddle.save(sd, "new/model_state.pdparams")
