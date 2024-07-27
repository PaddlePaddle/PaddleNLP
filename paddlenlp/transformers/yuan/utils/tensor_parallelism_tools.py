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

""" Yuan model tools"""

import paddle


def rearrange_model_weights(model_path, save_path, tp_degree, hidden_layers):

    print("load yuan weights ......")
    model = paddle.load(model_path)
    print("load yuan weights finish ......")

    size = model["model.layers.0.self_attn.q_proj.weight"].shape[0]
    step = size // tp_degree
    for i in range(0, hidden_layers):

        q = model[f"model.layers.{i}.self_attn.q_proj.weight"]
        k = model[f"model.layers.{i}.self_attn.k_proj.weight"]
        q_slices = [q[:, i : i + step] for i in range(0, size, step)]
        k_slices = [k[:, i : i + step] for i in range(0, size, step)]
        q1 = paddle.concat(q_slices[0::2], 1)
        q2 = paddle.concat(k_slices[0::2], 1)
        k1 = paddle.concat(q_slices[1::2], 1)
        k2 = paddle.concat(k_slices[1::2], 1)

        model[f"model.layers.{i}.self_attn.q_proj.weight"] = paddle.concat([q1, q2], 1)
        model[f"model.layers.{i}.self_attn.k_proj.weight"] = paddle.concat([k1, k2], 1)
        print(i, " layer is finished ......")

    paddle.save(model, save_path)
    print("Model weights saved.")


# Example usage:
rearrange_model_weights(
    model_path="/workspace/model_state.pdparams",
    save_path="/workspace/yuan_paddle/model_state.pdparams",
    tp_degree=2,  # set tensor parallel degree
    hidden_layers=24,  # set the number of hidden_layers, from config.json
)
