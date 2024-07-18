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

""" Yuan2.0 model tools"""

import torch

def rearrange_model_weights(model_path, save_path, tp_degree, hidden_layers):

    print("load yuan weights ......")
    model = torch.load(model_path)
    print("load yuan weights finish ......")

    size = model["model.layers.0.self_attn.q_proj.weight"].shape[0]
    step = size // tp_degree
    for i in range(0, hidden_layers):

        q = model[f"model.layers.{i}.self_attn.q_proj.weight"]
        k = model[f"model.layers.{i}.self_attn.k_proj.weight"]
        q_slices = [q[i:i+step,:] for i in range(0, size, step)]
        k_slices = [k[i:i+step,:] for i in range(0, size, step)]
        q1=torch.cat(q_slices[0::2],0)
        q2=torch.cat(k_slices[0::2],0)
        k1=torch.cat(q_slices[1::2],0)
        k2=torch.cat(k_slices[1::2],0)

        model[f"model.layers.{i}.self_attn.q_proj.weight"]=torch.cat([q1,q2],0)
        model[f"model.layers.{i}.self_attn.k_proj.weight"]=torch.cat([k1,k2],0)
        print(i, " layer is finished ......")

    torch.save(model, save_path)
    print("Model weights saved.")


# Example usage:
rearrange_model_weights(
    model_path="/workspace/pytorch_model.bin",
    save_path="/workspace/yuan_paddle_tp2/pytorch_model.bin",
    tp_degree=2,  # set tensor parallel degree
    hidden_layers=24,  # set the number of hidden_layers, from config.json
)
