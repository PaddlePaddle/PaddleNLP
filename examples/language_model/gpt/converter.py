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

import sys
import paddle
import torch
import numpy as np

paddle.set_device("cpu")

model = torch.load(sys.argv[1], map_location='cpu')

print("The origin model keys:")
for x in sorted(list(model.keys())):
    print(x)

state = {}
for sub_name, sub_param in model.items():
    if sub_name.startswith("transformer"):
        sub_name = sub_name[12:-1]
    if sub_name.startswith("h."):
        final_name = sub_name.replace("h.", "gpt.decoder.layers.")
    else:
        final_name = sub_name
    state[final_name] = sub_param.numpy()


def trans_name(key):
    k = key
    k = k.replace("mlp.c_fc", "linear1")
    k = k.replace("mlp.c_proj", "linear2")
    k = k.replace("attn.c_proj", "self_attn.out_proj")
    k = k.replace("ln_1", "norm1")
    k = k.replace("ln_2", "norm2")
    k = k.replace("ln_f", "gpt.decoder.norm")
    k = k.replace("wte", "gpt.embeddings.word_embeddings")
    k = k.replace("wpe", "gpt.embeddings.position_embeddings")
    return k


new_state_dict = {}
all_num = 0
for key in sorted(list(state.keys())):
    all_num += state[key].size
    new_key = trans_name(key)
    if "attn.c_attn" in key:
        shape = state[key].shape
        print(shape)
        if "weight" in key:
            q, k, v = np.split(state[key], 3, axis=1)
        else:
            print("BIAS SHAPE", state[key].shape, state[key].transpose().shape)
            q, k, v = np.split(state[key], 3, axis=-1)
            q = q.reshape((-1))
            k = k.reshape((-1))
            v = v.reshape((-1))
        q_name = new_key.replace("attn.c_attn", "self_attn.q_proj")
        k_name = new_key.replace("attn.c_attn", "self_attn.k_proj")
        v_name = new_key.replace("attn.c_attn", "self_attn.v_proj")
        new_state_dict[q_name] = paddle.to_tensor(q, dtype="float32")
        new_state_dict[k_name] = paddle.to_tensor(k, dtype="float32")
        new_state_dict[v_name] = paddle.to_tensor(v, dtype="float32")
        continue
    new_state_dict[new_key] = paddle.to_tensor(state[key], dtype="float32")
print("all shape numel:{}".format(all_num))
for key, value in new_state_dict.items():
    print("key:{}, shape:{}, dtype:{}".format(key, value.shape, value.dtype))

orgin_path = sys.argv[1]
if ".bin" in orgin_path:
    save_path = orgin_path.replace(".bin", ".pdparams")
else:
    save_path = os.path.join(orgin_path, ".pdparams")
paddle.save(new_state_dict, save_path)
