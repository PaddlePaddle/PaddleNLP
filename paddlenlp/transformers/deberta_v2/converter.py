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

import numpy as np
import paddle
import torch
from transformers import AutoModel

from .configuration import DebertaConfig
from .modeling import DebertaModel as PaddleDebertaModel

hf_model = AutoModel.from_pretrained("microsoft/deberta-v2-xlarge")
pp_model = PaddleDebertaModel(DebertaConfig())

paddle_state_dict = {}
skip_weights = ["embeddings.position_ids"]

for k, v in hf_model.named_parameters():
    if k in skip_weights:
        continue
    if v.ndim == 2 and "embeddings" not in k:
        v = v.transpose(0, 1)
    print(f"{k}: {list(v.shape)}")
    paddle_state_dict[k] = paddle.to_tensor(v.data.numpy())
pp_model.set_dict(paddle_state_dict)

paddle.save(paddle_state_dict, "./model_state.pdparams")

pp_parameters = paddle.load("./model_state.pdparams")
pp_model.set_dict(pp_parameters)

input_ids = np.random.randint(1, 1000, size=(2, 10))
pp_inputs = paddle.to_tensor(input_ids)
hf_inputs = torch.tensor(input_ids)
pp_model.eval()
hf_model.eval()

pp_output = pp_model(pp_inputs, output_hidden_states=True)
hf_output = hf_model(hf_inputs, output_hidden_states=True)

for i in range(len(pp_model.encoder.layer) + 1):
    diff = abs(hf_output["hidden_states"][i].detach().numpy() - pp_output["hidden_states"][i].numpy())
    print(f"layer {i} max diff: {np.max(diff)}, min diff: {np.min(diff)}")
