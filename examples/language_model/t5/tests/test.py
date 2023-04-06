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

# from transformers import T5Tokenizer, T5Model, AutoModel

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5Model.from_pretrained("t5-small", trust_remote_code=True)

# input_ids = tokenizer(
#     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
# ).input_ids  # Batch size 1
# print(input_ids)
# decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
# print(decoder_input_ids)
# # forward pass
# outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)
# import numpy as np
# from paddlenlp.transformers import AutoModel
# import paddle
# model = AutoModel.from_pretrained("THUDM/glm-large-chinese")
# model.eval()
# loss = model(input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]))
# print(loss)
# print(loss.logits.shape)
# print(loss.logits)
# ret = loss.logits.abs().mean().item()
# print(ret)
# np.testing.assert_allclose(ret, 3.9399895668029785, rtol=1e-7)


# from transformers import AutoModel
# import numpy as np
# import torch
# model = AutoModel.from_pretrained("t5-small", trust_remote_code=True)
# model.eval()
# loss = model(input_ids=torch.arange(100, 110, dtype=torch.long).reshape(1, -1), decoder_input_ids=torch.arange(100, 105, dtype=torch.long).reshape(1, -1))
# ret = loss.last_hidden_state.abs().mean().item()
#         # Torch T5 has bug in GELU activation
# print(ret)
# np.testing.assert_allclose(ret, 0.1365441530942917, rtol=1e-7)

import numpy as np
import paddle

from paddlenlp.transformers import AutoModel

model = AutoModel.from_pretrained("t5-small", from_hf_hub=True)
model.eval()
loss = model(
    input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]),
    decoder_input_ids=paddle.arange(100, 105, dtype="int64").reshape([1, -1]),
    return_dict=True,
)
ret = loss.last_hidden_state.abs().mean().item()
print(ret)
np.testing.assert_allclose(ret, 2.109480381011963, rtol=1e-7)
