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

from paddlenlp.transformers import GLMModel


def func(self, *args, **kwargs):
    return


GLMModel.init_weights = func

# with paddle.LazyGuard():
paddle.set_default_dtype("float16")
model = GLMModel.from_pretrained(
    "glm-10b",
    load_state_as_np=True,
    dtype="float16",
    paddle_dtype="float16",
)

model.eval()
ret = model(input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]))

# torch 2.1089835166931152
print("paddle mp", ret[0].abs().mean().item())
# print("paddle mp", ret.logits.abs().mean().item())
