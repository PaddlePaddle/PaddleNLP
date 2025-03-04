# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers.deepseek_v2.configuration import DeepseekV2Config
from paddlenlp.transformers.deepseek_v2.modeling import DeepseekV2DecoderLayer

config = DeepseekV2Config.from_pretrained("/root/.paddlenlp/models/deepseek-ai/DeepSeek-V3/config.json")
config.num_hidden_layers = 5
config.first_k_dense_replace = 6
config.intermediate_size = 2048
config.use_flash_attention = 1
config.use_fused_rope = 1
config.use_fused_rms_norm = 0
config.use_fast_layer_norm = 1
seq_len = 4096
print("Final pre-training config:", config.to_dict())


class DemoLayer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dec = DeepseekV2DecoderLayer(config, 0)

    def forward(self, hidden_states, position_ids):
        hidden_states = self.dec(hidden_states, position_ids)
        return hidden_states


hidden_states = paddle.randn(shape=[1, seq_len, config.hidden_size], dtype=paddle.bfloat16)
position_ids = paddle.arange(0, seq_len, dtype=paddle.int64)
position_ids = position_ids.unsqueeze(0)

model = DemoLayer(config)
beta1 = paddle.to_tensor([0.9], dtype="float32")
beta2 = paddle.to_tensor([0.99], dtype="float32")
optimizer = paddle.optimizer.AdamW(
    learning_rate=0.1, parameters=model.parameters(), beta1=beta1, beta2=beta2, weight_decay=0.01
)

model, optimizer = paddle.amp.decorate(models=model, optimizers=optimizer, level="O2", dtype="bfloat16")

for step in range(20):
    print("===> run step ", step, flush=1)

    # if step == 5:
    #     paddle.base.core.nvprof_start()
    #     paddle.base.core.nvprof_enable_record_event()
    #     paddle.base.core.nvprof_nvtx_push(str(step))
    # if step == 10:
    #     paddle.base.core.nvprof_nvtx_pop()
    #     paddle.base.core.nvprof_stop()
    #     import sys
    #     sys.exit()
    # if step >= 5 and step < 10:
    #     paddle.base.core.nvprof_nvtx_pop()
    #     paddle.base.core.nvprof_nvtx_push(str(step))

    with paddle.amp.auto_cast(
        dtype="bfloat16", enable=True, custom_white_list=None, custom_black_list=None, level="O2"
    ):
        out = model(hidden_states, position_ids)
        loss = paddle.mean(out)
        print("loss: ", loss)
    paddle.autograd.backward(loss, retain_graph=True)
