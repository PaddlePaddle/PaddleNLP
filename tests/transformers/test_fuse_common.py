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


import copy
import glob
import os
import tempfile
import unittest

import paddle

input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])


def prepare_config(config):
    config.hidden_size = 512
    config.num_layers = 2
    config.num_hidden_layers = 2
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    config.intermediate_size = config.hidden_size * 3
    config.fuse_attention_qkv = False
    config.fuse_attention_ffn = False
    return config


def common_test_load(model, model_class, config, tempdir):
    model.eval()
    with paddle.no_grad():
        first = model(input_ids)[0]

    model_fused = model_class.from_pretrained(tempdir, config=config)

    model_fused.eval()
    with paddle.no_grad():
        second = model_fused(input_ids)[0]
    assert paddle.allclose(paddle.mean(first), paddle.mean(second), atol=1e-7)
    assert paddle.allclose(first, second, atol=1e-4)

    files = glob.glob(tempdir + "/*")
    for f in files:
        os.remove(f)


def common_test_save(model, config, model_class=None):
    config = copy.deepcopy(config)
    config.fuse_attention_qkv = True
    config.fuse_attention_ffn = True

    with tempfile.TemporaryDirectory() as tempdir:
        # test load pdparams: model.pdparams
        model.save_pretrained(save_dir=tempdir)
        common_test_load(model, model_class, config, tempdir)

        # test load shard pdparams: model-001-0f-008.pdparams
        model.save_pretrained(tempdir, max_shard_size="5MB")
        common_test_load(model, model_class, config, tempdir)

        # test save safetensors: model.safetensors
        model.save_pretrained(tempdir, safe_serialization=True)
        common_test_load(model, model_class, config, tempdir)

        # test load shard safetensors: model-001-0f-008.safetensors
        model.save_pretrained(tempdir, max_shard_size="5MB", safe_serialization=True)
        common_test_load(model, model_class, config, tempdir)


def _test_llama():
    from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig()
    config = prepare_config(config)
    model = LlamaForCausalLM.from_config(config)
    common_test_save(model, config, LlamaForCausalLM)


def _test_gpt():
    from paddlenlp.transformers import GPTConfig, GPTForCausalLM

    config = GPTConfig()
    config = prepare_config(config)
    model = GPTForCausalLM.from_config(config)
    common_test_save(model, config, GPTForCausalLM)


def _test_opt():
    from paddlenlp.transformers import OPTConfig, OPTForCausalLM

    config = OPTConfig()
    config = prepare_config(config)
    config.intermediate_size = 512
    config.word_embed_proj_dim = 512
    model = OPTForCausalLM.from_config(config)
    common_test_save(model, config, OPTForCausalLM)


class TestTensorParallel(unittest.TestCase):
    def test_model_load_merge(self):
        _test_llama()
        _test_gpt()
        _test_opt()
