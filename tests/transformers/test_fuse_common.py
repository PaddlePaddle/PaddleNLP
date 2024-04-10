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


import glob
import os
import tempfile
import unittest


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


def common_test_load(model_class, config, tempdir):
    config.fuse_attention_qkv = True
    config.fuse_attention_ffn = True
    model_class.from_pretrained(tempdir, config=config)
    files = glob.glob(tempdir + "/*")
    for f in files:
        os.remove(f)


def common_test_save(model, config, model_class=None):
    with tempfile.TemporaryDirectory() as tempdir:
        # test load pdparams: model.pdparams
        model.save_pretrained(save_dir=tempdir)
        common_test_load(model_class, config, tempdir)

        # test load shard pdparams: model-001-0f-008.pdparams
        model.save_pretrained(tempdir, max_shard_size="5MB")
        common_test_load(model_class, config, tempdir)

        # test save safetensors: model.safetensors
        model.save_pretrained(tempdir, safe_serialization=True)
        common_test_load(model_class, config, tempdir)

        # test load shard safetensors: model-001-0f-008.safetensors
        model.save_pretrained(tempdir, max_shard_size="5MB", safe_serialization=True)
        common_test_load(model_class, config, tempdir)


def _test_llama():
    from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig()
    config = prepare_config(config)
    model = LlamaForCausalLM.from_config(config)
    common_test_save(model, config, LlamaForCausalLM)


def _test_chatglm():
    from paddlenlp.transformers import ChatGLMConfig, ChatGLMForCausalLM

    config = ChatGLMConfig()
    config = prepare_config(config)
    model = ChatGLMForCausalLM.from_config(config)
    common_test_save(model, ChatGLMForCausalLM)


def _test_bloom():
    from paddlenlp.transformers import BloomConfig, BloomForCausalLM

    config = BloomConfig()
    config = prepare_config(config)
    model = BloomForCausalLM.from_config(config)
    common_test_save(model, BloomForCausalLM)


class TestTensorParallel(unittest.TestCase):
    def test_model_load_merge(self):
        _test_llama()
        # _test_chatglm()
        # _test_bloom()


import unittest

suit = unittest.TestSuite()
suit.addTest(TestTensorParallel("test_model_load_merge"))
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suit)
