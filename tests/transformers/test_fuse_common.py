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


def _test_fast_ffn():
    from functools import partial

    import paddle
    from paddle import nn

    from paddlenlp.transformers import PretrainedModel
    from paddlenlp.transformers.configuration_utils import PretrainedConfig

    class TestConfig(PretrainedConfig):
        def __init__(self, fast_ffn_state=False, convert_fast_ffn=False):
            self.fast_ffn_state = fast_ffn_state
            self.convert_fast_ffn = convert_fast_ffn
            super().__init__()

    class TestMLP(nn.Layer):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.hidden_size = config.hidden_size
            self.gate_up_fused_proj = nn.Linear(self.hidden_size, self.hidden_size * 2, bias_attr=True)

        def forward(self, hidden_state):
            hidden_state = self.gate_up_fused_proj(hidden_state)
            if self.config.use_fast_ffn:
                x, y = paddle.chunk(hidden_state, chunks=2, axis=-1)
            else:
                x, y = hidden_state[..., ::2], hidden_state[..., 1::2]

            return nn.functional.silu(x) * y

    class TestPretrainedModel(PretrainedModel):
        config_class = TestConfig

        @classmethod
        def _get_fuse_or_split_param_mappings(cls, config: TestConfig, is_fuse=False):
            # return parameter fuse utils
            from paddlenlp.transformers.conversion_utils import split_or_fuse_func

            fn = split_or_fuse_func(is_fuse=is_fuse)

            convert_fast_ffn_keys = (
                "layers.0.gate_up_fused_proj.weight",
                "layers.0.gate_up_fused_proj.weight",
            )
            convert_fast_ffn_bias_keys = (
                "layers.0.gate_up_fused_proj.bias",
                "layers.0.gate_up_fused_proj.bias",
            )
            fast_ffn_state = getattr(config, "fast_ffn_state", False)
            convert_fast_ffn = getattr(config, "convert_fast_ffn", False)
            convert_fast_ffn &= not fast_ffn_state

            final_actions = {}
            if is_fuse:
                # use_fast_ffn only in one condition
                # convert when use_fast_ffn is False
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in convert_fast_ffn_keys])
                    final_actions[keys] = partial(fn, convert_fast_ffn=convert_fast_ffn)
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in convert_fast_ffn_bias_keys])
                    final_actions[keys] = partial(fn, convert_fast_ffn=convert_fast_ffn)
            return final_actions

        def _init_weights(self, layer):
            if isinstance(layer, (nn.Linear, nn.Embedding)):
                if isinstance(layer.weight, paddle.Tensor):
                    layer.weight.set_value(paddle.tensor.normal(mean=0.0, std=1.0, shape=layer.weight.shape))
                if hasattr(layer, "bias") and isinstance(layer.bias, paddle.Tensor):
                    layer.bias.set_value(paddle.tensor.normal(mean=0.0, std=1.0, shape=layer.bias.shape))

    class TestModel(TestPretrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.layers = nn.LayerList([TestMLP(config=config) for i in range(config.num_hidden_layers)])

        def forward(self, hidden_state):
            for idx, (decoder_layer) in enumerate(self.layers):
                hidden_state = decoder_layer(hidden_state)
            return hidden_state

    class TestForCausalLM(TestPretrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.embedding_layer = nn.Embedding(65535, self.config.hidden_size)
            self.test = TestModel(config=config)

        def forward(self, input_ids):
            hidden_state = self.embedding_layer(input_ids)
            return self.test(hidden_state)

    config = TestConfig()
    config = prepare_config(config)
    config.use_fast_ffn = False
    model = TestForCausalLM.from_config(config)

    config = copy.deepcopy(config)
    config.use_fast_ffn = True
    config.fast_ffn_state = False
    config.convert_fast_ffn = True

    with tempfile.TemporaryDirectory() as tempdir:
        # test load pdparams: model.pdparams
        model.save_pretrained(save_dir=tempdir)
        common_test_load(model, TestForCausalLM, config, tempdir)

        # test load shard pdparams: model-001-0f-008.pdparams
        model.save_pretrained(tempdir, max_shard_size="5MB")
        common_test_load(model, TestForCausalLM, config, tempdir)

        # test save safetensors: model.safetensors
        model.save_pretrained(tempdir, safe_serialization=True)
        common_test_load(model, TestForCausalLM, config, tempdir)

        # test load shard safetensors: model-001-0f-008.safetensors
        model.save_pretrained(tempdir, max_shard_size="5MB", safe_serialization=True)
        common_test_load(model, TestForCausalLM, config, tempdir)


class TestFuseOrSplit(unittest.TestCase):
    def test_model_load_merge(self):
        _test_llama()
        _test_gpt()
        _test_opt()
        _test_fast_ffn()
