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

import unittest

import paddle
from parameterized import parameterized_class

from paddlenlp.transformers import ChatGLMv2Config, ChatGLMv2ForCausalLM, ChatGLMv2Model
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    GenerationD2STestMixin,
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class ChatGLMv2Tester:
    def __init__(
        self,
        parent,
        is_training=True,
        num_hidden_layers=3,
        seq_length=10,
        batch_size=2,
        vocab_size=123,
        kv_channels=4,
        hidden_size=8,
        ffn_hidden_size=8,
        num_attention_heads=2,
        rmsnorm=True,
        use_cache=True,
    ):
        self.parent = parent
        self.is_training = is_training
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.kv_channels = kv_channels
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.rmsnorm = rmsnorm
        self.use_cache = use_cache

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")

        labels = None
        context_length = self.seq_length // 2
        if self.parent.use_labels:
            labels = paddle.ones([self.batch_size, self.seq_length], dtype=input_ids.dtype) * -100
            labels[:, context_length:] = input_ids[:, context_length:]

        config = self.get_config()
        return config, input_ids, labels

    def get_config(self):
        return ChatGLMv2Config(
            vocab_size=self.vocab_size,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            num_attention_heads=self.num_attention_heads,
            kv_channels=self.kv_channels,
            use_cache=self.use_cache,
            rmsnorm=self.rmsnorm,
        )

    def create_and_check_model(self, config, input_ids, labels):
        model = ChatGLMv2Model(config)
        model.eval()

        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.seq_length, self.batch_size, self.hidden_size])

    def create_and_check_model_past_large_inputs(self, config, input_ids, labels):
        model = ChatGLMv2Model(config)
        model.eval()

        outputs = model(input_ids, return_dict=self.parent.return_dict)
        past_key_values = outputs.past_key_values[0] if self.parent.return_dict else outputs[1][0]

        next_tokens = ids_tensor([self.batch_size, 3], self.vocab_size, dtype="int64")
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = model.get_masks(next_input_ids)

        outputs = model(next_input_ids, attention_mask=next_attention_mask, return_dict=self.parent.return_dict)
        output_from_no_past = outputs.past_key_values[0] if self.parent.return_dict else outputs[1][0]

        outputs = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            return_dict=self.parent.return_dict,
        )
        output_from_past = outputs.past_key_values[0] if self.parent.return_dict else outputs[1][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1], dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, labels = config_and_inputs
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict

    def create_and_check_lm_head_model(self, config, input_ids, labels, *args):
        model = ChatGLMv2ForCausalLM(config)
        model.eval()

        result = model(
            input_ids,
            labels=labels if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            loss = result.loss if self.parent.return_dict else result[0]
            self.parent.assertIsNotNone(loss)
            logits = result.logits if self.parent.return_dict else result[1]
            past_key_values = result.past_key_values[0] if self.parent.return_dict else result[2][0]
        else:
            loss = result.loss if self.parent.return_dict else None
            self.parent.assertIsNone(loss)
            logits = result.logits if self.parent.return_dict else result[0]
            past_key_values = result.past_key_values[0] if self.parent.return_dict else result[1][0]
        self.parent.assertEqual(logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        if config.use_cache:
            self.parent.assertTrue(isinstance(past_key_values, tuple))
            self.parent.assertEqual(
                past_key_values[0].shape,
                [self.seq_length, self.batch_size, config.multi_query_group_num, config.kv_channels],
            )
        else:
            self.parent.assertTrue(past_key_values is None)

    def create_and_check_model_attention_mask(self, config: ChatGLMv2Config, input_ids, labels):
        model = ChatGLMv2ForCausalLM(config)
        model.eval()
        attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length])
        result_2d = model(input_ids, attention_mask=attn_mask_2d)[0]
        batch, seq_length = input_ids.shape
        causal_mask = paddle.tril(paddle.ones((batch, seq_length, seq_length), dtype=attn_mask_2d.dtype))
        attn_mask_3d = causal_mask & attn_mask_2d.unsqueeze(-1)
        result_3d = model(input_ids, attention_mask=attn_mask_3d)[0]
        attn_mask_4d = attn_mask_3d.unsqueeze(1)
        result_4d = model(input_ids, attention_mask=attn_mask_4d)[0]
        result_no_attention_mask = model(input_ids, attention_mask=None)[0]
        # Assert non-padding tokens have the same logits with different attention_mask shape
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_3d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_4d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_no_attention_mask[attn_mask_2d]).all())


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, True],
        [True, False],
    ],
)
class ChatGLMv2Test(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = ChatGLMv2Model
    return_dict: bool = True
    use_labels: bool = False
    use_test_model_name_list = False

    all_model_classes = (ChatGLMv2Model, ChatGLMv2ForCausalLM)
    all_generative_model_classes = {ChatGLMv2ForCausalLM: (ChatGLMv2Model, "chatglm_v2")}

    def setUp(self):
        self.model_tester = ChatGLMv2Tester(self)

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict[self.input_name]
        print(input_ids)
        attention_mask = paddle.ones_like(input_ids)

        max_batch_size = 2
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:max_batch_size, :sequence_length]
        attention_mask = attention_mask[:max_batch_size, :sequence_length]

        # generate max 3 tokens
        max_length = 3

        return config, input_ids, attention_mask, max_length

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_ChatGLMv2_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_attention_mask(*config_and_inputs)


class ChatGLMV2GenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
    internal_testing_model = "__internal_testing__/tiny-random-chatglm2"


if __name__ == "__main__":
    unittest.main()
