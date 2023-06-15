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

import numpy as np
import paddle

from paddlenlp.transformers import (
    ChatGLMConfig,
    ChatGLMForConditionalGeneration,
    ChatGLMModel,
)
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import ModelTesterMixin, ids_tensor


class ChatGLMTester:
    def __init__(
        self,
        parent,
        vocab_size=130528,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        layernorm_epsilon=1e-5,
        use_cache=False,
        bos_token_id=130004,
        eos_token_id=130005,
        pad_token_id=3,
        mask_token_id=130000,
        gmask_token_id=130001,
        max_sequence_length=10,
        inner_hidden_size=256,
        position_encoding_2d=True,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        output_predict=True,
        recompute=False,
        attention_scale=True,
        activation="gelu",
        batch_size: int = 2,
        seq_length: int = 10,
        num_image_tokens=0,
        use_labels: bool = False,
        return_dict=False,
    ):
        self.parent: ChatGLMTest = parent
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.layernorm_epsilon = layernorm_epsilon
        self.inner_hidden_size = inner_hidden_size
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.gmask_token_id = gmask_token_id
        self.position_encoding_2d = position_encoding_2d
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.output_predict = output_predict
        self.recompute = recompute
        self.attention_scale = attention_scale
        self.activation = activation
        self.num_image_tokens = num_image_tokens

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_labels = use_labels
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids[input_ids == self.gmask_token_id] = self.mask_token_id
        input_ids[input_ids == self.bos_token_id] = self.mask_token_id

        context_length = np.random.randint(1, self.seq_length - 2)
        input_ids[:, context_length - 2] = self.gmask_token_id
        input_ids[:, context_length - 1] = self.bos_token_id

        labels = None
        if self.use_labels:
            labels = paddle.ones([self.batch_size, self.seq_length]) * -100
            labels[:, context_length:] = input_ids[:, context_length:]

        config = self.get_config()
        return config, input_ids, labels

    def get_config(self):
        return ChatGLMConfig(
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_sequence_length=self.max_sequence_length,
            layernorm_epsilon=self.layernorm_epsilon,
            inner_hidden_size=self.inner_hidden_size,
            use_cache=self.use_cache,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
            gmask_token_id=self.gmask_token_id,
            position_encoding_2d=self.position_encoding_2d,
            quantization_bit=self.quantization_bit,
            pre_seq_len=self.pre_seq_len,
            prefix_projection=self.prefix_projection,
            output_predict=self.output_predict,
            recompute=self.recompute,
            attention_scale=self.attention_scale,
            activation=self.activation,
            num_image_tokens=self.num_image_tokens,
        )

    def create_and_check_model(self, config, input_ids, labels):
        model = ChatGLMModel(config)
        model.eval()

        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.seq_length, self.batch_size, self.hidden_size])

    def create_and_check_model_past_large_inputs(self, config, input_ids, labels):
        model = ChatGLMModel(config)
        model.eval()

        outputs = model(input_ids, return_dict=self.return_dict)
        past_key_values = outputs.past_key_values[0] if self.return_dict else outputs[1][0]

        next_tokens = ids_tensor([self.batch_size, 3], self.vocab_size)
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = model.get_masks(next_input_ids)

        outputs = model(next_input_ids, attention_mask=next_attention_mask, return_dict=self.return_dict)
        output_from_no_past = outputs.past_key_values[0] if self.return_dict else outputs[1][0]

        outputs = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            return_dict=self.return_dict,
        )
        output_from_past = outputs.past_key_values[0] if self.return_dict else outputs[1][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
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
        model = ChatGLMForConditionalGeneration(config)
        model.eval()

        result = model(
            input_ids,
            labels=labels if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            loss = result.loss if self.parent.return_dict else result[0]
            self.parent.assertEqual(loss.shape, [1])
            logits = result.logits if self.parent.return_dict else result[1]
            past_key_values = result.past_key_values[0] if self.parent.return_dict else result[2][0]
        else:
            loss = result.loss if self.parent.return_dict else None
            self.parent.assertTrue(loss is None)
            logits = result.logits if self.parent.return_dict else result[0]
            past_key_values = result.past_key_values[0] if self.parent.return_dict else result[1][0]
        self.parent.assertEqual(logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        if config.use_cache:
            self.parent.assertTrue(isinstance(past_key_values, tuple))
            self.parent.assertEqual(past_key_values[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        else:
            self.parent.assertTrue(past_key_values is None)


class ChatGLMTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ChatGLMModel
    return_dict = False
    use_labels = False

    all_model_classes = (ChatGLMModel, ChatGLMForConditionalGeneration)

    def setUp(self):
        super().setUp()

        self.model_tester = ChatGLMTester(self)
        self.config_tester = ConfigTester(self, config_class=ChatGLMConfig, vocab_size=256, hidden_size=24)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_name_list(self):
        pass

    def test_resize_tokens_embeddings(self):
        pass

    def test_chatglm_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)


if __name__ == "__main__":
    unittest.main()
