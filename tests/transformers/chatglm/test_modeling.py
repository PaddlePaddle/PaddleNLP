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
from __future__ import annotations

import unittest

import numpy as np
import paddle

from paddlenlp.transformers import ChatGLMConfig, ChatGLMForCausalLM, ChatGLMModel
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


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
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")
        input_ids[input_ids == self.gmask_token_id] = self.mask_token_id
        input_ids[input_ids == self.bos_token_id] = self.mask_token_id

        context_length = np.random.randint(1, self.seq_length - 2)
        input_ids[:, context_length - 2] = self.gmask_token_id
        input_ids[:, context_length - 1] = self.bos_token_id

        attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)
        attention_mask = attention_mask.unsqueeze([1, 2])
        attention_mask = attention_mask * attention_mask.transpose([0, 1, 3, 2])

        MASK, gMASK = self.mask_token_id, self.gmask_token_id
        use_gmasks = []
        mask_positions = []
        context_lengths = []
        for seq in input_ids:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            use_gmasks.append(use_gmask)
            mask_positions.append(paddle.where(seq == mask_token)[0][0])
            context_lengths.append(context_length)

        position_ids = paddle.arange(self.seq_length, dtype="int64").unsqueeze(0).tile([self.batch_size, 1])
        for i, context_length in enumerate(context_lengths):
            position_ids[i, context_length:] = mask_positions[i]

        block_position_ids = [
            paddle.concat(
                (
                    paddle.zeros([context_length], dtype="int64"),
                    paddle.arange(self.seq_length - context_length, dtype="int64") + 1,
                )
            )
            for context_length in context_lengths
        ]
        block_position_ids = paddle.stack(block_position_ids, axis=0)
        position_ids = paddle.stack((position_ids, block_position_ids), axis=1)

        labels = None
        if self.use_labels:
            labels = paddle.ones([self.batch_size, self.seq_length]) * -100
            labels[:, context_length:] = input_ids[:, context_length:]

        config = self.get_config()
        return config, input_ids, labels, attention_mask, position_ids

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

    def create_and_check_model(self, config, input_ids, labels, attention_mask, position_ids):
        model = ChatGLMModel(config)
        model.eval()

        result = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        self.parent.assertEqual(result[0].shape, [self.seq_length, self.batch_size, self.hidden_size])

    # def create_and_check_model_past_large_inputs(self, config, input_ids, labels):
    #    model = ChatGLMModel(config)
    #    model.eval()

    #    outputs = model(input_ids, attention_mask=attention_mask, return_dict=self.return_dict)
    #    past_key_values = outputs.past_key_values[0] if self.return_dict else outputs[1][0]

    #    next_tokens = ids_tensor([self.batch_size, 3], self.vocab_size)
    #    next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
    #    next_attention_mask = model.get_masks(next_input_ids)

    #    outputs = model(next_input_ids, attention_mask=next_attention_mask, return_dict=self.return_dict)
    #    output_from_no_past = outputs.past_key_values[0] if self.return_dict else outputs[1][0]

    #    outputs = model(
    #        next_tokens,
    #        attention_mask=next_attention_mask,
    #        past_key_values=past_key_values,
    #        return_dict=self.return_dict,
    #    )
    #    output_from_past = outputs.past_key_values[0] if self.return_dict else outputs[1][0]

    #    # select random slice
    #    random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
    #    output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
    #    output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

    #    self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

    #    # test that outputs are equal for slice
    #    self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, labels, attention_mask, position_ids = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
        return config, inputs_dict

    def create_and_check_lm_head_model(self, config, input_ids, labels, attention_mask, position_ids):
        model = ChatGLMForCausalLM(config)
        model.eval()

        result = model(
            input_ids,
            attention_mask=attention_mask,
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

    def create_and_check_model_attention_mask(self, config: ChatGLMConfig, input_ids, labels):
        model = ChatGLMModel(config)
        model.eval()
        attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length])
        result_2d = model(input_ids, attention_mask=attn_mask_2d)[0].transpose([1, 0, 2])
        batch, seq_length = input_ids.shape
        causal_mask = paddle.tril(paddle.ones((batch, seq_length, seq_length), dtype=attn_mask_2d.dtype))
        attn_mask_3d = causal_mask & attn_mask_2d.unsqueeze(-1)
        result_3d = model(input_ids, attention_mask=attn_mask_3d)[0].transpose([1, 0, 2])

        # use 4d mask for chatglm must prepocess prefix mask and padding mask
        attn_mask_4d = attn_mask_3d.unsqueeze(1)
        context_lengths, pad_lengths = [], []
        for seq in input_ids:
            context_lengths.append(paddle.where(seq == self.bos_token_id)[0][0])
            pad_lengths.append(paddle.where(seq != self.pad_token_id)[0][0])

        for i, context_length in enumerate(context_lengths):
            attn_mask_4d[i, :, :, :context_length] = 1
        print(attn_mask_4d)

        for i, pad_length in enumerate(pad_lengths):
            attn_mask_4d[i, :pad_length, :pad_length] = 0
        print(attn_mask_4d)

        result_4d = model(input_ids, attention_mask=attn_mask_4d)[0].transpose([1, 0, 2])
        result_no_attention_mask = model(input_ids, attention_mask=None)[0].transpose([1, 0, 2])
        # Assert non-padding tokens have the same logits with different attention_mask shape
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_3d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_4d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_no_attention_mask[attn_mask_2d]).all())


class ChatGLMTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = ChatGLMModel
    return_dict = False
    use_labels = False

    all_model_classes = (ChatGLMModel, ChatGLMForCausalLM)
    all_generative_model_classes = {ChatGLMForCausalLM: (ChatGLMModel, "chatglm")}

    def setUp(self):
        super().setUp()

        self.model_tester = ChatGLMTester(self)
        self.config_tester = ConfigTester(self, config_class=ChatGLMConfig, vocab_size=256, hidden_size=24)

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict[self.input_name]
        attention_mask = inputs_dict["attention_mask"]
        position_ids = inputs_dict["position_ids"]

        max_batch_size = 2
        sequence_length = input_ids.shape[-1]
        input_ids = input_ids[:max_batch_size, :sequence_length]
        attention_mask = attention_mask[:max_batch_size, :, :sequence_length, :sequence_length]
        position_ids = position_ids[:max_batch_size, :, :sequence_length]

        # generate max 3 tokens
        max_length = 3

        if config.eos_token_id or config.pad_token_id:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            config["pad_token_id"] = config["eos_token_id"]

        return config, input_ids, attention_mask, max_length

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_name_list(self):
        pass

    def test_group_beam_search_generate(self):
        pass

    def test_beam_search_generate(self):
        pass

    def test_generate_without_input_ids(self):
        pass

    def test_resize_tokens_embeddings(self):
        pass

    def test_chatglm_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    # def test_model_attention_mask(self):
    #    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #    self.model_tester.create_and_check_model_attention_mask(*config_and_inputs)


# class ChatGLMGenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
#     internal_testing_model = "__internal_testing__/tiny-random-chatglm"
#     TokenizerClass = ChatGLMTokenizer
#     CausalLMClass = ChatGLMForCausalLM

#     def test_to_static_use_top_k(self):
#         tokenizer = self.TokenizerClass.from_pretrained(self.internal_testing_model)
#         model = self.CausalLMClass.from_pretrained(self.internal_testing_model)

#         model_kwargs = tokenizer(
#             self.article,
#             max_length=self.max_length,
#             truncation=True,
#             truncation_side="left",
#             return_tensors="pd",
#             padding=True,
#             add_special_tokens=True,
#         )

#         model.eval()

#         # Llama model do not contians ``
#         model.is_encoder_decoder = False

#         max_length = self.max_length

#         model_kwargs["use_cache"] = True
#         model_kwargs["max_length"] = max_length + model_kwargs["input_ids"].shape[-1]

#         decoded_ids = model.greedy_search(
#             logits_processors=None,
#             bos_token_id=model.config.bos_token_id,
#             pad_token_id=model.config.pad_token_id,
#             eos_token_id=model.config.eos_token_id,
#             **model_kwargs,
#         )[0]

#         dygraph_decoded_ids = decoded_ids.tolist()

#         with static_mode_guard():
#             with tempfile.TemporaryDirectory() as tempdir:
#                 path = os.path.join(tempdir, "model")
#                 model.to_static(
#                     path,
#                     config=dict(
#                         use_top_p=False,
#                     ),
#                 )

#                 model_path = os.path.join(tempdir, "model.pdmodel")
#                 params_path = os.path.join(tempdir, "model.pdiparams")
#                 config = paddle.inference.Config(model_path, params_path)

#                 config.disable_gpu()
#                 config.disable_glog_info()
#                 predictor = paddle.inference.create_predictor(config)

#                 model_kwargs["top_k"] = 1
#                 model_kwargs["attention_mask"] = model_kwargs["attention_mask"].astype("int64")
#                 model_kwargs["max_length"] = self.max_length
#                 # create input
#                 for key in model_kwargs.keys():
#                     if paddle.is_tensor(model_kwargs[key]):
#                         model_kwargs[key] = model_kwargs[key].numpy()
#                     elif isinstance(model_kwargs[key], float):
#                         model_kwargs[key] = np.array(model_kwargs[key], dtype="float32")
#                     else:
#                         model_kwargs[key] = np.array(model_kwargs[key], dtype="int64")

#                 input_handles = {}
#                 for name in predictor.get_input_names():
#                     input_handles[name] = predictor.get_input_handle(name)

#                     input_handles[name].copy_from_cpu(model_kwargs[name])

#                 predictor.run()
#                 output_names = predictor.get_output_names()
#                 output_handle = predictor.get_output_handle(output_names[0])
#                 results = output_handle.copy_to_cpu()

#                 static_decoded_ids = results.tolist()

#         self.assertEqual(dygraph_decoded_ids, static_decoded_ids)

#     def test_to_static_use_top_p(self):
#         tokenizer = self.TokenizerClass.from_pretrained(self.internal_testing_model)
#         model = self.CausalLMClass.from_pretrained(self.internal_testing_model)

#         model_kwargs = tokenizer(
#             self.article,
#             max_length=self.max_length,
#             truncation=True,
#             truncation_side="left",
#             return_tensors="pd",
#             padding=True,
#             add_special_tokens=True,
#         )

#         model.eval()

#         # Llama model do not contians ``
#         model.is_encoder_decoder = False

#         max_length = self.max_length

#         model_kwargs["use_cache"] = True
#         model_kwargs["max_length"] = max_length + model_kwargs["input_ids"].shape[-1]

#         with static_mode_guard():
#             with tempfile.TemporaryDirectory() as tempdir:

#                 path = os.path.join(tempdir, "model")
#                 model.to_static(
#                     path,
#                     config=dict(
#                         use_top_p=False,
#                     ),
#                 )

#                 model_path = os.path.join(tempdir, "model.pdmodel")
#                 params_path = os.path.join(tempdir, "model.pdiparams")
#                 config = paddle.inference.Config(model_path, params_path)

#                 config.disable_gpu()
#                 config.disable_glog_info()
#                 predictor = paddle.inference.create_predictor(config)

#                 model_kwargs["attention_mask"] = model_kwargs["attention_mask"].astype("int64")
#                 model_kwargs["top_k"] = 1
#                 model_kwargs["max_length"] = self.max_length
#                 # create input
#                 for key in model_kwargs.keys():
#                     if paddle.is_tensor(model_kwargs[key]):
#                         model_kwargs[key] = model_kwargs[key].numpy()
#                     else:
#                         model_kwargs[key] = np.array(model_kwargs[key])

#                 input_handles = {}
#                 for name in predictor.get_input_names():
#                     input_handles[name] = predictor.get_input_handle(name)
#                     input_handles[name].copy_from_cpu(model_kwargs[name])

#                 predictor.run()
#                 output_names = predictor.get_output_names()
#                 output_handle = predictor.get_output_handle(output_names[0])
#                 results = output_handle.copy_to_cpu()

#         self.assertIsNotNone(results)


if __name__ == "__main__":
    unittest.main()
