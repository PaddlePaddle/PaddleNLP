# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import math
import unittest
from typing import Dict, List, Tuple
from unittest.util import safe_repr

import paddle

from paddlenlp.transformers import (
    AutoTokenizer,
    MambaConfig,
    MambaForCausalLM,
    MambaModel,
)
from paddlenlp.transformers.mamba.modeling import MambaCache

from ...testing_utils import slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import ModelTesterMixin, ids_tensor


class MambaModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        tie_word_embeddings=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1
        self.tie_word_embeddings = tie_word_embeddings

    def get_large_model_config(self):
        return MambaConfig.from_pretrained("state-spaces/mamba-2.8b-hf")

    def prepare_config_and_inputs(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config(
            gradient_checkpointing=gradient_checkpointing,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )

        return (
            config,
            input_ids,
            None,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        return MambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            activation_function=self.hidden_act,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        return (
            config,
            input_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def create_and_check_mamba_model(self, config, input_ids, *args):
        config.output_hidden_states = True
        model = MambaModel(config=config)
        model.eval()

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(len(result.hidden_states), config.num_hidden_layers + 1)

    def create_and_check_causal_lm(self, config, input_ids, *args):
        model = MambaForCausalLM(config)
        model.eval()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.ndim, 0)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_state_equivalency(self, config, input_ids, *args):
        model = MambaModel(config=config)
        model.eval()

        outputs = model(input_ids)
        output_whole = outputs.last_hidden_state

        outputs = model(
            input_ids[:, :-1],
            use_cache=True,
        )
        output_one = outputs.last_hidden_state

        # Using the state computed on the first inputs, we will get the same output
        outputs = model(
            input_ids[:, -1:],
            use_cache=True,
            cache=outputs.cache,
        )
        output_two = outputs.last_hidden_state

        self.parent.assertTrue(
            paddle.allclose(paddle.concat([output_one, output_two], axis=1), output_whole, atol=1e-5)
        )

    def create_and_check_mamba_cached_slow_forward_and_backwards(
        self, config, input_ids, *args, gradient_checkpointing=False
    ):
        model = MambaModel(config)
        if gradient_checkpointing:
            model.enable_recompute = gradient_checkpointing

        # create cache
        cache = model(input_ids, use_cache=True).cache
        cache.reset()
        # use cache
        token_emb = model.embeddings(input_ids)
        outputs = model.layers[0].mixer.slow_forward(
            token_emb,
            cache,
        )

        loss = paddle.log(1 + paddle.abs(outputs.sum()))
        self.parent.assertEqual(loss.ndim, 0)
        self.parent.assertEqual(outputs.shape, [self.batch_size, self.seq_length, self.hidden_size])
        loss.backward()

    def create_and_check_mamba_lm_head_forward_and_backwards(
        self, config, input_ids, *args, gradient_checkpointing=False
    ):
        model = MambaForCausalLM(config)
        if gradient_checkpointing:
            model.backbone.enable_recompute = gradient_checkpointing

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.ndim, 0)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        result.loss.backward()

    def prepare_config_and_inputs_for_common(self):
        (
            config,
            input_ids,
            _,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


class MambaModelTest(ModelTesterMixin, unittest.TestCase):
    use_test_model_name_list = False
    all_model_classes = (MambaModel, MambaForCausalLM)
    all_generative_model_classes = (MambaForCausalLM,)
    has_attentions = False  # Mamba does not support attentions
    fx_compatible = False  # FIXME let's try to support this @ArthurZucker
    test_torchscript = False  # FIXME let's try to support this @ArthurZucker
    test_missing_keys = False
    test_model_parallel = False
    test_pruning = False
    test_head_masking = False  # Mamba does not have attention heads

    def setUp(self):
        self.model_tester = MambaModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=MambaConfig, n_embd=37, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def assertInterval(self, member, container, msg=None):
        r"""
        Simple utility function to check if a member is inside an interval.
        """
        if isinstance(member, paddle.Tensor):
            max_value, min_value = member.max().item(), member.min().item()
        elif isinstance(member, list) or isinstance(member, tuple):
            max_value, min_value = max(member), min(member)

        if not isinstance(container, list):
            raise TypeError("container should be a list or tuple")
        elif len(container) != 2:
            raise ValueError("container should have 2 elements")

        expected_min, expected_max = container

        is_inside_interval = (min_value >= expected_min) and (max_value <= expected_max)

        if not is_inside_interval:
            standardMsg = "%s not found in %s" % (safe_repr(member), safe_repr(container))
            self.fail(self._formatMessage(msg, standardMsg))

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mamba_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba_model(*config_and_inputs)

    def test_mamba_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm(*config_and_inputs)

    # def test_state_equivalency(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_state_equivalency(*config_and_inputs)

    def test_mamba_cached_slow_forward_and_backwards(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba_cached_slow_forward_and_backwards(*config_and_inputs)

    def test_mamba_lm_head_forward_and_backwards(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba_lm_head_forward_and_backwards(*config_and_inputs)

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, param in model.named_parameters():
                if "dt_proj.bias" in name:
                    dt = paddle.exp(
                        paddle.to_tensor([0, 1]) * (math.log(config.time_step_max) - math.log(config.time_step_min))
                        + math.log(config.time_step_min)
                    ).clip(min=config.time_step_floor)
                    inv_dt = dt + paddle.log(-paddle.expm1(-dt))
                    if not param.stop_gradient:
                        self.assertTrue(param.data.max().item() <= inv_dt[1])
                        self.assertTrue(param.data.min().item() >= inv_dt[0])
                    elif "A_log" in name:
                        A = paddle.arange(1, config.mamba_d_state + 1, dtype=paddle.float32)[None, :]
                        self.assertTrue(
                            paddle.allclose(param.data, paddle.log(A).expand_as(param), atol=1e-5, rtol=1e-5)
                        )
                    elif "D" in name:
                        # check if it's a ones like
                        self.assertTrue(
                            paddle.allclose(param.data, paddle.ones_like(param.data), atol=1e-5, rtol=1e-5)
                        )

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with paddle.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, MambaCache):  # MODIFIED PART START
                        recursive_check(tuple_object.conv_states, dict_object.conv_states)
                        recursive_check(tuple_object.ssm_states, dict_object.ssm_states)
                    elif isinstance(tuple_object, (List, Tuple)):  # MODIFIED PART END
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            paddle.allclose(tuple_object, dict_object, atol=1e-5),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {paddle.max(paddle.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {paddle.isnan(tuple_object).any()} and `inf`: {paddle.isinf(tuple_object)}. Dict has"
                                f" `nan`: {paddle.isnan(dict_object).any()} and `inf`: {paddle.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            # tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            # dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            # check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            # tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            # dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            # check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})


class MambaIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model_id = "state-spaces/mamba-2.8b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @slow
    def test_simple_generate(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "state-spaces/mamba-130m-hf",
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", dtype="float16")
        input_ids = tokenizer("Hey how are you doing?", return_tensors="pd")["input_ids"]

        out = model.generate(input_ids, do_sample=False, use_cache=True, max_new_tokens=10)[0]
        output_sentence = tokenizer.decode(out[0, :])
        self.assertEqual(output_sentence.strip(), "I'm so glad you're here.")

        with paddle.no_grad():
            logits = model(input_ids=input_ids).logits

        EXPECTED_LOGITS_NO_GRAD = paddle.to_tensor(
            [
                -55.5938, -69.7500, -49.8438, -51.6875, -57.5938, -57.8750, -56.9062,
                -57.8438, -54.6250, -55.8438, -55.2500, -57.9688, -60.5000, -46.9688,
                -52.    , -49.7188, -55.9062, -57.8438, -56.6875, -57.0312, -57.2500,
                -58.2188, -57.7188, -58.7188, -59.5000, -59.    , -58.6250, -52.8438,
                -53.3750, -57.2812, -56.8438, -55.6250, -53.2500, -55.7188, -56.9375,
                -56.8438, -56.1562, -54.6562, -56.3438, -57.4062
            ], dtype=paddle.float32)  # fmt: skip

        self.assertTrue(paddle.allclose(logits[0, 0, :40].cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1e-3))

    @slow
    def test_simple_generate_cuda_kernels_tiny(self):
        expected_output = "John and I am a newbie to the world"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pd").input_ids
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", dtype="float16")

        output = model.generate(input_ids, max_new_tokens=10)[0]
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence.strip(), expected_output)

    @slow
    def test_simple_generate_cuda_kernels_small(self):
        expected_output = "I am a\n\nI am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pd").input_ids
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-790m-hf", dtype="float16")

        output = model.generate(input_ids, max_new_tokens=10)[0]
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence.strip(), expected_output)

    @slow
    def test_simple_generate_cuda_kernels_mid(self):
        expected_output = "John and I am a\n\nI am a single father of a beautiful daughter. I am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pd").input_ids
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf", dtype="float16")

        output = model.generate(input_ids, max_new_tokens=20)[0]
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence.strip(), expected_output)

    @slow
    def test_simple_generate_cuda_kernels_big(self):
        expected_output = "John and I am a new member of this forum. I am a retired Marine and I am a member of the Marine Corps League. I am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pd").input_ids
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf", dtype="float16")

        output = model.generate(input_ids, max_new_tokens=30)[0]
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence.strip(), expected_output)

    @slow
    def test_compile_mamba_cache(self):
        expected_output = "John and I am a\n\nI am a single father of a beautiful daughter. I am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pd").input_ids
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf", dtype="float16")

        output = model.generate(input_ids, max_new_tokens=20, cache_implementation="mamba")[0]
        output_sentence = self.tokenizer.decode(output[0].tolist())
        self.assertEqual(output_sentence.strip(), expected_output)
