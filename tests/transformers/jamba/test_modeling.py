# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the Paddle Jamba model."""

import math
import tempfile
import unittest

import paddle
from parameterized import parameterized

from paddlenlp.transformers import (
    AutoTokenizer,
    JambaConfig,
    JambaForCausalLM,
    JambaModel,
)
from paddlenlp.transformers.jamba.modeling import (
    FakeMLPForwardBackward,
    HybridMambaAttentionDynamicCache,
    get_triangle_upper_mask,
    is_autocast_enabled,
    is_casual_mask,
    repeat_kv,
)

from ...testing_utils import slow

# from ..generation import GenerationTesterMixin
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    ids_tensor,
    random_attention_mask,
)


class JambaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        attn_layer_offset=1,
        attn_layer_period=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_layer_offset = attn_layer_offset
        self.attn_layer_period = attn_layer_period
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return JambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            attn_layer_offset=self.attn_layer_offset,
            attn_layer_period=self.attn_layer_period,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=True,
            initializer_range=self.initializer_range,
            use_mamba_kernels=False,
            num_experts=2,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = JambaModel(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = JambaForCausalLM(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids, labels=token_labels)
        result = model(input_ids)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = JambaForCausalLM(config=config)
        model.eval()

        # first forward pass
        # Attention: Jamba needs the cache to be initialized to return a cache!
        past_key_values = HybridMambaAttentionDynamicCache(
            config,
            input_ids.shape[0],
            model._dtype,
        )
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True,)[
            "hidden_states"
        ][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            cache_position=paddle.arange(
                input_ids.shape[1],
                input_ids.shape[1] + next_tokens.shape[1],
            ),
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    # def create_and_check_for_sequence_classification(
    #     self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    # ):
    #     config.num_labels = self.num_labels
    #     model = JambaForSequenceClassification(config)
    #     model.eval()
    #     result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
    #     self.parent.assertEqual(result.logits.shape, [self.batch_size, self.num_labels])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class JambaModelTest(ModelTesterMixin, unittest.TestCase):
    use_test_model_name_list = False
    all_model_classes = (
        JambaModel,
        JambaForCausalLM,
    )
    all_generative_model_classes = (JambaForCausalLM,)

    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = JambaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=JambaConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_casual_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        pass
        # config_and_inputs = self.model_tester.prepare_config_and_inputs()
        # self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_experts = 16
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids != config.pad_token_id
        model = JambaForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        bs, seqlen = input_ids.shape
        self.assertEqual(result.router_logits[0].shape, [bs * seqlen, config.num_experts])
        self.assertTrue(
            paddle.allclose(result.aux_loss.cpu(), paddle.to_tensor(2, dtype=paddle.float32), rtol=1e-2, atol=1e-2)
        )

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        pad_length = 1000
        # Add padding tokens to input_ids
        padding_block = config.pad_token_id * paddle.ones([input_ids.shape[0], pad_length], dtype=paddle.int32)
        padded_input_ids = paddle.concat((padding_block, input_ids), axis=1)  # this is to simulate padding to the left
        # make sure that padded_input_ids dtype is int64
        padded_input_ids = padded_input_ids.cast("int64")
        padded_attention_mask = padded_input_ids != config.pad_token_id

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        self.assertTrue(paddle.allclose(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4))

        # We make sure that the loss of including padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic paddle.testing.assert_not_close
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())

    def test_initialization(self):
        r"""
        Overriding the test_initialization test as the A_log and D params of the Mamba block are initialized differently
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if not param.stop_gradient:
                    if "A_log" in name:
                        A = paddle.arange(1, config.mamba_d_state + 1, dtype=paddle.float32)[None, :]
                        self.assertTrue(
                            paddle.allclose(param.data, paddle.log(A).expand_as(param), atol=1e-5, rtol=1e-5)
                        )
                    elif "D" in name:
                        # check if it's a ones like
                        self.assertTrue(
                            paddle.allclose(param.data, paddle.ones_like(param.data), atol=1e-5, rtol=1e-5)
                        )
                    else:
                        if "lm_head.weight" in name:
                            continue
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        r"""
        Overriding the test_mismatched_shapes_have_properly_initialized_weights test because A_log and D params of the
        Mamba block are initialized differently and we tested that in test_initialization
        """
        self.skipTest("Cumbersome and redundant for Jamba")

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the Jamba model outputs attention only for its attention layers
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        expected_num_attentions = math.ceil(
            (self.model_tester.num_hidden_layers - self.model_tester.attn_layer_offset)
            / self.model_tester.attn_layer_period
        )

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.eval()

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_attentions)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    @slow
    def test_flash_attn_2_fp32_ln(self):
        r"""
        Overriding the test_flash_attn_2_fp32_ln test as the Jamba model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name]
                dummy_attention_mask = inputs_dict.get("attention_mask", paddle.ones_like(dummy_input))
                # NOTE: Jamba does not support right padding + use_cache with FA2.
                dummy_attention_mask[:, -1] = 1

                model = model_class.from_pretrained(
                    tmpdirname,
                    dtype="float16",
                    low_cpu_mem_usage=True,
                )
                model.config.use_flash_attention = True

                for _, param in model.named_parameters():
                    # upcast only layer norms
                    if (param.dtype == paddle.float16) or (param.dtype == paddle.bfloat16):
                        param.data = param.data.to(dtype=paddle.float32)

                _ = model(dummy_input)
                # with attention mask
                _ = model(dummy_input, attention_mask=dummy_attention_mask)

    @slow
    def test_flash_attn_2_generate_use_cache(self):
        r"""
        Overriding the test_flash_attn_2_generate_use_cache test as the Jamba model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """

        max_new_tokens = 30

        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [paddle.float32, paddle.bfloat16]:
                dummy_input = dummy_input.cast(paddle.float16)
            if dummy_input.dtype == paddle.int32:
                dummy_input = dummy_input.cast(paddle.int64)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_attention_mask = inputs_dict.get("attention_mask", paddle.ones_like(dummy_input))
                # NOTE: Jamba does not support right padding + use_cache with FA2.
                dummy_attention_mask[:, -1] = 1

                model = model_class.from_pretrained(
                    tmpdirname,
                    dtype="float16",
                    low_cpu_mem_usage=True,
                )
                model.config.use_flash_attention = True

                # Just test that a large cache works as expected
                _ = model.generate(
                    dummy_input,
                    attention_mask=dummy_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        r"""
        Overriding the test_flash_attn_2_inference_padding_right test as the Jamba model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """
        self.skipTest("Jamba flash attention does not support right padding")

    @unittest.skip("Jamba has its own special cache type")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass

    def test_config_num_key_value_heads_none(self):
        config = JambaConfig(num_attention_heads=4, num_key_value_heads=None)
        self.assertTrue(
            config.num_key_value_heads == config.num_attention_heads,
        )

    def test_is_autocast_enabled(self):
        self.assertFalse(is_autocast_enabled())

    def test_get_triangle_upper_mask(self):
        bsz = 1
        n_head = 2
        q_len = 10
        kv_seq_len = 16
        x = paddle.randn([bsz, n_head, q_len, kv_seq_len], dtype="float32")
        mask = paddle.randint(0, 2, [bsz, n_head, q_len, kv_seq_len], dtype="int64").cast("bool")
        tri_mask1 = get_triangle_upper_mask(x, mask=None)
        tri_mask2 = get_triangle_upper_mask(x, mask=mask)
        self.assertTrue(
            tri_mask1.shape == [bsz, 1, q_len, kv_seq_len],
        )
        self.assertTrue(paddle.equal_all(tri_mask2, mask))
        self.assertTrue(is_casual_mask(tri_mask1))
        self.assertFalse(is_casual_mask(tri_mask2))

    def test_repeat_kv(self):
        shape = [1, 2, 3, 4]
        hidden_states = paddle.randn(shape, dtype="float32")
        output = repeat_kv(hidden_states, n_rep=1)
        self.assertTrue(paddle.equal_all(hidden_states, output))

    def test_FakeMLPForwardBackward(self):
        x = paddle.randn([1, 2, 3], dtype="float32")
        x.stop_gradient = False
        gate_weight = paddle.randn([4, 5, 6], dtype="float32")
        gate_weight.stop_gradient = False
        up_weight = paddle.randn([1, 3, 5], dtype="float32")
        up_weight.stop_gradient = False
        down_weight = paddle.randn([2, 4, 6], dtype="float32")
        down_weight.stop_gradient = False
        out = FakeMLPForwardBackward.apply(x, gate_weight=gate_weight, up_weight=up_weight, down_weight=down_weight)
        loss = out.sum()
        loss.backward()
        self.assertTrue(
            loss == 0
            and x.grad.sum() == 0
            and gate_weight.grad.sum() == 0
            and up_weight.grad.sum() == 0
            and down_weight.grad.sum() == 0
        )

    def test_from_hf_hub(self):
        model_id = "ai21labs/Jamba-tiny-random"
        model = JambaForCausalLM.from_pretrained(model_id, dtype="bfloat16", from_hf_hub=True, convert_from_torch=True)
        self.assertTrue(model.config.vocab_size == 65536)


@slow
class JambaModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None

    @classmethod
    def setUpClass(cls):
        model_id = "ai21labs/Jamba-tiny-random"
        cls.model = JambaForCausalLM.from_pretrained(model_id, dtype="bfloat16", low_cpu_mem_usage=True)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @slow
    def test_simple_generate(self):

        input_ids = self.tokenizer("Hey how are you doing on this lovely evening?", return_tensors="pd")["input_ids"]
        out = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(out[0][0, :])
        self.assertEqual(
            output_sentence.strip(),
            "Canyon rins hugaughter glamour Rutgers Singh Hebrew cases Cats",
        )

        with paddle.no_grad():
            logits = self.model(input_ids=input_ids).logits

        EXPECTED_LOGITS_NO_GRAD = paddle.to_tensor(
            [
                0.0118, -0.2256, 0.0376, -0.0996, 0.0457, 0.2773, -0.1455, 0.1650,
                -0.2910, -0.0261, 0.0240, -0.5586, -0.2139, -0.1406, -0.1582, 0.1318,
                0.0684, 0.2217, 0.1699, -0.2275, -0.1182, -0.1157, -0.1387, 0.0272,
                0.1245, 0.2334, 0.0425, 0.1099, -0.1348, -0.2305, 0.1445, -0.3945,
                0.1768, -0.4570, -0.0439, 0.2412, 0.1553, -0.1914, 0.2383, -0.0593
            ]
            , dtype=paddle.float32)  # fmt: skip
        self.assertTrue(
            paddle.allclose(logits[0, -1, :40].cast("float32").cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1e-3)
        )

    @slow
    def test_simple_batched_generate_with_padding(self):

        inputs = self.tokenizer(
            ["Hey how are you doing on this lovely evening?", "Tell me a story"], padding=True, return_tensors="pd"
        )
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_sentences = self.tokenizer.batch_decode(out[0], skip_special_tokens=False)
        self.assertEqual(
            output_sentences[0].strip(),
            "Canyon rins hugaughter glamour Rutgers Singh Hebrew cases Cats",
        )
        self.assertEqual(
            output_sentences[1].strip(),
            "ptus Nets Madison El chamadamodern updximVaparsed",
        )

        with paddle.no_grad():
            logits = self.model(input_ids=inputs["input_ids"]).logits

        EXPECTED_LOGITS_NO_GRAD_0 = paddle.to_tensor(
            [
                0.0148, -0.2246, 0.0403, -0.1006, 0.0452, 0.2734, -0.1465, 0.1641,
                -0.2930, -0.0256, 0.0259, -0.5586, -0.2119, -0.1406, -0.1621, 0.1348,
                0.0679, 0.2227, 0.1719, -0.2305, -0.1162, -0.1167, -0.1396, 0.0262,
                0.1299, 0.2314, 0.0408, 0.1118, -0.1338, -0.2324, 0.1436, -0.3906,
                0.1748, -0.4570, -0.0449, 0.2412, 0.1572, -0.1914, 0.2363, -0.0630
            ]
            , dtype=paddle.float32)  # fmt: skip

        EXPECTED_LOGITS_NO_GRAD_1 = paddle.to_tensor(
            [
                -0.1338, 0.2363, -0.4160, -0.0280, -0.0422, 0.0303, 0.2578, 0.0859,
                0.1465, 0.2236, -0.1162, -0.1406, -0.1484, -0.1079, -0.0045, -0.2812,
                0.1982, -0.2676, 0.0559, -0.2002, -0.2559, -0.1182, -0.2012, 0.2148,
                0.0532, 0.1699, 0.1797, 0.1309, 0.1699, -0.1226, -0.2695, -0.2891,
                0.2344, 0.2637, 0.0479, -0.1807, 0.2178, -0.1260, 0.1797, 0.0046
            ]
            , dtype=paddle.float32)  # fmt: skip

        self.assertTrue(
            paddle.allclose(logits[0, -1, :40].cast("float32").cpu(), EXPECTED_LOGITS_NO_GRAD_0, rtol=1e-3, atol=1e-3)
        )
        self.assertTrue(
            paddle.allclose(logits[1, -1, :40].cast("float32").cpu(), EXPECTED_LOGITS_NO_GRAD_1, rtol=1e-3, atol=1e-3)
        )
