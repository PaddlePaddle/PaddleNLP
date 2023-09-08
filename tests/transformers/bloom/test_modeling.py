# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import copy
import math
import random
import tempfile
import unittest

import numpy as np
import paddle
import pytest
from parameterized import parameterized, parameterized_class

from paddlenlp.transformers import (
    BloomConfig,
    BloomForCausalLM,
    BloomForSequenceClassification,
    BloomForTokenClassification,
    BloomModel,
    BloomTokenizer,
)
from paddlenlp.transformers.bloom.modeling import BloomForGeneration
from tests.testing_utils import PaddleNLPModelTest, require_package, slow
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (  # GenerationD2STestMixin,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


class BloomModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=20,
        is_training=False,
        use_input_mask=True,
        vocab_size=100,
        hidden_size=32,
        n_layer=2,
        n_head=8,
        masked_softmax_fusion=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        pretraining_tp=1,  # TP rank used when training with megatron
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = self.n_head = n_head
        self.num_hidden_layers = self.n_layer = n_layer
        self.n_head = n_head
        self.masked_softmax_fusion = masked_softmax_fusion
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.pretraining_tp = pretraining_tp
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.type_sequence_label_size = type_sequence_label_size
        self.scope = None
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 3

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length], dtype="int64")

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size, dtype="int64")
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels, dtype="int64")
            choice_labels = ids_tensor([self.batch_size], self.num_choices, dtype="int64")

        config = self.get_config()

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return BloomConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            masked_softmax_fusion=self.masked_softmax_fusion,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            apply_residual_connection_post_layernorm=self.apply_residual_connection_post_layernorm,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            attention_softmax_in_fp32=self.attention_softmax_in_fp32,
            pretraining_tp=self.pretraining_tp,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
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

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = paddle.cast(
            ids_tensor([self.batch_size, self.seq_length], vocab_size=2), dtype="float32"
        )

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_gpt_model(self, config, input_ids, input_mask, *args):
        model = BloomModel(config)
        model.eval()

        result = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)
        result = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)
        result = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(len(result[1]), config["n_layer"])

    def create_and_check_gpt_model_past(self, config, input_ids, input_mask, *args):
        model = BloomModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, use_cache=False, return_dict=self.parent.return_dict)
        outputs_use_cache_conf = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)

        self.parent.assertTrue(len(outputs) + 1 == len(outputs_use_cache_conf))

        output, past = outputs_use_cache_conf[:2]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config["vocab_size"], dtype="int64")

        # append to next input_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids, return_dict=self.parent.return_dict)[0]
        output_from_past = model(next_tokens, use_cache=True, cache=past, return_dict=self.parent.return_dict)[0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_gpt_model_attention_mask_past(self, config, input_ids, input_mask, *args):
        model = BloomModel(config)
        model.eval()

        # create attention mask
        attn_mask = paddle.ones(input_ids.shape, dtype="float32")
        half_seq_length = self.seq_length // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past = model(input_ids, attention_mask=attn_mask, use_cache=True, return_dict=self.parent.return_dict)[
            :2
        ]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config["vocab_size"], dtype="int64")

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length, dtype="int64").item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config["vocab_size"], dtype="int64").squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        attn_mask = paddle.concat(
            [attn_mask, paddle.ones((attn_mask.shape[0], 1), dtype="float32")],
            axis=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask, return_dict=self.parent.return_dict)[0]
        output_from_past = model(
            next_tokens, cache=past, use_cache=True, attention_mask=attn_mask, return_dict=self.parent.return_dict
        )[0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1], dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_gpt_model_past_large_inputs(self, config, input_ids, input_mask, *args):
        model = BloomModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True, return_dict=self.parent.return_dict)

        output, past = outputs[:2]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config["vocab_size"], dtype="int64")
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2, dtype="int64")

        # append to next input_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)

        output_from_no_past = model(
            next_input_ids, attention_mask=next_attention_mask, return_dict=self.parent.return_dict
        )[0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            cache=past,
            use_cache=True,
            return_dict=self.parent.return_dict,
        )[0]
        self.parent.assertTrue(output_from_past.shape[1] == next_tokens.shape[1])

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1], dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
        model = BloomForCausalLM(config)
        model.eval()

        result = model(
            input_ids,
            use_cache=True,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_forward_and_backwards(self, config, input_ids, input_mask, *args):
        model = BloomForCausalLM(config)

        if self.parent.use_labels:
            loss, logits = model(input_ids, labels=input_ids, return_dict=self.parent.return_dict)
            self.parent.assertEqual(loss.shape, [1])
            self.parent.assertEqual(logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
            loss.backward()

    def create_and_check_gpt_for_sequence_classification(self, config, input_ids, input_mask, sequence_labels, *args):
        config.num_labels = self.num_labels
        model = BloomForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            labels=sequence_labels if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.num_labels])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_labels])

    def create_and_check_gpt_for_token_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, *args
    ):
        config.num_labels = self.num_labels
        model = BloomForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            labels=token_labels if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.num_labels])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_labels])

    def create_and_check_gpt_weight_initialization(self, config, *args):
        model = BloomModel(config)
        model_std = model.config["initializer_range"] / math.sqrt(2 * model.config["n_layer"])
        for key in model.state_dict().keys():
            if "out_proj" in key and "weight" in key:
                self.parent.assertLessEqual(abs((paddle.std(model.state_dict()[key]) - model_std).numpy()), 0.02)
                self.parent.assertLessEqual(abs((paddle.mean(model.state_dict()[key]) - 0.0).numpy()), 0.01)

    def create_and_check_model_attention_mask(
        self, config: BloomConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BloomModel(config)
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

        inputs_dict = {
            "input_ids": input_ids,
        }

        return config, inputs_dict

    def prepare_config_and_inputs_for_gpt(self):
        config = self.get_config()
        # excluding eos_token_id which is equal to vocab_size - 1
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1, dtype="int64")
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


@parameterized_class(
    ("return_dict", "use_labels"),
    [[False, False], [False, True], [True, False], [True, True]],
)
class BloomModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = BloomModel
    use_labels = False
    return_dict = False
    use_test_model_name_list = False

    all_model_classes = (BloomModel, BloomForCausalLM, BloomForSequenceClassification, BloomForTokenClassification)
    all_generative_model_classes = {BloomForCausalLM: (BloomModel, "bloom")}

    all_parallelizable_model_classes = BloomForCausalLM
    test_missing_keys = False
    test_tie_weights = False
    test_model_parallel = True

    # special case for DoubleHeads model
    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class)

        return inputs_dict

    def setUp(self):
        self.model_tester = BloomModelTester(self)
        self.test_resize_embeddings = False
        random.seed(128)
        np.random.seed(128)
        paddle.seed(128)

    def test_gpt_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_model(*config_and_inputs)

    def test_gpt_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_model_past(*config_and_inputs)

    def test_gpt_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_model_attention_mask_past(*config_and_inputs)

    def test_gpt_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_model_past_large_inputs(*config_and_inputs)

    def test_gpt_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_gpt_sequence_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_for_sequence_classification(*config_and_inputs)

    def test_gpt_token_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_for_token_classification(*config_and_inputs)

    def test_gpt_weight_initialization(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_weight_initialization(*config_and_inputs)

    def test_model_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_attention_mask(*config_and_inputs)

    def test_inputs_embeds(self):
        # NOTE: rewrite test inputs embeds for gpt model since couldn't detect eos token id from inputs_embeds
        # get config for model and inputs_dict for model forward
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_gpt()
        # test all model classes
        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            with paddle.no_grad():
                ids_output = model(**inputs)

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with paddle.no_grad():
                embeds_output = model(**inputs)

            self.assertTrue(paddle.allclose(ids_output, embeds_output, rtol=1e-4, atol=1e-4))


class BloomCompatibilityTest(unittest.TestCase):
    test_model_id = "hf-internal-testing/tiny-random-BloomModel"

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import BloomModel

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        model = BloomModel.from_pretrained(cls.test_model_id)
        model.save_pretrained(cls.torch_model_path)

    @parameterized.expand(
        [
            ("BloomModel", "BloomModel"),
            ("BloomForSequenceClassification", "BloomForSequenceClassification"),
            ("BloomForTokenClassification", "BloomForTokenClassification"),
            ("BloomForCausalLM", "BloomForCausalLM"),
        ]
    )
    @require_package("transformers", "torch")
    def test_gpt_classes_from_local_dir(self, paddle_class_name, pytorch_class_name=None):
        pytorch_class_name = pytorch_class_name or paddle_class_name
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create commmon input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch model
            import torch
            import transformers

            torch_model_class = getattr(transformers, pytorch_class_name)
            torch_model = torch_model_class.from_pretrained(self.torch_model_path)
            torch_model.eval()

            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 3. forward the paddle model
            from paddlenlp import transformers

            paddle_model_class = getattr(transformers, paddle_class_name)
            paddle_model = paddle_model_class.from_pretrained(tempdir, convert_from_torch=True)
            paddle_model.eval()

            paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=False)[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().numpy().reshape([-1])[:9],
                    torch_logit.detach().cpu().numpy().reshape([-1])[:9],
                    atol=1e-3,
                )
            )


class BloomModelLanguageGenerationTest(PaddleNLPModelTest):
    def _test_lm_generate_gpt_helper(
        self,
        verify_outputs=True,
    ):
        model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
        model.eval()

        # The dog
        input_ids = paddle.to_tensor([[464, 3290]], dtype="int64")

        # The dog was found in a field near the intersection of West and West Streets.\n\nThe dog
        # fmt: off
        expected_output_ids = [
            373,
            1043,
            287,
            257,
            2214,
            1474,
            262,
            16246,
            286,
            2688,
            290,
            2688,
            27262,
            13,
            198,
            198,
            464,
            3290,
        ]
        # fmt: on
        output_ids, _ = model.generate(input_ids, decode_strategy="greedy_search", max_length=18)
        if verify_outputs:
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @pytest.mark.skip("compelte `generate` method in another pr")
    @slow
    def test_lm_generate_gpt(self):
        self._test_lm_generate_gpt_helper()

    @slow
    def test_gpt_for_generation(self):
        model_name = "bigscience/bloom-560m"
        tokenizer = BloomTokenizer.from_pretrained(model_name)

        config = BloomConfig.from_pretrained(model_name)
        config.top_k = 1
        model = BloomForGeneration.from_pretrained(model_name, config=config)
        model.eval()

        paddle.seed(128)
        np.random.seed(128)
        random.seed(128)

        tokenized = tokenizer("I love you,", return_tensors="pd")
        input_ids = tokenized["input_ids"]

        output_ids, _ = model(
            input_ids,
        )
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(output_str)

        output_seq, _ = model(input_ids=input_ids)
        output_seq_strs = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        print(output_seq_strs)

        EXPECTED_OUTPUT_STR = " baby.\nI love you, baby.\nI love you, baby.\nI love you, baby.\n"

        self.assertEqual(output_seq_strs[0], EXPECTED_OUTPUT_STR)
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)

    @pytest.mark.skip("compelte `generate` method in another pr")
    @slow
    def test_gpt_sample(self):
        tokenizer = BloomTokenizer.from_pretrained("bigscience/bloom-560m")
        model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
        model.eval()

        paddle.seed(128)
        np.random.seed(128)
        random.seed(128)

        tokenized = tokenizer("where is the captial of china: ", return_tensors="pd")
        input_ids = tokenized["input_ids"]

        output_ids, _ = model.generate(
            input_ids,
            top_k=1,
        )
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(output_str)

        output_seq, _ = model.generate(
            input_ids=input_ids,
            top_k=1,
        )
        output_seq_strs = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        print(output_seq_strs)

        EXPECTED_OUTPUT_STR = "the result is not accurate with BloomForGeneration."

        self.assertEqual(output_seq_strs[0], EXPECTED_OUTPUT_STR)
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)


# class BloomGenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
#    max_length = 100
#    internal_testing_model = "__internal_testing__/tiny-random-bloom"
