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

import random
import unittest

import paddle

from paddlenlp.transformers import (
    XLNetForMultipleChoice,
    XLNetForQuestionAnswering,
    XLNetForSequenceClassification,
    XLNetForTokenClassification,
    XLNetLMHeadModel,
    XLNetPretrainedModel,
    XLNetModel,
)
from ..test_modeling_common import ids_tensor, floats_tensor, random_attention_mask, ModelTesterMixin
from ...testing_utils import slow


class XLNetModelTester:

    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 14
        self.seq_length = 7
        self.mem_len = 10
        # self.key_len = seq_length + mem_len
        self.clamp_len = -1
        self.reuse_len = 15
        self.is_training = True
        self.use_labels = True
        self.vocab_size = 99
        self.cutoffs = [10, 50, 80]
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.d_inner = 128
        self.num_hidden_layers = 5
        self.type_sequence_label_size = 2
        self.bi_data = False
        self.same_length = False
        self.initializer_range = 0.05
        self.seed = 1
        self.type_vocab_size = 2
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 5
        self.num_choices = 4

    def prepare_config_and_inputs(self):
        input_ids_1 = ids_tensor([self.batch_size, self.seq_length],
                                 self.vocab_size)
        input_ids_2 = ids_tensor([self.batch_size, self.seq_length],
                                 self.vocab_size)
        segment_ids = ids_tensor([self.batch_size, self.seq_length],
                                 self.type_vocab_size)
        input_mask = random_attention_mask([self.batch_size, self.seq_length])

        input_ids_q = ids_tensor([self.batch_size, self.seq_length + 1],
                                 self.vocab_size)
        perm_mask = paddle.zeros(
            [self.batch_size, self.seq_length + 1, self.seq_length + 1])
        perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        target_mapping = paddle.zeros([
            self.batch_size,
            1,
            self.seq_length + 1,
        ])
        target_mapping[:, 0, -1] = 1.0  # predict last token

        config = self.get_config()

        return (
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
        )

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.hidden_size,
            "n_head": self.num_attention_heads,
            "d_inner": self.d_inner,
            "n_layer": self.num_hidden_layers,
            "mem_len": self.mem_len,
            "clamp_len": self.clamp_len,
            "same_length": self.same_length,
            "reuse_len": self.reuse_len,
            "bi_data": self.bi_data,
            "initializer_range": self.initializer_range,
            # "bos_token_id": self.bos_token_id,
            # "pad_token_id": self.pad_token_id,
            # "eos_token_id": self.eos_token_id,
        }

    def set_seed(self):
        random.seed(self.seed)
        paddle.seed(self.seed)

    def create_and_check_xlnet_base_model(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
    ):
        model = XLNetModel(**config)
        model.eval()

        result = model(input_ids_1, input_mask=input_mask)
        result = model(input_ids_1, attention_mask=input_mask)
        result = model(input_ids_1, token_type_ids=segment_ids)
        result = model(input_ids_1, return_dict=True)

        config["mem_len"] = 0
        model = XLNetModel(**config)
        model.eval()
        base_model_output = model(input_ids_1, return_dict=True)
        self.parent.assertEqual(len(base_model_output), 4)

        self.parent.assertEqual(
            result["last_hidden_state"].shape,
            [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_use_mems_train(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
    ):
        model = XLNetForSequenceClassification(XLNetModel(**config))
        model.train()

        train_size = input_ids_1.shape[0]

        batch_size = 4
        for i in range(train_size // batch_size + 1):
            input_ids = input_ids_1[i:(i + 1) * batch_size]
            outputs = model(input_ids=input_ids, return_dict=True)
            self.parent.assertIsNone(outputs["mems"])

    def create_and_check_xlnet_base_model_with_att_output(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
    ):
        model = XLNetModel(**config)
        model.eval()

        attentions = model(input_ids_1,
                           target_mapping=target_mapping,
                           return_dict=True)["attentions"]

        self.parent.assertEqual(len(attentions), config["n_layer"])
        self.parent.assertIsInstance(attentions[0], tuple)
        self.parent.assertEqual(len(attentions[0]), 2)
        self.parent.assertTrue(attentions[0][0].shape, attentions[0][0].shape)

    def create_and_check_xlnet_lm_head(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
    ):
        model = XLNetLMHeadModel(XLNetModel(**config))
        model.eval()

        result1 = model(input_ids_1,
                        token_type_ids=segment_ids,
                        return_dict=True)
        result2 = model(input_ids_2,
                        token_type_ids=segment_ids,
                        mems=result1["mems"],
                        return_dict=True)

        _ = model(input_ids_q,
                  perm_mask=perm_mask,
                  target_mapping=target_mapping)

        self.parent.assertEqual(
            result1["logits"].shape,
            [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertEqual(
            result2["logits"].shape,
            [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_xlnet_qa(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
    ):
        model = XLNetForQuestionAnswering(XLNetModel(**config))
        model.eval()

        result = model(input_ids_1)

        result_with_mask = model(
            input_ids_1,
            input_mask=input_mask,
        )

        self.parent.assertEqual(result[0].shape,
                                [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[1].shape,
                                [self.batch_size, self.seq_length])

    def create_and_check_xlnet_token_classif(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
    ):
        model = XLNetForTokenClassification(XLNetModel(**config))
        model.eval()

        result = model(input_ids_1, return_dict=True)
        self.parent.assertEqual(
            result["logits"].shape,
            [self.batch_size, self.seq_length, self.type_sequence_label_size])

    def create_and_check_xlnet_sequence_classif(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
    ):
        model = XLNetForSequenceClassification(XLNetModel(**config))
        model.eval()

        result = model(input_ids_1, return_dict=True)

        self.parent.assertEqual(
            result["logits"].shape,
            [self.batch_size, self.type_sequence_label_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids_1}
        return config, inputs_dict


class XLNetModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = XLNetModel
    all_model_classes = (
        XLNetModel,
        XLNetLMHeadModel,
        XLNetForTokenClassification,
        XLNetForSequenceClassification,
        XLNetForQuestionAnswering,
        XLNetForMultipleChoice,
    )

    def setUp(self):
        self.model_tester = XLNetModelTester(self)

    def test_xlnet_base_model(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_base_model(*config_and_inputs)

    def test_seq_classification_use_mems_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_use_mems_train(*config_and_inputs)

    def test_xlnet_base_model_with_att_output(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_base_model_with_att_output(
            *config_and_inputs)

    def test_xlnet_lm_head(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_lm_head(*config_and_inputs)

    def test_xlnet_sequence_classif(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_sequence_classif(
            *config_and_inputs)

    def test_xlnet_token_classif(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_token_classif(
            *config_and_inputs)

    def test_xlnet_qa(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_qa(*config_and_inputs)

    def test_retain_grad_hidden_states_attentions(self):
        # xlnet cannot keep gradients in attentions or hidden states
        return

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)

        for param in [
                "q", "k", "v", "o", "r", "r_r_bias", "r_s_bias", "r_w_bias",
                "seg_embed", "mask_emb"
        ]:
            if hasattr(module, param) and getattr(module, param) is not None:
                weight = getattr(module, param)
                weight.data.fill_(3)

    def _check_hidden_states_for_generate(self,
                                          batch_size,
                                          hidden_states,
                                          min_length,
                                          max_length,
                                          config,
                                          use_cache=False,
                                          num_beam_groups=1):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [
                isinstance(iter_hidden_states, tuple)
                for iter_hidden_states in hidden_states
            ],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states),
                         (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            # check hidden size
            for i, layer_hidden_states in enumerate(iter_hidden_states):
                # every 2nd tensor is from extra stream
                if i % 2 != 0:
                    seq_len = 1
                else:
                    # for first item dummy PAD token is appended so need one more
                    seq_len = (min_length + 1) if idx == 0 else min_length

                expected_shape = (batch_size * num_beam_groups, seq_len,
                                  config.hidden_size)
                self.assertEqual(layer_hidden_states.shape, expected_shape)

    def _check_attentions_for_generate(self,
                                       batch_size,
                                       attentions,
                                       min_length,
                                       max_length,
                                       config,
                                       use_cache=False,
                                       num_beam_groups=1):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual([
            isinstance(iter_attentions, tuple) for iter_attentions in attentions
        ], [True] * len(attentions))
        self.assertEqual(len(attentions),
                         (max_length - min_length) * num_beam_groups)

        for idx, attentions_item in enumerate(attentions):
            for iter_attentions in attentions_item:
                tgt_len = min_length

                # for first item dummy PAD token is appended so need one more
                if idx == 0:
                    tgt_len += 1

                src_len = min_length + idx + 1

                expected_shape = (
                    batch_size * num_beam_groups,
                    config.num_attention_heads,
                    tgt_len,
                    src_len,
                )
                # check attn size
                self.assertListEqual(
                    [
                        layer_attention.shape
                        for layer_attention in iter_attentions
                    ],
                    [expected_shape] * len(iter_attentions),
                )

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                XLNetPretrainedModel.pretrained_init_configuration)[:1]:
            model = XLNetModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class XLNetModelLanguageGenerationTest(unittest.TestCase):

    @slow
    def test_lm_generate_xlnet_base_cased(self):
        model = XLNetLMHeadModel.from_pretrained("xlnet-base-cased")
        # fmt: off
        input_ids = paddle.to_tensor([[
            67,
            2840,
            19,
            18,
            1484,
            20,
            965,
            29077,
            8719,
            1273,
            21,
            45,
            273,
            17,
            10,
            15048,
            28,
            27511,
            21,
            4185,
            11,
            41,
            2444,
            9,
            32,
            1025,
            20,
            8719,
            26,
            23,
            673,
            966,
            19,
            29077,
            20643,
            27511,
            20822,
            20643,
            19,
            17,
            6616,
            17511,
            18,
            8978,
            20,
            18,
            777,
            9,
            19233,
            1527,
            17669,
            19,
            24,
            673,
            17,
            28756,
            150,
            12943,
            4354,
            153,
            27,
            442,
            37,
            45,
            668,
            21,
            24,
            256,
            20,
            416,
            22,
            2771,
            4901,
            9,
            12943,
            4354,
            153,
            51,
            24,
            3004,
            21,
            28142,
            23,
            65,
            20,
            18,
            416,
            34,
            24,
            2958,
            22947,
            9,
            1177,
            45,
            668,
            3097,
            13768,
            23,
            103,
            28,
            441,
            148,
            48,
            20522,
            19,
            12943,
            4354,
            153,
            12860,
            34,
            18,
            326,
            27,
            17492,
            684,
            21,
            6709,
            9,
            8585,
            123,
            266,
            19,
            12943,
            4354,
            153,
            6872,
            24,
            3004,
            20,
            18,
            9225,
            2198,
            19,
            12717,
            103,
            22,
            401,
            24,
            6348,
            9,
            12943,
            4354,
            153,
            1068,
            2768,
            2286,
            19,
            33,
            104,
            19,
            176,
            24,
            9313,
            19,
            20086,
            28,
            45,
            10292,
            9,
            4,
            3,
        ]], )
        # fmt: on
        #  In 1991, the remains of Russian Tsar Nicholas II and his family
        #  (except for Alexei and Maria) are discovered.
        #  The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
        #  remainder of the story. 1883 Western Siberia,
        #  a young Grigori Rasputin is asked by his father and a group of men to perform magic.
        #  Rasputin has a vision and denounces one of the men as a horse thief. Although his
        #  father initially slaps him for making such an accusation, Rasputin watches as the
        #  man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
        #  the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
        #  with people, even a bishop, begging for his blessing. """

        # fmt: off
        expected_output_ids = [
            67,
            2840,
            19,
            18,
            1484,
            20,
            965,
            29077,
            8719,
            1273,
            21,
            45,
            273,
            17,
            10,
            15048,
            28,
            27511,
            21,
            4185,
            11,
            41,
            2444,
            9,
            32,
            1025,
            20,
            8719,
            26,
            23,
            673,
            966,
            19,
            29077,
            20643,
            27511,
            20822,
            20643,
            19,
            17,
            6616,
            17511,
            18,
            8978,
            20,
            18,
            777,
            9,
            19233,
            1527,
            17669,
            19,
            24,
            673,
            17,
            28756,
            150,
            12943,
            4354,
            153,
            27,
            442,
            37,
            45,
            668,
            21,
            24,
            256,
            20,
            416,
            22,
            2771,
            4901,
            9,
            12943,
            4354,
            153,
            51,
            24,
            3004,
            21,
            28142,
            23,
            65,
            20,
            18,
            416,
            34,
            24,
            2958,
            22947,
            9,
            1177,
            45,
            668,
            3097,
            13768,
            23,
            103,
            28,
            441,
            148,
            48,
            20522,
            19,
            12943,
            4354,
            153,
            12860,
            34,
            18,
            326,
            27,
            17492,
            684,
            21,
            6709,
            9,
            8585,
            123,
            266,
            19,
            12943,
            4354,
            153,
            6872,
            24,
            3004,
            20,
            18,
            9225,
            2198,
            19,
            12717,
            103,
            22,
            401,
            24,
            6348,
            9,
            12943,
            4354,
            153,
            1068,
            2768,
            2286,
            19,
            33,
            104,
            19,
            176,
            24,
            9313,
            19,
            20086,
            28,
            45,
            10292,
            9,
            4,
            3,
            19,
            12943,
            4354,
            153,
            27,
            442,
            22,
            2771,
            4901,
            9,
            69,
            27,
            442,
            22,
            2771,
            24,
            11335,
            20,
            18,
            9225,
            2198,
            9,
            69,
            27,
            442,
            22,
            2771,
            24,
            11335,
            20,
            18,
            9225,
            2198,
            9,
            69,
            27,
            442,
            22,
            2771,
        ]
        # fmt: on
        #  In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria)
        #  are discovered. The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich,
        #  narrates the remainder of the story. 1883 Western Siberia, a young Grigori Rasputin
        #  is asked by his father and a group of men to perform magic. Rasputin has a vision and
        #  denounces one of the men as a horse thief. Although his father initially slaps
        #  him for making such an accusation, Rasputin watches as the man is chased outside and beaten.
        #  Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a priest.
        #  Rasputin quickly becomes famous, with people, even a bishop, begging for his blessing.
        #  <sep><cls>, Rasputin is asked to perform magic. He is asked to perform a ritual of the Virgin Mary.
        #  He is asked to perform a ritual of the Virgin Mary. He is asked to perform

        output_ids = model.generate(input_ids, max_length=200, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
