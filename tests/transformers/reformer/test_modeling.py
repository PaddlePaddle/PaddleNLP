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

import pdb
import random
import tempfile
import unittest

import numpy as np
import paddle
import paddle.nn as nn
from parameterized import parameterized_class

from paddlenlp.transformers import (
    ReformerModel,
    ReformerTokenizer,
    ReformerForMaskedLM,
    ReformerForQuestionAnswering,
    ReformerForSequenceClassification,
    ReformerModelWithLMHead,
)
from paddlenlp.transformers.reformer.configuration import ReformerConfig
from paddlenlp.transformers.reformer.modeling import REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,  ReformerLayer
from tests.testing_utils import require_package, slow

from ..test_generation_utils import GenerationTesterMixin
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask, floats_tensor

class ReformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=32,
        is_training=True,
        is_decoder=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=32,
        attention_head_size=16,
        hidden_size=32,
        num_attention_heads=2,
        local_attn_chunk_length=4,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        num_buckets=None,
        num_hashes=1,
        lsh_attn_chunk_length=None,
        lsh_num_chunks_before=None,
        lsh_num_chunks_after=None,
        chunk_size_lm_head=0,
        chunk_size_feed_forward=0,
        feed_forward_size=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        local_attention_probs_dropout_prob=0.1,
        lsh_attention_probs_dropout_prob=None,
        max_position_embeddings=512,
        initializer_range=0.02,
        axial_norm_std=1.0,
        layer_norm_eps=1e-12,
        axial_pos_embds=True,
        axial_pos_shape=[4, 8],
        axial_pos_embds_dim=[16, 16],
        attn_layers=["local", "local", "local", "local"],
        pad_token_id=0,
        eos_token_id=2,
        scope=None,
        hash_seed=0,
        num_labels=2,
        num_hidden_layers=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.is_decoder = is_decoder
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.attention_head_size = attention_head_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.local_attn_chunk_length = local_attn_chunk_length
        self.local_num_chunks_after = local_num_chunks_after
        self.local_num_chunks_before = local_num_chunks_before
        self.num_hashes = num_hashes
        self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.hidden_act = hidden_act
        self.feed_forward_size = feed_forward_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.axial_pos_embds = axial_pos_embds
        self.axial_pos_shape = tuple(axial_pos_shape)
        self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
        self.axial_norm_std = axial_norm_std
        self.chunk_size_lm_head = chunk_size_lm_head
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.scope = scope
        self.attn_layers = attn_layers
        self.pad_token_id = pad_token_id
        self.hash_seed = hash_seed
        self.num_hidden_layers = num_hidden_layers

        attn_chunk_length = local_attn_chunk_length if local_attn_chunk_length is not None else lsh_attn_chunk_length
        num_chunks_after = local_num_chunks_after if local_num_chunks_after is not None else lsh_num_chunks_after
        num_chunks_before = local_num_chunks_before if local_num_chunks_before is not None else lsh_num_chunks_before

        self.encoder_seq_length = seq_length // attn_chunk_length + (self.seq_length % attn_chunk_length != 0)
        self.key_length = (num_chunks_before + num_chunks_after + 1) * attn_chunk_length
        self.chunk_length = attn_chunk_length
        self.num_labels = num_labels

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        choice_labels = None
        if self.use_labels:
            choice_labels = ids_tensor([self.batch_size], 2)

        config = self.get_config()

        return (
            config,
            input_ids,
            input_mask,
            choice_labels,
        )

    def get_pipeline_config(self) -> ReformerConfig:
        config = self.get_config()
        config.vocab_size = 100
        config.max_position_embeddings = 100
        config.axial_pos_shape = (4, 25)
        config.is_decoder = False
        return config

    def get_config(self) -> ReformerConfig:
        return ReformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            feed_forward_size=self.feed_forward_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            local_attention_probs_dropout_prob=self.local_attention_probs_dropout_prob,
            lsh_attention_probs_dropout_prob=self.lsh_attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            is_decoder=self.is_decoder,
            axial_pos_embds=self.axial_pos_embds,
            axial_pos_shape=self.axial_pos_shape,
            axial_pos_embds_dim=self.axial_pos_embds_dim,
            local_attn_chunk_length=self.local_attn_chunk_length,
            local_num_chunks_after=self.local_num_chunks_after,
            local_num_chunks_before=self.local_num_chunks_before,
            num_hashes=self.num_hashes,
            num_buckets=self.num_buckets,
            lsh_attn_chunk_length=self.lsh_attn_chunk_length,
            lsh_num_chunks_after=self.lsh_num_chunks_after,
            lsh_num_chunks_before=self.lsh_num_chunks_before,
            attn_layers=self.attn_layers,
            pad_token_id=self.pad_token_id,
            hash_seed=self.hash_seed,
            attention_head_size=self.attention_head_size,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            axial_norm_std=self.axial_norm_std,
        )

    def create_and_check_reformer_model(self, config: ReformerConfig, input_ids, input_mask, choice_labels):
        model = ReformerModel(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)

        # 2 * hidden_size because we use reversible resnet layers
        self.parent.assertEqual(
            result[0].shape, [self.batch_size, self.seq_length, 2 * self.hidden_size]
        )

    def create_and_check_reformer_model_with_lm_backward(self, config: ReformerConfig, input_ids, input_mask, choice_labels):
        if not self.is_training:
            return

        config.is_decoder = False
        config.lsh_num_chunks_after = 1
        model = ReformerForMaskedLM(config=config)
        model.train()
        loss = model(input_ids, attention_mask=input_mask, labels=input_ids)[0]
        loss.backward()

    def create_and_check_reformer_with_lm(self, config: ReformerConfig, input_ids, input_mask, choice_labels):
        config.lsh_num_chunks_after = 0
        config.is_decoder = True
        model = ReformerModelWithLMHead(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=input_ids)
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_reformer_with_mlm(self, config: ReformerConfig, input_ids, input_mask, choice_labels):
        config.is_decoder = False
        model = ReformerForMaskedLM(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=input_ids)
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_reformer_model_with_attn_mask(
        self, config: ReformerConfig, input_ids, input_mask, choice_labels, is_decoder=False
    ):
        # no special position embeddings
        config.axial_pos_embds = False
        config.is_decoder = is_decoder

        if self.lsh_attn_chunk_length is not None:
            # need to set chunk length equal sequence length to be certain that chunking works
            config.lsh_attn_chunk_length = self.seq_length

        model = ReformerModel(config=config)
        model.eval()
        # set all position encodings to zero so that postions don't matter
        with paddle.no_grad():
            embedding = model.embeddings.position_embeddings.embedding
            embedding.weight = paddle.create_parameter(embedding.weight.shape, dtype='float32', default_initializer=nn.initializer.Constant(value=0))
            embedding.weight.requires_grad = False

        half_seq_len = self.seq_length // 2
        roll = self.chunk_length

        half_input_ids = input_ids[:, :half_seq_len]

        # normal padded
        attn_mask = paddle.concat(
            [paddle.ones_like(half_input_ids), paddle.zeros_like(half_input_ids)],
            axis=-1,
        )
        input_ids_padded = paddle.concat(
            [half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)],
            axis=-1,
        )

        # shifted padded
        input_ids_roll = paddle.concat(
            [half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)],
            axis=-1,
        )
        input_ids_roll = paddle.roll(input_ids_roll, roll, axis=-1)
        attn_mask_roll = paddle.roll(attn_mask, roll, axis=-1)

        output_padded = model(input_ids_padded, attention_mask=attn_mask)[0][:, :half_seq_len]
        output_padded_rolled = model(input_ids_roll, attention_mask=attn_mask_roll)[0][:, roll : half_seq_len + roll]

        self.parent.assertTrue(paddle.allclose(output_padded, output_padded_rolled, atol=1e-3))

    def create_and_check_reformer_layer_dropout_seed(
        self, config: ReformerConfig, input_ids, input_mask, choice_labels, is_decoder=False
    ):
        config.is_decoder = is_decoder
        layer = ReformerLayer(config)
        layer.train()
        shape = (
            self.batch_size,
            self.seq_length,
            config.hidden_size,
        )  # Batch x SeqLen x hiddenSize

        # get random tensors
        hidden_states = floats_tensor(shape)
        prev_attn_output = floats_tensor(shape)

        # now the random seeds for attention and feed forward is initialized
        # forward tensors with dropout
        layer_outputs = layer(prev_attn_output, hidden_states, attention_mask=input_mask)

        next_attn_output = layer_outputs.attn_output
        next_hidden_states = layer_outputs.hidden_states

        paddle.seed(layer.attention_seed)
        attn_outputs = layer.attention(hidden_states, attention_mask=input_mask)
        self.parent.assertTrue(
            paddle.allclose(
                prev_attn_output + attn_outputs.hidden_states,
                next_attn_output,
                atol=1e-3,
            )
        )

        paddle.seed(layer.feed_forward_seed)
        feed_forward_hidden_states = layer.feed_forward(next_attn_output)
        self.parent.assertTrue(
            paddle.allclose(
                next_hidden_states,
                hidden_states + feed_forward_hidden_states,
                atol=1e-3,
            )
        )

    def create_and_check_reformer_feed_backward_chunking(self, config: ReformerConfig, input_ids, input_mask, choice_labels):
        if not self.is_training:
            return

        # disable dropout
        config.hidden_dropout_prob = 0
        config.local_attention_probs_dropout_prob = 0
        config.lsh_attention_probs_dropout_prob = 0
        config.lsh_num_chunks_after = 1
        config.is_decoder = False

        paddle.seed(0)
        model = ReformerForMaskedLM(config=config)
        model.train()
        #model.zero_grad()
        loss_no_chunk, output_no_chunk = model(input_ids, labels=input_ids, attention_mask=input_mask)[:2]
        loss_no_chunk.backward()
        grad_slice_word_no_chunk = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        grad_slice_position_factor_1_no_chunk = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        grad_slice_position_factor_2_no_chunk = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]

        config.chunk_size_lm_head = 1
        config.chunk_size_feed_forward = 1

        paddle.seed(0)
        model = ReformerForMaskedLM(config=config)
        model.train()
        #model.zero_grad()
        loss_chunk, output_chunk = model(input_ids, labels=input_ids, attention_mask=input_mask)[:2]
        loss_chunk.backward()
        grad_slice_word_chunk = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        grad_slice_position_factor_1_chunk = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        grad_slice_position_factor_2_chunk = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        self.parent.assertTrue(paddle.allclose(loss_chunk, loss_no_chunk, atol=1e-3))
        self.parent.assertTrue(paddle.allclose(grad_slice_word_no_chunk, grad_slice_word_chunk, atol=1e-3))
        self.parent.assertTrue(
            paddle.allclose(grad_slice_position_factor_1_chunk, grad_slice_position_factor_1_no_chunk, atol=1e-3)
        )
        self.parent.assertTrue(
            paddle.allclose(grad_slice_position_factor_2_chunk, grad_slice_position_factor_2_no_chunk, atol=1e-3)
        )

    def create_and_check_reformer_model_generate(self, config: ReformerConfig, input_ids, input_mask, choice_labels):
        config.is_decoder = True
        config.lsh_num_chunks_after = 0
        config.bos_token_id = 0
        config.eos_token_id = None
        config.max_length = 20

        model = ReformerModelWithLMHead(config=config)
        model.eval()
        output = model.generate()
        self.parent.assertIsNotNone(output)

    def create_and_check_reformer_no_chunking(self, config: ReformerConfig, input_ids, input_mask, choice_labels):
        # force chunk length to be bigger than input_ids
        config.lsh_attn_chunk_length = 2 * input_ids.shape[-1]
        config.local_attn_chunk_length = 2 * input_ids.shape[-1]
        config.lsh_num_chunks_after = 1
        config.is_decoder = False
        model = ReformerForMaskedLM(config=config)
        model.eval()
        output_logits = model(input_ids, attention_mask=input_mask) # (loss, logits, hidden_states, attentions)
        self.parent.assertTrue(output_logits[0].shape[1] == input_ids.shape[-1])

    def create_and_check_reformer_for_question_answering(self, config: ReformerConfig, input_ids, input_mask, choice_labels):
        model = ReformerForQuestionAnswering(config=config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            start_positions=choice_labels,
            end_positions=choice_labels,
        )
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[2].shape, [self.batch_size, self.seq_length])

    def create_and_check_cache(self, config: ReformerConfig, input_ids, input_mask, choice_labels):
        config.is_decoder = True
        config.lsh_num_chunks_before = 1
        config.lsh_num_chunks_after = 0
        model = ReformerModelWithLMHead(config=config)
        model.eval()
        input_ids_first = input_ids[:, :-1]
        input_ids_second = input_ids[:, -1:]

        # return saved cache
        cache = model(input_ids_first, use_cache=True)[1]

        # calculate last output with and without cache
        outputs_with_cache = model(input_ids_second, cache=cache, use_cache=True)[0]
        outputs_without_cache = model(input_ids)[0][:, -1]

        # select random slice idx
        random_slice_idx = paddle.randint(outputs_without_cache.shape[-1], shape=(1, 1)).item()

        # outputs should be similar within range
        self.parent.assertTrue(
            paddle.allclose(
                outputs_with_cache[:, 0, random_slice_idx], outputs_without_cache[:, random_slice_idx], atol=1e-2
            )
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, choice_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict

    def create_and_check_reformer_for_sequence_classification(
        self, config, input_ids, input_mask, choice_labels, is_decoder
    ):
        config.is_decoder = is_decoder
        sequence_labels = ids_tensor([self.batch_size], config.num_labels)
        model = ReformerForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.num_labels])


class ReformerTesterMixin:
    """
    Reformer Local and Reformer LSH run essentially the same tests
    """

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_reformer_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model(*config_and_inputs)

    def test_reformer_lm_model_backward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_with_lm_backward(*config_and_inputs)

    def test_reformer_model_attn_masking(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, is_decoder=True)
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, is_decoder=False)

    def test_reformer_with_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_with_lm(*config_and_inputs)

    def test_reformer_with_mlm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_with_mlm(*config_and_inputs)

    def test_reformer_layer_training_dropout(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_layer_dropout_seed(*config_and_inputs, is_decoder=True)
        self.model_tester.create_and_check_reformer_layer_dropout_seed(*config_and_inputs, is_decoder=False)

    def test_reformer_chunking_backward_equality(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_feed_backward_chunking(*config_and_inputs)

    def test_reformer_no_chunking(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_no_chunking(*config_and_inputs)

    def test_reformer_qa_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_for_question_answering(*config_and_inputs)

    '''def test_reformer_cached_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_cache(*config_and_inputs)'''

    def test_reformer_cached_generate(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_generate(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_for_sequence_classification(*config_and_inputs, is_decoder=False)


class ReformerLocalAttnModelTest(ReformerTesterMixin, ModelTesterMixin, unittest.TestCase):
#class ReformerLocalAttnModelTest(ReformerTesterMixin, GenerationTesterMixin, ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (ReformerModel, ReformerModelWithLMHead, ReformerForSequenceClassification, ReformerForQuestionAnswering)
    )
    #all_generative_model_classes = (ReformerModelWithLMHead,)
    all_generative_model_classes = {ReformerModelWithLMHead: (ReformerModel, "Reformer")}
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    test_sequence_classification_problem_types = True
    base_model_class = ReformerModel
    def setUp(self):
        self.model_tester = ReformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ReformerConfig, hidden_size=37)

    @slow
    def test_model_from_pretrained(self):
        for model_name in REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ReformerModelWithLMHead.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, list) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)

        for idx, iter_attentions in enumerate(attentions):
            tgt_len = min_length + idx if not use_cache else 1
            num_chunks = tgt_len // config.local_attn_chunk_length + (tgt_len % config.local_attn_chunk_length != 0)
            tgt_chunk_len = config.local_attn_chunk_length
            src_chunk_len = config.local_attn_chunk_length * (
                1 + config.local_num_chunks_after + config.local_num_chunks_before
            )

            if use_cache:
                expected_shape = (
                    batch_size * num_beam_groups,
                    config.num_attention_heads,
                    tgt_len,
                    min_length // config.local_attn_chunk_length + 1 + idx,
                )
            else:
                expected_shape = (
                    batch_size * num_beam_groups,
                    config.num_attention_heads,
                    num_chunks,
                    tgt_chunk_len,
                    src_chunk_len,
                )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, list) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            seq_len = min_length + idx
            seq_len = config.local_attn_chunk_length * (
                seq_len // config.local_attn_chunk_length + (seq_len % config.local_attn_chunk_length != 0)
            )

            if use_cache:
                seq_len = 1

            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )



class ReformerLSHAttnModelTest(ReformerTesterMixin, ModelTesterMixin, unittest.TestCase):
# class ReformerLSHAttnModelTest(ReformerTesterMixin, ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (ReformerModel, ReformerModelWithLMHead, ReformerForSequenceClassification, ReformerForQuestionAnswering)
    )
    #all_generative_model_classes = (ReformerModelWithLMHead,)
    all_generative_model_classes = {ReformerModelWithLMHead: (ReformerModel, "Reformer")}
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    base_model_class = ReformerModel
    def setUp(self):
        self.model_tester = ReformerModelTester(
            self,
            batch_size=13,
            seq_length=13,
            use_input_mask=True,
            use_labels=True,
            is_training=False,
            is_decoder=True,
            vocab_size=32,
            attention_head_size=16,
            hidden_size=64,
            num_attention_heads=2,
            num_buckets=2,
            num_hashes=4,
            lsh_attn_chunk_length=4,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            chunk_size_lm_head=5,
            chunk_size_feed_forward=6,
            feed_forward_size=32,
            hidden_act="relu",
            hidden_dropout_prob=0.1,
            lsh_attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            initializer_range=0.02,
            axial_norm_std=1.0,
            layer_norm_eps=1e-12,
            axial_pos_embds=True,
            axial_pos_shape=[4, 8],
            axial_pos_embds_dim=[16, 48],
            # sanotheu
            # attn_layers=[lsh,lsh,lsh,lsh],
            attn_layers=["lsh"],
            pad_token_id=0,
            eos_token_id=2,
            scope=None,
            hash_seed=0,
            num_labels=2,
            num_hidden_layers=1,
        )
        self.config_tester = ConfigTester(self, config_class=ReformerConfig, hidden_size=37)

    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, list) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)

        for idx, iter_attentions in enumerate(attentions):
            tgt_len = min_length + idx if not use_cache else 1
            num_chunks = tgt_len // config.lsh_attn_chunk_length + (tgt_len % config.lsh_attn_chunk_length != 0)
            tgt_chunk_len = config.lsh_attn_chunk_length
            src_chunk_len = config.lsh_attn_chunk_length * (
                1 + config.lsh_num_chunks_after + config.lsh_num_chunks_before
            )

            if use_cache:
                expected_shape = (
                    batch_size * num_beam_groups,
                    config.num_attention_heads,
                    config.num_hashes,
                    tgt_len,
                    config.num_hashes * (1 + config.lsh_num_chunks_after + config.lsh_num_chunks_before),
                )
            else:
                expected_shape = (
                    batch_size * num_beam_groups,
                    config.num_attention_heads,
                    num_chunks * config.num_hashes,
                    tgt_chunk_len,
                    src_chunk_len,
                )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, list) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            seq_len = min_length + idx if not use_cache else 1
            seq_len = config.lsh_attn_chunk_length * (
                seq_len // config.lsh_attn_chunk_length + (seq_len % config.lsh_attn_chunk_length != 0)
            )

            if use_cache:
                seq_len = 1

            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    def test_problem_types(self):
        # Fails because the sequence length is not a multiple of 4
        pass



'''class ReformerIntegrationTests(unittest.TestCase):
    """
    These integration tests test the current layer activations and gradients againts the output of the Hugging Face Reformer model at time of integration: 29/06/2020. During integration, the model was tested against the output of the official Trax ReformerLM model for various cases ("lsh" only, "lsh" only, masked / non-masked, different chunk length, ....). In order to recover the original trax integration tests, one should use patrickvonplaten's fork of trax and the code that lives on the branch `reformer_trax_tests`.
    """

    def _get_basic_config_and_input(self):
        config = {
            "vocab_size": 320,
            "attention_head_size": 8,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_buckets": 2,
            "num_hashes": 4,
            "lsh_attn_chunk_length": 4,
            "local_attn_chunk_length": 4,
            "lsh_num_chunks_before": 1,
            "lsh_num_chunks_after": 0,
            "local_num_chunks_before": 1,
            "local_num_chunks_after": 0,
            "chunk_size_lm_head": 0,
            "chunk_size_feed_forward": 0,
            "feed_forward_size": 32,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "lsh_attention_probs_dropout_prob": 0.0,
            "local_attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 32,
            "initializer_range": 0.02,
            "axial_norm_std": 1.0,
            "layer_norm_eps": 1e-12,
            "sinusoidal_pos_embds": False,
            "axial_pos_embds": True,
            "axial_pos_shape": [4, 8],
            "axial_pos_embds_dim": [8, 8],
            "hash_seed": 0,
            "is_decoder": True,
        }
        return config

    def _get_hidden_states(self):
        return paddle.to_tensor(
            [
                [
                    [
                        1.90826353e00,
                        -1.45999730e00,
                        -6.20405462e-01,
                        1.52503433e00,
                        -3.64464232e-01,
                        -8.27359235e-01,
                        8.39670803e-01,
                        2.44492178e-01,
                        4.98332758e-01,
                        2.69175139e00,
                        -7.08081422e-03,
                        1.04915401e00,
                        -1.83476661e00,
                        7.67220476e-01,
                        2.98580543e-01,
                        2.84803992e-02,
                    ],
                    [
                        -2.66374286e-02,
                        4.33497576e-01,
                        3.10386309e-01,
                        5.46039944e-01,
                        -2.47292666e-04,
                        -7.52305019e-01,
                        2.39162103e-01,
                        7.25216186e-01,
                        -7.58357372e-01,
                        4.20635998e-01,
                        -4.04739919e-02,
                        1.59924145e-01,
                        2.05135748e00,
                        -1.15997978e00,
                        5.37166397e-01,
                        2.62873606e-01,
                    ],
                    [
                        1.85247482e-01,
                        7.07046037e-01,
                        -6.77089715e-01,
                        -2.24209655e00,
                        -3.75307980e-02,
                        -8.59380874e-01,
                        -2.81027884e00,
                        1.01276376e00,
                        -1.69438001e00,
                        4.17574660e-01,
                        -1.49196962e00,
                        -1.76483717e00,
                        -1.94566312e-01,
                        -1.71183858e00,
                        7.72903565e-01,
                        -1.11557056e00,
                    ],
                    [
                        9.46069193e-01,
                        1.53417623e-01,
                        -9.58686996e-01,
                        1.18126669e-01,
                        1.75967724e00,
                        1.62194590e00,
                        -5.74108159e-01,
                        6.79920443e-01,
                        5.44028163e-01,
                        2.05466114e-01,
                        -3.63045868e-01,
                        2.41865062e-01,
                        3.20348382e-01,
                        -9.05611176e-01,
                        -1.92690727e-01,
                        -1.19917547e00,
                    ],
                ]
            ],
            dtype='float32',
        )

    def _get_attn_mask(self):
        return paddle.to_tensor([[0, 1, 0, 0]], dtype='int64')

    def _get_input_ids_and_mask(self):
        mask = paddle.to_tensor(
            [
                [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
            ],
            dtype='int64',
        )

        input_ids = paddle.to_tensor(
            [
                [
                    89,
                    279,
                    286,
                    84,
                    194,
                    316,
                    182,
                    28,
                    283,
                    37,
                    169,
                    7,
                    253,
                    267,
                    107,
                    250,
                    44,
                    7,
                    102,
                    62,
                    3,
                    243,
                    171,
                    265,
                    302,
                    48,
                    164,
                    264,
                    148,
                    229,
                    280,
                    150,
                ],
                [
                    9,
                    192,
                    66,
                    112,
                    163,
                    83,
                    135,
                    70,
                    224,
                    96,
                    31,
                    80,
                    196,
                    80,
                    63,
                    22,
                    85,
                    100,
                    47,
                    283,
                    0,
                    163,
                    126,
                    143,
                    195,
                    82,
                    53,
                    82,
                    18,
                    27,
                    182,
                    52,
                ],
            ],
            dtype='int64',
        )

        return input_ids, mask

    def test_lsh_layer_forward(self):
        config = self._get_basic_config_and_input()
        config["lsh_num_chunks_before"] = 0
        config["attn_layers"] = ["lsh"]
        config["num_hidden_layers"] = 1
        config["is_decoder"] = False
        hidden_states = self._get_hidden_states()
        paddle.seed(0)
        layer = ReformerLayer(ReformerConfig(**config))
        layer.eval()
        reformer_output = layer(prev_attn_output=hidden_states.clone(), hidden_states=hidden_states)
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = paddle.to_tensor(
            [1.6879, -1.3083, -0.4708, 1.3555, -0.6292],
            dtype='float32',
        )
        self.assertTrue(paddle.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_lsh_layer_forward_complex(self):
        config = self._get_basic_config_and_input()
        config["lsh_num_chunks_before"] = 0
        config["attn_layers"] = ["lsh"]
        config["num_hidden_layers"] = 1
        config["num_buckets"] = [2, 4]
        attn_mask = self._get_attn_mask()
        hidden_states = self._get_hidden_states()
        paddle.seed(0)
        layer = ReformerLayer(ReformerConfig(**config))
        layer.eval()
        reformer_output = layer(
            prev_attn_output=hidden_states.clone(),
            hidden_states=hidden_states,
            attention_mask=attn_mask,
        )
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = paddle.to_tensor(
            [1.6439, -1.2306, -0.5108, 1.3006, -0.6537],
            dtype='float32',
        )
        self.assertTrue(paddle.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_local_layer_forward(self):
        config = self._get_basic_config_and_input()
        config["local_num_chunks_before"] = 0
        config["attn_layers"] = ["local"]
        config["num_hidden_layers"] = 1
        config["is_decoder"] = False
        hidden_states = self._get_hidden_states()
        paddle.seed(0)
        layer = ReformerLayer(ReformerConfig(**config))
        layer.eval()
        reformer_output = layer(prev_attn_output=hidden_states, hidden_states=hidden_states)
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = paddle.to_tensor(
            [1.4212, -2.0576, -0.9688, 1.4599, -0.1344],
            dtype='float32',
        )
        self.assertTrue(paddle.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_local_layer_forward_complex(self):
        config = self._get_basic_config_and_input()
        config["local_num_chunks_before"] = 0
        config["attn_layers"] = ["local"]
        config["num_hidden_layers"] = 1
        attn_mask = self._get_attn_mask()
        hidden_states = self._get_hidden_states()
        paddle.seed(0)
        layer = ReformerLayer(ReformerConfig(**config))
        layer.eval()
        reformer_output = layer(
            prev_attn_output=hidden_states,
            hidden_states=hidden_states,
            attention_mask=attn_mask,
        )
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = paddle.to_tensor(
            [1.4750, -2.0235, -0.9743, 1.4463, -0.1269],
            dtype='float32',
        )
        self.assertTrue(paddle.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_lsh_model_forward(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["lsh", "lsh", "lsh", "lsh"]
        config["num_hidden_layers"] = 4
        config["num_buckets"] = [2, 4]
        paddle.seed(0)
        model = ReformerModel(ReformerConfig(**config))
        model.eval()
        input_ids, attn_mask = self._get_input_ids_and_mask()
        hidden_states = model(input_ids=input_ids, attention_mask=attn_mask)[0]
        output_slice = hidden_states[0, 0, :5]
        expected_output_slice = paddle.to_tensor(
            [-0.9896, -0.9396, -1.0831, -0.0597, 0.2456],
            dtype='float32',
        )
        import numpy as np
        np.testing.assert_allclose(output_slice, expected_output_slice, atol=1e-3)
        # self.assertTrue(paddle.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_local_model_forward(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["local", "local", "local", "local"]
        config["num_hidden_layers"] = 4
        paddle.seed(0)
        model = ReformerModel(ReformerConfig(**config))
        model.eval()
        input_ids, attn_mask = self._get_input_ids_and_mask()
        hidden_states = model(input_ids=input_ids, attention_mask=attn_mask)[0]
        output_slice = hidden_states[0, 0, :5]
        expected_output_slice = paddle.to_tensor(
            [-1.6791, 0.7171, 0.1594, 0.4063, 1.2584],
            dtype='float32',
        )
        self.assertTrue(paddle.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_lm_model_forward(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["local", "lsh", "local", "lsh", "local", "lsh"]
        config["num_hidden_layers"] = 6
        config["num_buckets"] = [2, 4]
        config["is_decoder"] = False
        paddle.seed(0)
        model = ReformerForMaskedLM(ReformerConfig(**config))
        model.eval()
        input_ids, attn_mask = self._get_input_ids_and_mask()
        hidden_states = model(input_ids=input_ids, attention_mask=attn_mask)[0]
        output_slice = hidden_states[1, -1, :5]
        expected_output_slice = paddle.to_tensor(
            [0.1018, -0.2026, 0.2116, 0.0270, -0.1233],
            dtype='float32',
        )

        self.assertTrue(paddle.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_local_lm_model_grad(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["local", "local", "local", "local"]
        config["num_hidden_layers"] = 4
        config["hidden_dropout_prob"] = 0.0
        config["local_attention_probs_dropout_prob"] = 0.0
        paddle.seed(0)
        model = ReformerModelWithLMHead(ReformerConfig(**config))
        model.train()
        #model.zero_grad()
        input_ids, _ = self._get_input_ids_and_mask()
        loss = model(input_ids=input_ids, labels=input_ids)[0]

        self.assertTrue(paddle.allclose(loss, paddle.to_tensor(5.8019, dtype='float32'), atol=1e-3))
        loss.backward()

        # check last grads to cover all proable errors
        grad_slice_word = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        expected_grad_slice_word = paddle.to_tensor(
            [-0.0005, -0.0001, -0.0002, -0.0006, -0.0006],
            dtype='float32',
        )
        grad_slice_position_factor_1 = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        expected_grad_slice_pos_fac_1 = paddle.to_tensor(
            [-0.5235, 0.5704, 0.0922, -0.3140, 0.9928],
            dtype='float32',
        )
        grad_slice_position_factor_2 = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        expected_grad_slice_pos_fac_2 = paddle.to_tensor(
            [1.7960, 1.7668, 0.5593, 0.0907, 1.8342],
            dtype='float32',
        )
        self.assertTrue(paddle.allclose(grad_slice_word, expected_grad_slice_word, atol=1e-3))
        self.assertTrue(paddle.allclose(grad_slice_position_factor_1, expected_grad_slice_pos_fac_1, atol=1e-3))
        self.assertTrue(paddle.allclose(grad_slice_position_factor_2, expected_grad_slice_pos_fac_2, atol=1e-3))

    def test_lsh_lm_model_grad(self):
        config = self._get_basic_config_and_input()
        config["attn_layers"] = ["lsh", "lsh", "lsh", "lsh"]
        config["num_hidden_layers"] = 4
        config["hidden_dropout_prob"] = 0.0
        config["lsh_attention_probs_dropout_prob"] = 0.0
        config["num_buckets"] = [2, 4]
        config["num_hashes"] = 6
        paddle.seed(0)
        model = ReformerModelWithLMHead(ReformerConfig(**config))
        model.train()
        #model.zero_grad()
        input_ids, _ = self._get_input_ids_and_mask()
        loss = model(input_ids=input_ids, labels=input_ids)[0]

        self.assertTrue(paddle.allclose(loss, paddle.to_tensor(5.7854, dtype='float32'), atol=1e-3))
        loss.backward()
        # check last grads to cover all proable errors
        grad_slice_word = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        expected_grad_slice_word = paddle.to_tensor(
            [0.0004, 0.0003, 0.0006, -0.0004, 0.0002],
            dtype='float32',
        )
        grad_slice_position_factor_1 = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        expected_grad_slice_pos_fac_1 = paddle.to_tensor(
            [-0.3792, 0.5593, -1.6993, 0.2033, 0.4131],
            dtype='float32',
        )
        grad_slice_position_factor_2 = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        expected_grad_slice_pos_fac_2 = paddle.to_tensor(
            [-1.4212, -0.3201, -1.1944, 0.1258, 0.2856],
            dtype='float32',
        )
        self.assertTrue(paddle.allclose(grad_slice_word, expected_grad_slice_word, atol=1e-3))
        self.assertTrue(paddle.allclose(grad_slice_position_factor_1, expected_grad_slice_pos_fac_1, atol=1e-3))
        self.assertTrue(paddle.allclose(grad_slice_position_factor_2, expected_grad_slice_pos_fac_2, atol=1e-3))

    @slow
    def test_pretrained_generate_crime_and_punish(self):
        model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")
        tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")
        model.eval()

        input_ids = tokenizer.encode("A few months later", return_tensors="pt")
        output_ids = model.generate(
            input_ids, max_length=50, num_beams=4, early_stopping=True, do_sample=False, num_hashes=8
        )
        output = tokenizer.decode(output_ids[0])

        self.assertEqual(
            output,
            "A few months later state expression in his ideas, at the first entrance. He was positively for an inst",
        )

    @slow
    def test_pretrained_generate_use_cache_equality(self):
        model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")
        tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")
        model.eval()
        input_ids = tokenizer.encode("A few months later", return_tensors="pt")
        output_ids_with_cache = model.generate(input_ids, max_length=130, num_hashes=8, use_cache=False)
        output_ids_without_cache = model.generate(input_ids, max_length=130, num_hashes=8, use_cache=True)

        output_with_cache = tokenizer.decode(output_ids_with_cache[0])
        output_without_cache = tokenizer.decode(output_ids_without_cache[0])

        self.assertEqual(output_with_cache, output_without_cache)'''