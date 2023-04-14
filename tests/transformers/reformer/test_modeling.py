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

import unittest

import paddle
import paddle.nn as nn
from parameterized import parameterized_class

from paddlenlp.transformers import (
    ReformerForMaskedLM,
    ReformerForQuestionAnswering,
    ReformerForSequenceClassification,
    ReformerModel,
    ReformerModelWithLMHead,
)
from paddlenlp.transformers.reformer.configuration import ReformerConfig
from paddlenlp.transformers.reformer.modeling import (
    REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    ReformerLayer,
)
from tests.testing_utils import slow

from ..test_configuration_common import ConfigTester
from ..test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


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
        result = model(input_ids, attention_mask=input_mask, return_dict=self.parent.return_dict)
        result = model(input_ids, return_dict=self.parent.return_dict)

        # 2 * hidden_size because we use reversible resnet layers
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, 2 * self.hidden_size])

    def create_and_check_reformer_model_with_lm_backward(
        self, config: ReformerConfig, input_ids, input_mask, choice_labels
    ):
        if not self.is_training:
            return

        config.is_decoder = False
        config.lsh_num_chunks_after = 1
        model = ReformerForMaskedLM(config=config)
        model.train()
        loss = model(input_ids, attention_mask=input_mask, labels=input_ids, return_dict=self.parent.return_dict)[0]
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
        result = model(input_ids, attention_mask=input_mask, labels=input_ids, return_dict=self.parent.return_dict)
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
            embedding.weight = paddle.create_parameter(
                embedding.weight.shape, dtype="float32", default_initializer=nn.initializer.Constant(value=0)
            )
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

        output_padded = model(input_ids_padded, attention_mask=attn_mask, return_dict=self.parent.return_dict)[0][
            :, :half_seq_len
        ]
        output_padded_rolled = model(
            input_ids_roll, attention_mask=attn_mask_roll, return_dict=self.parent.return_dict
        )[0][:, roll : half_seq_len + roll]

        self.parent.assertTrue(paddle.allclose(output_padded, output_padded_rolled, atol=1e-4))

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
                atol=1e-4,
            )
        )

        paddle.seed(layer.feed_forward_seed)
        feed_forward_hidden_states = layer.feed_forward(next_attn_output)
        self.parent.assertTrue(
            paddle.allclose(
                next_hidden_states,
                hidden_states + feed_forward_hidden_states,
                atol=1e-4,
            )
        )

    def create_and_check_reformer_feed_backward_chunking(
        self, config: ReformerConfig, input_ids, input_mask, choice_labels
    ):
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
        loss_no_chunk, output_no_chunk = model(
            input_ids, labels=input_ids, attention_mask=input_mask, return_dict=self.parent.return_dict
        )[:2]
        loss_no_chunk.backward()
        grad_slice_word_no_chunk = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        grad_slice_position_factor_1_no_chunk = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        grad_slice_position_factor_2_no_chunk = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]

        config.chunk_size_lm_head = 1
        config.chunk_size_feed_forward = 1

        paddle.seed(0)
        model = ReformerForMaskedLM(config=config)
        model.train()
        loss_chunk, output_chunk = model(
            input_ids, labels=input_ids, attention_mask=input_mask, return_dict=self.parent.return_dict
        )[:2]
        loss_chunk.backward()
        grad_slice_word_chunk = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        grad_slice_position_factor_1_chunk = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        grad_slice_position_factor_2_chunk = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        self.parent.assertTrue(paddle.allclose(loss_chunk, loss_no_chunk, atol=1e-4))
        self.parent.assertTrue(paddle.allclose(grad_slice_word_no_chunk, grad_slice_word_chunk, atol=1e-4))
        self.parent.assertTrue(
            paddle.allclose(grad_slice_position_factor_1_chunk, grad_slice_position_factor_1_no_chunk, atol=1e-4)
        )
        self.parent.assertTrue(
            paddle.allclose(grad_slice_position_factor_2_chunk, grad_slice_position_factor_2_no_chunk, atol=1e-4)
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
        output_logits = model(
            input_ids, attention_mask=input_mask, return_dict=self.parent.return_dict
        )  # (loss, logits, hidden_states, attentions)
        self.parent.assertTrue(output_logits[0].shape[1] == input_ids.shape[-1])

    def create_and_check_reformer_for_question_answering(
        self, config: ReformerConfig, input_ids, input_mask, choice_labels, sequence_labels
    ):
        model = ReformerForQuestionAnswering(config=config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            start_positions=choice_labels,
            end_positions=choice_labels,
            return_dict=self.parent.return_dict,
        )
        if sequence_labels is not None:
            start_logits, end_logits = result[1], result[2]
        else:
            start_logits, end_logits = result[0], result[1]
        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

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
        result = model(
            input_ids, attention_mask=input_mask, labels=sequence_labels, return_dict=self.parent.return_dict
        )
        if sequence_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_labels])


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
)
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

    def test_reformer_cached_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_cache(*config_and_inputs)

    def test_reformer_cached_generate(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_generate(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_for_sequence_classification(*config_and_inputs, is_decoder=False)


class ReformerLocalAttnModelTest(ReformerTesterMixin, ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        ReformerModel,
        ReformerModelWithLMHead,
        ReformerForSequenceClassification,
        ReformerForQuestionAnswering,
    )
    all_generative_model_classes = {ReformerModelWithLMHead: (ReformerModel, "Reformer")}
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    test_sequence_classification_problem_types = True
    test_tie_weights = True
    base_model_class = ReformerModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = True

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
    all_model_classes = (
        ReformerModel,
        ReformerModelWithLMHead,
        ReformerForSequenceClassification,
        ReformerForQuestionAnswering,
    )
    all_generative_model_classes = {ReformerModelWithLMHead: (ReformerModel, "Reformer")}
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    test_tie_weights = False  # reformer tie_weights implementation is problematic for now
    base_model_class = ReformerModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = True

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


class ReformerModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = ReformerModel
    hf_remote_test_model_path = "PaddleCI/tiny-random-reformer"
    paddlehub_remote_test_model_path = "__internal_testing__/tiny-random-reformer"

    @slow
    def test_inference_no_attention(self):
        model = ReformerModel.from_pretrained("reformer-enwik8")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 2048]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.08537189, -0.01475962, 0.28183940],
                    [0.11155435, 0.03538624, 0.37847346],
                    [0.12673721, 0.07730877, 0.48841247],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = ReformerModel.from_pretrained("reformer-enwik8")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 2048]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.08537189, -0.01475962, 0.28183940],
                    [0.11155435, 0.03538624, 0.37847346],
                    [0.12673721, 0.07730877, 0.48841247],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))
