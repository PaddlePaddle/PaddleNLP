# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
import tempfile

import paddle
from parameterized import parameterized_class

from paddlenlp.transformers import (
    AutoTokenizer,
    MBartConfig,
    MBartForConditionalGeneration,
    MBartForQuestionAnswering,
    MBartForSequenceClassification,
    MBartModel,
)
from paddlenlp.transformers.mbart.modeling import MBartDecoder
from tests.testing_utils import PaddleNLPModelTest, slow

from ..test_generation_utils import GenerationTesterMixin
from ..test_modeling_common import ModelTesterMixin, ids_tensor


def prepare_mbart_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = (input_ids == config.pad_token_id).astype("float32").unsqueeze([1, 2]) * -1e4
    if decoder_attention_mask is None:
        decoder_attention_mask = (decoder_input_ids == config.pad_token_id).astype("float32").unsqueeze([1, 2]) * -1e4
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
    }


class MBartModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        decoder_start_token_id=2,
        activation_function="relu",
        activation_dropout=0.0,
        init_std=0.02,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.init_std = init_std

        # forcing a certain token to be generated, sets all other tokens to -inf
        # if however the token to be generated is already at -inf then it can lead token
        # `nan` values and thus break generation
        self.forced_bos_token_id = None
        self.forced_eos_token_id = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")
        input_ids = paddle.clip(ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64"), 3)
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")

        config = self.get_config()
        inputs_dict = prepare_mbart_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return MBartConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            forced_bos_token_id=self.forced_bos_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            activation_function=self.activation_function,
            activation_dropout=self.activation_dropout,
            init_std=self.init_std,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = MBartModel(config).get_decoder()
        model.eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        cache = model.decoder.gen_cache(paddle.randn(shape=[input_ids.shape[0], input_ids.shape[1], config.d_model]))

        # first forward pass
        outputs = model(
            input_ids, decoder_attention_mask=attention_mask, cache=cache, return_dict=self.parent.return_dict
        )

        output, past_key_values = outputs[:2]

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size, dtype="int64")
        next_attn_mask = (1 - ids_tensor((self.batch_size, 3), 2, dtype="int64").unsqueeze([1, 2])).astype(
            "float32"
        ) * -1e4

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(
            next_input_ids, decoder_attention_mask=next_attention_mask, cache=None, return_dict=self.parent.return_dict
        )
        if self.parent.return_dict:
            output_from_no_past = output_from_no_past[0]
        output_from_past = model(next_tokens, decoder_attention_mask=next_attention_mask, cache=past_key_values)[0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1], dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))


@parameterized_class(
    ("return_dict",),
    [
        [False],
        [True],
    ],
)
class MBartModelTest(ModelTesterMixin, GenerationTesterMixin, PaddleNLPModelTest):
    base_model_class = MBartModel

    all_model_classes = (
        MBartModel,
        MBartForConditionalGeneration,
        MBartForSequenceClassification,
        MBartForQuestionAnswering,
    )

    all_generative_model_classes = {MBartForConditionalGeneration: (MBartModel, "mbart")}
    is_encoder_decoder = True
    test_missing_keys = False
    return_dict = False

    def setUp(self):
        self.model_tester = MBartModelTester(self)

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_class.from_pretrained(tmpdirname)  # assign a model but never use

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_inputs_embeds_for_mbart(self):
        # NOTE: rewrite test inputs embeds for mbart model since scaler not equal to 1.0
        # get config for model and inputs_dict for model forward
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        scaler = config.d_model**0.5
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
                inputs["inputs_embeds"] = wte(input_ids) * scaler
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids) * scaler
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids) * scaler

            with paddle.no_grad():
                embeds_output = model(**inputs)

            self.assertTrue(paddle.allclose(ids_output, embeds_output, rtol=1e-4, atol=1e-4))


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if paddle.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (paddle.greater_than((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return paddle.to_tensor(tok_lst, dtype="int64")


class AbstractSeq2SeqIntegrationTest(PaddleNLPModelTest):
    maxDiff = 1000  # longer string compare tracebacks
    checkpoint_name = None

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.checkpoint_name)
        return cls

    def model(self):
        """Only load the model if needed."""
        model = MBartForConditionalGeneration.from_pretrained(self.checkpoint_name)
        model.eval()
        return model


@parameterized_class(
    ("return_dict",),
    [
        [False],
        [True],
    ],
)
class MBartEnroIntegrationTest(AbstractSeq2SeqIntegrationTest):
    checkpoint_name = "mbart-large-en-ro"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        """ Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.""",
    ]
    tgt_text = [
        "Şeful ONU declară că nu există o soluţie militară în Siria",
        'Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar acordat de Rusia Siriei este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor face decât să înrăutăţească violenţele şi mizeria a milioane de oameni.',
    ]
    expected_src_tokens = [8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23, 51712, 2, 250004]
    return_dict = False

    @slow
    def test_enro_generate_one(self):
        batch = self.tokenizer(
            ["UN Chief Says There Is No Military Solution in Syria"], return_tensors="pd", return_token_type_ids=False
        )
        model = self.model()
        translated_tokens = model.generate(**batch, max_length=128)[0]
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        self.assertEqual(self.tgt_text[0], decoded[0])

    @slow
    def test_enro_generate_batch(self):
        batch = self.tokenizer(
            self.src_text, return_tensors="pd", padding=True, truncation=True, return_token_type_ids=False
        )
        model = self.model()
        translated_tokens = model.generate(**batch, max_length=128, decode_strategy="greedy_search")[0]
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        for i in range(len(self.tgt_text)):
            assert str(self.tgt_text[i]) == str(decoded[i]), f"{i}"

    def test_mbart_fast_forward(self):
        config = MBartConfig(
            vocab_size=99,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
        )
        lm_model = MBartForConditionalGeneration(config)
        context = paddle.to_tensor([[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]], dtype="int64")
        summary = paddle.to_tensor([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]], dtype="int64")
        loss, logits = lm_model(
            input_ids=context, decoder_input_ids=summary, labels=summary, return_dict=self.return_dict
        )[:2]
        expected_shape = [*summary.shape, config.vocab_size]
        self.assertIsInstance(loss.item(), float)
        self.assertEqual(logits.shape, expected_shape)


class MBartCC25IntegrationTest(AbstractSeq2SeqIntegrationTest):
    checkpoint_name = "mbart-large-cc25"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        " I ate lunch twice yesterday",
    ]
    tgt_text = ["Şeful ONU declară că nu există o soluţie militară în Siria", "to be padded"]

    @slow
    def test_fill_mask(self):
        inputs = self.tokenizer(["One of the best <mask> I ever read!"], return_tensors="pd")
        model = self.model()
        outputs = model.generate(inputs["input_ids"], decoder_start_token_id=self.tokenizer.lang_code_to_id["en_XX"])[
            0
        ]
        prediction = self.tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)[
            0
        ]
        self.assertEqual(prediction, "of the best books I ever read!")


class MBartStandaloneDecoderModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        d_model=16,
        decoder_seq_length=7,
        is_training=True,
        is_decoder=True,
        use_attention_mask=True,
        use_cache=False,
        use_labels=True,
        decoder_start_token_id=2,
        decoder_ffn_dim=32,
        decoder_layers=4,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        max_position_embeddings=30,
        is_encoder_decoder=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.num_hidden_layers = decoder_layers
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.num_attention_heads = decoder_attention_heads
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.max_position_embeddings = max_position_embeddings

        self.use_cache = use_cache
        self.is_encoder_decoder = is_encoder_decoder

        self.scope = None
        self.decoder_key_length = decoder_seq_length
        self.base_model_out_len = 2
        self.decoder_attention_idx = 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size, dtype="int64")

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, 1, 1, self.decoder_seq_length], vocab_size=2, dtype="int64")

        lm_labels = None
        if self.parent.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size, dtype="int64")

        config = MBartConfig(
            embed_tokens=None,
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            decoder_layers=self.decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            decoder_attention_heads=self.decoder_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
        )

        return (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        )

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        attention_mask,
        lm_labels,
    ):
        # self.use_cache = True
        model = MBartDecoder(config)
        model.eval()

        encoder_output = paddle.randn(shape=input_ids.shape + [self.d_model])
        origin_cache = model.decoder.gen_cache(encoder_output)

        # first forward pass
        outputs = model(input_ids, cache=origin_cache, return_dict=self.parent.return_dict)
        # outputs_use_cache_conf = model(input_ids, return_dict=self.parent.return_dict)
        outputs_no_past = model(input_ids, cache=None, return_dict=self.parent.return_dict)

        # self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))  # didn't support using cache by config yet
        if not self.parent.return_dict:
            self.parent.assertTrue(len(outputs) == len((outputs_no_past,)) + 1)
        else:
            self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        past_key_values = outputs[1]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size, dtype="int64")

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        output_from_no_past = model(next_input_ids, return_dict=self.parent.return_dict)
        if self.parent.return_dict:
            output_from_no_past = output_from_no_past[0]
        output_from_past = model(next_tokens, cache=past_key_values, return_dict=self.parent.return_dict)[0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1], dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3)

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config,
        input_ids,
        attention_mask,
        lm_labels,
    ):
        model = MBartDecoder(config)
        model.eval()

        # create attention mask
        attn_mask = paddle.ones(input_ids.shape, dtype="int64")

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0
        attn_mask = attn_mask.unsqueeze([1, 2])

        encoder_output = paddle.randn(shape=input_ids.shape + [self.d_model])
        origin_cache = model.decoder.gen_cache(encoder_output)
        # first forward pass
        past_key_values = model(
            input_ids,
            # attention_mask=attn_mask,
            decoder_attention_mask=attn_mask,
            cache=origin_cache,
            return_dict=self.parent.return_dict,
        )[1]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size, dtype="int64")

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length, dtype="int64").item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size, dtype="int64").squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        attn_mask = paddle.concat(
            [attn_mask, paddle.ones((attn_mask.shape[0], 1, 1, 1), dtype="int64")],
            axis=-1,
        )
        # get two different outputs
        output_from_no_past = model(
            next_input_ids, decoder_attention_mask=attn_mask, return_dict=self.parent.return_dict
        )
        if self.parent.return_dict:
            output_from_no_past = output_from_no_past[0]
        output_from_past = model(
            next_tokens, decoder_attention_mask=attn_mask, cache=past_key_values, return_dict=self.parent.return_dict
        )[0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1], dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
)
class MBartStandaloneDecoderModelTest(ModelTesterMixin, GenerationTesterMixin, PaddleNLPModelTest):
    base_model_class = MBartModel

    all_model_classes = ()
    use_test_model_name_list = False
    all_generative_model_classes = {}
    is_encoder_decoder = False
    use_labels = False

    def setUp(self):
        self.model_tester = MBartStandaloneDecoderModelTester(self, is_training=False)

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_decoder_model_attn_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
        return

    def test_model_name_list(self):
        pass
