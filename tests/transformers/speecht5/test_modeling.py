# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
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

""" Testing suite for the PyTorch SpeechT5 model. """

import inspect
import unittest

import paddle

from paddlenlp.trainer.trainer_utils import set_seed
from paddlenlp.transformers import (
    SpeechT5Config,
    SpeechT5ForSpeechToSpeech,
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
    SpeechT5Model,
    SpeechT5Processor,
)

from ...testing_utils import slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


def prepare_inputs_dict(
    config,
    input_ids=None,
    input_values=None,
    decoder_input_ids=None,
    decoder_input_values=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if input_ids is not None:
        encoder_dict = {"input_ids": input_ids}
    else:
        encoder_dict = {"input_values": input_values}

    if decoder_input_ids is not None:
        decoder_dict = {"decoder_input_ids": decoder_input_ids}
    else:
        decoder_dict = {"decoder_input_values": decoder_input_values}

    if head_mask is None:
        head_mask = paddle.ones([config.encoder_layers, config.encoder_attention_heads])
    if decoder_head_mask is None:
        decoder_head_mask = paddle.ones([config.decoder_layers, config.decoder_attention_heads])
    if cross_attn_head_mask is None:
        cross_attn_head_mask = paddle.ones([config.decoder_layers, config.decoder_attention_heads])

    return {
        **encoder_dict,
        **decoder_dict,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class SpeechT5ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        vocab_size=81,
        hidden_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=4,
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

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length, self.hidden_size], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        decoder_input_values = floats_tensor([self.batch_size, self.seq_length, self.hidden_size], scale=1.0)
        decoder_attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()
        inputs_dict = prepare_inputs_dict(
            config,
            input_values=input_values,
            decoder_input_values=decoder_input_values,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return SpeechT5Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = SpeechT5Model(config=config)
        model.eval()

        input_values = inputs_dict["input_values"]
        attention_mask = inputs_dict["attention_mask"]
        decoder_input_values = inputs_dict["decoder_input_values"]

        result = model(input_values, attention_mask=attention_mask, decoder_input_values=decoder_input_values)
        self.parent.assertEqual(result.last_hidden_state.shape, [self.batch_size, self.seq_length, self.hidden_size])


class SpeechT5ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5Model,)
    pipeline_model_mapping = {
        "automatic-speech-recognition": SpeechT5ForSpeechToText,
        "feature-extraction": SpeechT5Model,
    }
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    use_test_model_name_list = False

    input_name = "input_values"

    def setUp(self):
        self.model_tester = SpeechT5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_values",
                "attention_mask",
                "decoder_input_values",
                "decoder_attention_mask",
            ]
            expected_arg_names.extend(
                ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                else ["encoder_outputs"]
            )
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    # this model has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # this model has no input embeddings
    def test_model_common_attributes(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
        pass


class SpeechT5ForSpeechToTextTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        encoder_seq_length=1024,  # speech is longer
        decoder_seq_length=7,
        is_training=False,
        hidden_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=4,
        conv_dim=(32, 32, 32),
        conv_stride=(4, 4, 4),
        conv_kernel=(8, 8, 8),
        conv_bias=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        vocab_size=81,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.vocab_size = vocab_size

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.encoder_seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.encoder_seq_length])

        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size).clip(2)
        decoder_attention_mask = random_attention_mask([self.batch_size, self.decoder_seq_length])

        config = self.get_config()
        inputs_dict = prepare_inputs_dict(
            config,
            input_values=input_values,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return SpeechT5Config(
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            conv_dim=self.conv_dim,
            conv_stride=self.conv_stride,
            conv_kernel=self.conv_kernel,
            conv_bias=self.conv_bias,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups,
            vocab_size=self.vocab_size,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = SpeechT5ForSpeechToText(config=config)
        model.eval()

        input_values = inputs_dict["input_values"]
        attention_mask = inputs_dict["attention_mask"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]

        result = model(input_values, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.decoder_seq_length, self.vocab_size])

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = SpeechT5ForSpeechToText(config=config).get_decoder()
        model.eval()
        input_ids = inputs_dict["decoder_input_ids"]
        attention_mask = inputs_dict["decoder_attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size).clip(2)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))


class SpeechT5ForSpeechToTextTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5ForSpeechToText,)
    all_generative_model_classes = (SpeechT5ForSpeechToText,)
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    use_test_model_name_list = False
    test_resize_embeddings = False

    input_name = "input_values"

    def setUp(self):
        self.model_tester = SpeechT5ForSpeechToTextTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)

            model.eval()

            subsampled_encoder_seq_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(
                encoder_seq_length
            )
            subsampled_encoder_key_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(
                encoder_key_length
            )

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)

            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )
            out_len = len(outputs)

            correct_outlen = 5

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            if "past_key_values" in outputs:
                correct_outlen += 1  # past_key_values have been returned

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    decoder_seq_length,
                    subsampled_encoder_key_length,
                ],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)

            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_values",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
            expected_arg_names.extend(
                ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                else ["encoder_outputs"]
            )
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            model.eval()

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length

            subsampled_seq_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(seq_length)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [subsampled_seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.weight",
                    "masked_spec_embed",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                ]
                if param.stop_gradient is False:
                    if any([x in name for x in uniform_init_parms]):
                        self.assertTrue(
                            -1.0 <= ((param.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # this model has no inputs_embeds
    def test_inputs_embeds(self):

        pass

    def test_resize_embeddings_untied(self):
        # TODO(wugaosheng): fix test_resize_embeddings_untied
        pass

    def test_resize_tokens_embeddings(self):
        # TODO(wugaosheng): fix test_resize_tokens_embeddings
        pass

    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
        pass

    # training is not supported yet
    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)


@slow
class SpeechT5ForSpeechToTextIntegrationTests(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_generation_librispeech(self):
        model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

        processor = self.default_processor

        input_speech = self._load_datasamples(1)

        input_values = processor(audio=input_speech, return_tensors="pd").input_values

        generated_ids = model.generate(input_values)
        generated_transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)

        EXPECTED_TRANSCRIPTIONS = [
            "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
        ]
        self.assertListEqual(generated_transcript, EXPECTED_TRANSCRIPTIONS)

    def test_generation_librispeech_batched(self):
        model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

        processor = self.default_processor

        input_speech = self._load_datasamples(4)

        inputs = processor(audio=input_speech, return_tensors="pd", padding=True)

        input_values = inputs.input_values
        attention_mask = inputs.attention_mask

        generated_ids = model.generate(input_values, attention_mask=attention_mask)
        generated_transcripts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        EXPECTED_TRANSCRIPTIONS = [
            "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
            "nor is mister quilter's manner less interesting than his matter",
            "he tells us that at this festive season of the year with christmas and rosebeaf looming before us"
            " similars drawn from eating and its results occur most readily to the mind",
            "he has grave doubts whether sir frederick latin's work is really greek after all and can discover in it"
            " but little of rocky ithica",
        ]
        self.assertListEqual(generated_transcripts, EXPECTED_TRANSCRIPTIONS)


class SpeechT5ForTextToSpeechTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=1024,  # speech is longer
        is_training=False,
        hidden_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=4,
        vocab_size=81,
        num_mel_bins=20,
        reduction_factor=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.reduction_factor = reduction_factor

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size).clip(2)
        attention_mask = random_attention_mask([self.batch_size, self.encoder_seq_length])

        decoder_input_values = floats_tensor([self.batch_size, self.decoder_seq_length, self.num_mel_bins], scale=1.0)
        decoder_attention_mask = random_attention_mask([self.batch_size, self.decoder_seq_length])

        config = self.get_config()
        inputs_dict = prepare_inputs_dict(
            config,
            input_ids=input_ids,
            decoder_input_values=decoder_input_values,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return SpeechT5Config(
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            vocab_size=self.vocab_size,
            num_mel_bins=self.num_mel_bins,
            reduction_factor=self.reduction_factor,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = SpeechT5ForTextToSpeech(config=config)
        model.eval()

        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]
        decoder_input_values = inputs_dict["decoder_input_values"]

        result = model(input_ids, attention_mask=attention_mask, decoder_input_values=decoder_input_values)
        self.parent.assertEqual(
            result.spectrogram.shape,
            [self.batch_size, self.decoder_seq_length * self.reduction_factor, self.num_mel_bins],
        )


class SpeechT5ForTextToSpeechTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5ForTextToSpeech,)
    all_generative_model_classes = (SpeechT5ForTextToSpeech,)
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    use_test_model_name_list = False

    input_name = "input_ids"

    def setUp(self):
        self.model_tester = SpeechT5ForTextToSpeechTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    # skipped because there is always dropout in SpeechT5SpeechDecoderPrenet
    def test_decoder_model_past_with_large_inputs(self):
        pass

    # skipped because there is always dropout in SpeechT5SpeechDecoderPrenet
    def test_determinism(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_ids",
                "attention_mask",
                "decoder_input_values",
                "decoder_attention_mask",
            ]
            expected_arg_names.extend(
                ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                else ["encoder_outputs"]
            )
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.weight",
                ]
                if param.stop_gradient is False:
                    if any([x in name for x in uniform_init_parms]):
                        self.assertTrue(
                            -1.0 <= ((param.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # this model has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # skipped because there is always dropout in SpeechT5SpeechDecoderPrenet
    def test_model_outputs_equivalence(self):
        pass

    # skipped because there is always dropout in SpeechT5SpeechDecoderPrenet
    def test_save_load(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
        pass

    @slow
    def test_torchscript_output_attentions(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    @slow
    def test_torchscript_output_hidden_state(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    @slow
    def test_torchscript_simple(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    # training is not supported yet
    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)


@slow
class SpeechT5ForTextToSpeechIntegrationTests(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

    def test_generation(self):
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

        processor = self.default_processor

        set_seed(555)  # make deterministic

        input_text = "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
        input_ids = processor(text=input_text, return_tensors="pd").input_ids

        generated_speech = model.generate_speech(input_ids)
        self.assertEqual(generated_speech.shape, (1820, model.config.num_mel_bins))


class SpeechT5ForSpeechToSpeechTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        encoder_seq_length=1024,  # speech is longer
        decoder_seq_length=1024,
        is_training=False,
        hidden_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=4,
        conv_dim=(32, 32, 32),
        conv_stride=(4, 4, 4),
        conv_kernel=(8, 8, 8),
        conv_bias=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        vocab_size=81,
        num_mel_bins=20,
        reduction_factor=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.reduction_factor = reduction_factor

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.encoder_seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.encoder_seq_length])

        decoder_input_values = floats_tensor([self.batch_size, self.decoder_seq_length, self.num_mel_bins], scale=1.0)
        decoder_attention_mask = random_attention_mask([self.batch_size, self.decoder_seq_length])

        config = self.get_config()
        inputs_dict = prepare_inputs_dict(
            config,
            input_values=input_values,
            decoder_input_values=decoder_input_values,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return SpeechT5Config(
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            conv_dim=self.conv_dim,
            conv_stride=self.conv_stride,
            conv_kernel=self.conv_kernel,
            conv_bias=self.conv_bias,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups,
            vocab_size=self.vocab_size,
            num_mel_bins=self.num_mel_bins,
            reduction_factor=self.reduction_factor,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = SpeechT5ForSpeechToSpeech(config=config)
        model.eval()

        input_values = inputs_dict["input_values"]
        attention_mask = inputs_dict["attention_mask"]
        decoder_input_values = inputs_dict["decoder_input_values"]

        result = model(input_values, attention_mask=attention_mask, decoder_input_values=decoder_input_values)
        self.parent.assertEqual(
            result.spectrogram.shape,
            [self.batch_size, self.decoder_seq_length * self.reduction_factor, self.num_mel_bins],
        )


class SpeechT5ForSpeechToSpeechTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5ForSpeechToSpeech,)
    all_generative_model_classes = (SpeechT5ForSpeechToSpeech,)
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    use_test_model_name_list = False

    input_name = "input_values"

    def setUp(self):
        self.model_tester = SpeechT5ForSpeechToSpeechTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    # skipped because there is always dropout in SpeechT5SpeechDecoderPrenet
    def test_decoder_model_past_with_large_inputs(self):
        pass

    # skipped because there is always dropout in SpeechT5SpeechDecoderPrenet
    def test_determinism(self):
        pass

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)

            model.eval()

            subsampled_encoder_seq_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(
                encoder_seq_length
            )
            subsampled_encoder_key_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(
                encoder_key_length
            )

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)

            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )
            out_len = len(outputs)

            correct_outlen = 5

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            if "past_key_values" in outputs:
                correct_outlen += 1  # past_key_values have been returned

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    decoder_seq_length,
                    subsampled_encoder_key_length,
                ],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)

            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_values",
                "attention_mask",
                "decoder_input_values",
                "decoder_attention_mask",
            ]
            expected_arg_names.extend(
                ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                else ["encoder_outputs"]
            )
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            model.eval()

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length

            subsampled_seq_length = model.speecht5.encoder.prenet._get_feat_extract_output_lengths(seq_length)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [subsampled_seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.weight",
                    "masked_spec_embed",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                ]
                if param.stop_gradient is False:
                    if any([x in name for x in uniform_init_parms]):
                        self.assertTrue(
                            -1.0 <= ((param.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # this model has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # this model has no input embeddings
    def test_model_common_attributes(self):
        pass

    # skipped because there is always dropout in SpeechT5SpeechDecoderPrenet
    def test_model_outputs_equivalence(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
        pass

    # skipped because there is always dropout in SpeechT5SpeechDecoderPrenet
    def test_save_load(self):
        pass

    @slow
    def test_torchscript_output_attentions(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    @slow
    def test_torchscript_output_hidden_state(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    @slow
    def test_torchscript_simple(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    # training is not supported yet
    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)


@slow
class SpeechT5ForSpeechToSpeechIntegrationTests(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_generation_librispeech(self):
        model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")

        processor = self.default_processor

        input_speech = self._load_datasamples(1)
        input_values = processor(audio=input_speech, return_tensors="pd").input_values

        speaker_embeddings = paddle.zeros((1, 512))
        generated_speech = model.generate_speech(input_values, speaker_embeddings=speaker_embeddings)

        self.assertEqual(generated_speech.shape[1], model.config.num_mel_bins)
        self.assertGreaterEqual(generated_speech.shape[0], 300)
        self.assertLessEqual(generated_speech.shape[0], 310)


class SpeechT5HifiGanTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        num_mel_bins=20,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.num_mel_bins = num_mel_bins

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.seq_length, self.num_mel_bins], scale=1.0)
        config = self.get_config()
        return config, input_values

    def get_config(self):
        return SpeechT5HifiGanConfig(
            model_in_dim=self.num_mel_bins,
        )

    def create_and_check_model(self, config, input_values):
        model = SpeechT5HifiGan(config=config)
        model.eval()
        result = model(input_values)
        self.parent.assertEqual(
            result.shape,
            [
                self.seq_length * 256,
            ],
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_values = self.prepare_config_and_inputs()
        inputs_dict = {"spectrogram": input_values}
        return config, inputs_dict


class SpeechT5HifiGanTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5HifiGan,)
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = False
    test_resize_position_embeddings = False
    test_head_masking = False
    test_mismatched_shapes = False
    test_missing_keys = False
    test_model_parallel = False
    is_encoder_decoder = False
    has_attentions = False
    use_test_model_name_list = False

    input_name = "spectrogram"

    def setUp(self):
        self.model_tester = SpeechT5HifiGanTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5HifiGanConfig)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        # self.config_tester.create_and_test_config_from_and_save_pretrained_subfolder()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "spectrogram",
            ]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    # this model does not output hidden states
    def test_hidden_states_output(self):
        pass

    # skip
    def test_initialization(self):
        pass

    # this model has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # this model has no input embeddings
    def test_model_common_attributes(self):
        pass

    # skip as this model doesn't support all arguments tested
    def test_model_outputs_equivalence(self):
        pass

    # this model does not output hidden states
    def test_retain_grad_hidden_states_attentions(self):
        pass

    # skip because it fails on automapping of SpeechT5HifiGanConfig
    def test_save_load_fast_init_from_base(self):
        pass

    # skip because it fails on automapping of SpeechT5HifiGanConfig
    def test_save_load_fast_init_to_base(self):
        pass

    def test_batched_inputs_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            model.eval()

            batched_inputs = inputs["spectrogram"].unsqueeze(0).tile([2, 1, 1])
            with paddle.no_grad():
                batched_outputs = model(batched_inputs)

            self.assertEqual(
                batched_inputs.shape[0], batched_outputs.shape[0], msg="Got different batch dims for input and output"
            )

    def test_unbatched_inputs_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            model.eval()

            with paddle.no_grad():
                outputs = model(inputs["spectrogram"])
            self.assertTrue(outputs.dim() == 1, msg="Got un-batched inputs but batched output")
