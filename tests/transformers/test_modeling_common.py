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

import copy
import inspect
import os
import random
import shutil
import tempfile
import unittest

import numpy as np
import paddle

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.env import MODEL_HOME

from ..testing_utils import slow


def ids_tensor(shape, vocab_size, dtype="int32"):
    #  Creates a random int32 tensor of the shape within the vocab size
    return paddle.randint(low=0, high=vocab_size, dtype=dtype, shape=shape)


def random_attention_mask(shape, dtype="int32"):
    attn_mask = ids_tensor(shape, vocab_size=2, dtype=dtype)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0):
    """Creates a random float32 tensor"""
    return scale * paddle.randn(shape, dtype="float32")


def check_two_model_parameter(first_model: PretrainedModel,
                              second_model: PretrainedModel):
    assert len(
        set(first_model.state_dict().keys()) -
        set(second_model.state_dict().keys())) == 0

    # random choice the keys to compare
    key = random.choice(list(first_model.state_dict().keys()))
    diff = first_model.state_dict()[key] - second_model.state_dict()[key]
    assert diff.sum().numpy().item() == 0


class ModelTesterMixin:
    model_tester = None
    base_model_class = None
    all_model_classes = ()
    all_generative_model_classes = ()
    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_mismatched_shapes = True
    test_missing_keys = True
    use_test_inputs_embeds = False
    use_test_model_name_list = True
    is_encoder_decoder = False
    has_attentions = True
    model_split_percents = [0.5, 0.7, 0.9]

    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class.__name__.endswith("ForMultipleChoice"):
            inputs_dict = {
                k: v.unsqueeze(1).expand(
                    shape=[-1, self.model_tester.num_choices, -1])
                if isinstance(v, paddle.Tensor) and v.ndim > 1 else v
                for k, v in inputs_dict.items()
            }
        return inputs_dict

    def _make_model_instance(self, config, model_class):
        if isinstance(config, PretrainedConfig):
            return model_class(config)
        if model_class == self.base_model_class:
            return model_class(**config)

        return model_class(self.base_model_class(**config))

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
        )
        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                outputs = model(
                    **self._prepare_for_class(inputs_dict, model_class))

            out_2 = outputs[0].numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.eval()
                with paddle.no_grad():
                    after_outputs = model(
                        **self._prepare_for_class(inputs_dict, model_class))

                # Make sure we don't have nans
                out_1 = after_outputs[0].numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
        )
        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                first = model(
                    **self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(
                    **self._prepare_for_class(inputs_dict, model_class))[0]

            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["input_ids"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @unittest.skip("Not implemented yet")
    def test_training(self):
        # TODO(guosheng): add more tests for training if loss is implemented
        pass

    @unittest.skip("Not implemented yet")
    def test_training_gradient_checkpointing(self):
        pass

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
        )
        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length",
                                     seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length",
                                     seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length",
                                     decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length",
                                     encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester,
                                                "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            signature = inspect.signature(model_class.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            if not all(
                    name in arg_names for name in
                ["output_attentions", "output_hidden_states", "return_dict"]):
                continue
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            inputs_dict["return_dict"] = True
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                outputs = model(
                    **self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if self.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions),
                             self.model_tester.num_hidden_layers)

            # TODO(guosheng): check that output_attentions also work using config

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [
                        self.model_tester.num_attention_heads,
                        encoder_seq_length, chunk_length, encoder_key_length
                    ],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        encoder_seq_length, encoder_key_length
                    ],
                )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                # Question Answering model returns start_logits and end_logits
                if model_class.__name__.endswith("ForQuestionAnswering"):
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions),
                                 self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length, decoder_key_length
                    ],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions),
                                 self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                outputs = model(
                    **self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if self.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions),
                             self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [
                        self.model_tester.num_attention_heads,
                        encoder_seq_length, chunk_length, encoder_key_length
                    ],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        encoder_seq_length, encoder_key_length
                    ],
                )

    def test_hidden_states_output(self):

        def check_hidden_states_output(inputs_dict, config, model_class):
            model = self._make_model_instance(config, model_class)
            model.eval()

            with paddle.no_grad():
                outputs = model(
                    **self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if self.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers",
                self.model_tester.num_hidden_layers + 1)
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(
                        self.model_tester,
                        "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if self.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester,
                                             "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
        )
        inputs_dict["return_dict"] = True
        for model_class in self.all_model_classes:
            signature = inspect.signature(model_class.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            if not all(
                    name in arg_names for name in
                ["output_attentions", "output_hidden_states", "return_dict"]):
                continue
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            # TODO(guosheng): check that output_hidden_states also work using config

    @unittest.skip("Not implemented")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_resize_position_vector_embeddings(self):
        if not self.test_resize_position_embeddings:
            return

        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = self._make_model_instance(config, model_class)

            if self.model_tester.is_training is False:
                model.eval()

            max_position_embeddings = config.max_position_embeddings

            # Retrieve the embeddings and clone theme
            if self.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings(
                )
                encoder_cloned_embeddings = encoder_model_embed.weight.clone()
                decoder_cloned_embeddings = decoder_model_embed.weight.clone()
            else:
                model_embed = model.get_position_embeddings()
                cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the position embeddings with a larger max_position_embeddings increases
            # the model's postion embeddings size
            model.resize_position_embeddings(max_position_embeddings + 10)
            self.assertEqual(model.config.max_position_embeddings,
                             max_position_embeddings + 10)

            # Check that it actually resizes the embeddings matrix
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings(
                )
                self.assertEqual(encoder_model_embed.weight.shape[0],
                                 encoder_cloned_embeddings.shape[0] + 10)
                self.assertEqual(decoder_model_embed.weight.shape[0],
                                 decoder_cloned_embeddings.shape[0] + 10)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0],
                                 cloned_embeddings.shape[0] + 10)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the position embeddings with a smaller max_position_embeddings decreases
            # the model's max_position_embeddings
            model.resize_position_embeddings(max_position_embeddings - 5)
            self.assertEqual(model.base_model.config["max_position_embeddings"],
                             max_position_embeddings - 5)

            # Check that it actually resizes the embeddings matrix
            if self.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings(
                )
                self.assertEqual(encoder_model_embed.weight.shape[0],
                                 encoder_cloned_embeddings.shape[0] - 5)
                self.assertEqual(decoder_model_embed.weight.shape[0],
                                 decoder_cloned_embeddings.shape[0] - 5)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0],
                                 cloned_embeddings.shape[0] - 5)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True

            if model.config.is_encoder_decoder:
                for p1, p2 in zip(encoder_cloned_embeddings,
                                  encoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
                for p1, p2 in zip(decoder_cloned_embeddings,
                                  decoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
            else:
                for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

    def test_resize_tokens_embeddings(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = self._make_model_instance(config, model_class)
            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config["vocab_size"]
            # Retrieve the embeddings and clone theme

            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.base_model.config["vocab_size"],
                             model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0],
                             cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.base_model.config["vocab_size"],
                             model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0],
                             cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"] = paddle.clip(inputs_dict["input_ids"],
                                                   max=model_vocab_size - 15 -
                                                   1)

            # make sure that decoder_input_ids are resized as well
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"] = paddle.clip(
                    inputs_dict["decoder_input_ids"],
                    max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if not paddle.equal_all(p1, p2).item():
                    models_equal = False
                    break

            self.assertTrue(models_equal)

    def test_inputs_embeds(self):
        # pass the test if don't need to test inputs embeddings
        if not self.use_test_inputs_embeds:
            return
        # get config for model and inputs_dict for model forward
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
        )
        # test all model classes
        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()

            inputs = copy.deepcopy(
                self._prepare_for_class(inputs_dict, model_class))

            with paddle.no_grad():
                ids_output = model(**inputs)

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids",
                                               encoder_input_ids)
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

            self.assertTrue(
                paddle.allclose(ids_output, embeds_output, rtol=1e-4,
                                atol=1e-4))

    def test_model_name_list(self):
        if not self.use_test_model_name_list:
            return
        config = self.model_tester.get_config()
        if isinstance(config, PretrainedConfig):
            model = self.base_model_class(config)
        else:
            model = self.base_model_class(**config)
        self.assertTrue(len(model.model_name_list) != 0)


class ModelTesterPretrainedMixin:
    base_model_class: PretrainedModel = None
    hf_remote_test_model_path: str = None
    paddlehub_remote_test_model_path: str = None

    @slow
    def test_model_from_pretrained_hf_hub(self):
        if self.hf_remote_test_model_path is None or self.base_model_class is None:
            return
        model = self.base_model_class.from_pretrained(
            self.hf_remote_test_model_path, from_hf_hub=True)
        self.assertIsNotNone(model)

    @slow
    def test_model_from_pretrained_paddle_hub(self):
        if self.paddlehub_remote_test_model_path is None or self.base_model_class is None:
            return
        model = self.base_model_class.from_pretrained(
            self.paddlehub_remote_test_model_path)
        self.assertIsNotNone(model)

    @slow
    def test_model_from_pretrained_with_cache_dir(self):
        for model_name in list(
                self.base_model_class.pretrained_init_configuration)[:1]:
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = str(tempdir)

                model = self.base_model_class.from_pretrained(model_name,
                                                              cache_dir=tempdir)
                self.assertIsNotNone(model)
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(
                            tempdir, self.base_model_class.
                            resource_files_names['model_state'])))
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(tempdir,
                                     self.base_model_class.model_config_file)))

    @slow
    def test_pretrained_save_and_load(self):
        """test the pretrained model save and load with two different ways: url-file-name & model_state name

            eg: `bert-base-uncased.pdparams` and `model_state.pdparams`
        """
        for model_name in list(
                self.base_model_class.pretrained_init_configuration)[:1]:
            model = self.base_model_class.from_pretrained(model_name)
            self.assertIsNotNone(model)

            # 1. save and load
            with tempfile.TemporaryDirectory() as tempdir:
                tempdirname = str(tempdir)
                model.save_pretrained(tempdirname)

                loaded_model = self.base_model_class.from_pretrained(
                    tempdirname)

                check_two_model_parameter(model, loaded_model)

            # 2. convert the weight file name
            with tempfile.TemporaryDirectory() as tempdir:
                tempdirname = str(tempdir) + '_old'

                shutil.copytree(
                    os.path.join(MODEL_HOME, model_name),
                    tempdirname,
                )
                files = os.listdir(tempdirname)

                saved_model_state_file = os.path.join(
                    tempdirname,
                    self.base_model_class.resource_files_names['model_state'])

                self.assertTrue(os.path.isfile(saved_model_state_file))

                # rename it to the old style: name of url, eg: model_state.pdparams -> bert-base-uncased.pdparams
                url = self.base_model_class.pretrained_resource_files_map[
                    'model_state'][model_name]
                pretrained_resource_file_name = os.path.split(url)[-1]
                target_file_path = os.path.join(tempdirname,
                                                pretrained_resource_file_name)

                shutil.copyfile(saved_model_state_file, target_file_path)
                os.remove(saved_model_state_file)

                new_model = self.base_model_class.from_pretrained(tempdirname)

                check_two_model_parameter(model, new_model)
