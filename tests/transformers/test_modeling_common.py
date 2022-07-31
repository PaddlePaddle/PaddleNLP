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
import copy

import tempfile
import numpy as np
import paddle

global_rng = random.Random()


def ids_tensor(shape, vocab_size):
    #  Creates a random int32 tensor of the shape within the vocab size
    return paddle.randint(low=0, high=vocab_size, dtype="int32", shape=shape)


def random_attention_mask(shape):
    attn_mask = ids_tensor(shape, vocab_size=2)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0):
    """Creates a random float32 tensor"""
    return scale * paddle.randn(shape, dtype="float32")


class ModelTesterMixin:
    model_tester = None
    base_model_class = None
    all_model_classes = ()
    all_generative_model_classes = ()
    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_mismatched_shapes = True
    test_missing_keys = True
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

    def test_save_load(self):
        config, input_ids, token_type_ids, input_mask = self.model_tester.prepare_config_and_inputs(
        )
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        for model_class in self.all_model_classes:
            if model_class == self.base_model_class:
                model = model_class(**config)
            else:
                model = model_class(self.base_model_class(**config))
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

    def test_resize_tokens_embeddings(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            if model_class == self.base_model_class:
                model = model_class(**config)
            else:
                model = model_class(self.base_model_class(**config))

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

    def test_model_name_list(self):
        config = self.model_tester.get_config()
        model = self.base_model_class(**config)
        self.assertTrue(len(model.model_name_list) != 0)
