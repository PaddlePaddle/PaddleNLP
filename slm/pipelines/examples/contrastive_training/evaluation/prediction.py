# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys

import numpy as np
import paddle

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import AutoTokenizer

sys.path.append(os.path.abspath("."))
from models.modeling import BiEncoderModel


class Eval_modle:
    def __init__(
        self,
        model: str = None,
        batch_size: int = 1,
        max_seq_len: int = 512,
        return_tensors: str = "np",
        model_type: str = "bloom",
    ):
        self.model = model
        self.batch_size = batch_size
        self.return_tensors = return_tensors
        self.model_type = model_type
        self._construct_model()
        self._construct_tokenizer()

    def _construct_model(self):
        """
        Construct the inference model for the predictor.
        """
        if self.model_type in ["bert", "roberta", "ernie"]:
            self._model = BiEncoderModel.from_pretrained(
                model_name_or_path=self.model,
                normalized=True,
                sentence_pooling_method="cls",
            )
            print(f"loading checkpoints {self.model}")
        else:
            raise NotImplementedError

        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._tokenizer.padding_side = "right"
        self.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
        # Fix windows dtype bug
        self._collator = DataCollatorWithPadding(self._tokenizer, return_tensors="pd")

    def _batchify(self, data, batch_size, max_seq_len=None):
        """
        Generate input batches.
        """

        def _parse_batch(batch_examples, max_seq_len=None):
            if isinstance(batch_examples[0], str):
                to_tokenize = [batch_examples]
            to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
            if max_seq_len is None:
                max_seq_len = self.max_seq_len
            tokenized_inputs = self._tokenizer(
                to_tokenize[0],
                padding=True,
                truncation=True,
                max_seq_len=max_seq_len,
                return_attention_mask=True,
            )
            return tokenized_inputs

        # Seperates data into some batches.
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        one_batch = []
        for example in range(len(data)):
            one_batch.append(data[example])
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch, max_seq_len)
                one_batch = []
        if one_batch:
            yield _parse_batch(one_batch, max_seq_len)

    def _check_input_text(self, inputs):
        """
        Check whether the input text meet the requirement.
        """
        # inputs = inputs[0]
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError("Invalid inputs, input text should not be empty text, please check your input.")
            inputs = [inputs]
        elif isinstance(inputs, list):
            if not (isinstance(inputs[0], str) and len(inputs[0].strip()) > 0):
                raise TypeError(
                    "Invalid inputs, input text should be list of str, and first element of list should not be empty text."
                )
        else:
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, but type of {} found!".format(type(inputs))
            )
        return inputs

    def _preprocess(self, inputs, batch_size=None, max_seq_len=None, **kwargs):
        """
        Transform the raw inputs to the model inputs, two steps involved:
           1) Transform the raw text/image to token ids/pixel_values.
           2) Generate the other model inputs from the raw text/image and token ids/pixel_values.
        """
        inputs = self._check_input_text(inputs)
        if batch_size is None:
            batch_size = self.batch_size
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        batches = self._batchify(inputs, batch_size, max_seq_len)
        outputs = {"batches": batches, "inputs": inputs}
        return outputs

    def _run_model(self, inputs, **kwargs):
        all_feats = []
        with paddle.no_grad():
            for batch_inputs in inputs["batches"]:
                batch_inputs = self._collator(batch_inputs)
                token_embeddings = self._model.encode(batch_inputs)
                all_feats.append(token_embeddings.detach().cpu().numpy())
            return all_feats

    def _postprocess(self, inputs):
        inputs = np.concatenate(inputs, axis=0)
        return inputs

    def run(self, *args, **kwargs):
        inputs = self._preprocess(*args, **kwargs)
        outputs = self._run_model(inputs, **kwargs)
        results = self._postprocess(outputs)
        return results
