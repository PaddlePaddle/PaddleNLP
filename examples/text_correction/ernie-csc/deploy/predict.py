# -*- coding: UTF-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import numpy as np

import paddle
import paddlenlp as ppnlp
from paddle import inference
from paddlenlp.data import Stack, Tuple, Pad

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_file", type=str, required=True, default='./static_graph_params.pdmodel', help="The path to model info in static graph.")
parser.add_argument("--params_file", type=str, required=True, default='./static_graph_params.pdiparams', help="The path to parameters in static graph.")

parser.add_argument("--data_dir", type=str, default=None, help="The folder where the dataset is located.")
parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to init model.")
parser.add_argument("--batch_size", type=int, default=2, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--max_seq_len", type=int, default=64, help="Number of words of the longest seqence.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu.")

args = parser.parse_args()
# yapf: enable


class Predictor(object):
    def __init__(self, model_file, params_file, device, max_seq_length):
        self.max_seq_length = max_seq_length

        config = paddle.inference.Config(model_file, params_file)
        if device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)

        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]

        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])

    def predict(self,
                data,
                word_vocab,
                label_vocab,
                normlize_vocab,
                batch_size=1):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
                A Example object contains `text`(word_ids) and `seq_len`(sequence length).
            word_vocab(obj:`dict`): The word id (key) to word str (value) map.
            label_vocab(obj:`dict`): The label id (key) to label str (value) map.
            normlize_vocab(obj:`dict`): The fullwidth char (key) to halfwidth char (value) map.
            batch_size(obj:`int`, defaults to 1): The number of batch.

        Returns:
            results(obj:`dict`): All the predictions labels.
        """
        examples = []

        for text in data:
            tokens = list(text.strip())
            token_ids, length = convert_example(
                tokens,
                self.max_seq_length,
                word_vocab=word_vocab,
                normlize_vocab=normlize_vocab)
            examples.append((token_ids, length))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
            Pad(axis=0, pad_val=pinyin_vocab.token_to_idx[pinyin_vocab.pad_token]),  # pinyin
            Stack(axis=0, dtype='int64'),  # length
        ): [data for data in fn(samples)]

        trans_func = partial(
            convert_example,
            tokenizer=tokenizer,
            pinyin_vocab=pinyin_vocab,
            max_seq_length=args.max_seq_length,
            is_test=True)

        batches = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]

        results = []

        for batch in batches:
            token_ids, length = batchify_fn(batch)
            self.input_handles[0].copy_from_cpu(token_ids)
            self.input_handles[1].copy_from_cpu(length)
            self.predictor.run()
            preds = self.output_handle.copy_to_cpu()
            result = parse_result(token_ids, preds, length, word_vocab,
                                  label_vocab)
            results.extend(result)
        return results
