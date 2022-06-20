# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from scipy.special import softmax

import paddle
from paddle import inference
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    args = parser.parse_args()
    return args


def convert_example(example, tokenizer, label_list, max_seq_length=128):
    text = example
    encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    segment_ids = encoded_inputs["token_type_ids"]

    return input_ids, segment_ids


class Predictor(object):

    def __init__(self, predictor, input_handles, output_handle, tokenizer,
                 max_seq_length):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handle = output_handle
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    @classmethod
    def create_predictor(cls, args):
        max_seq_length = args.max_seq_length
        config = paddle.inference.Config(args.model_path + ".pdmodel",
                                         args.model_path + ".pdiparams")
        if args.device == "gpu":
            # Set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif args.device == "cpu":
            # Set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        elif args.device == "xpu":
            # Set XPU configs accordingly
            config.enable_xpu(100)
        config.switch_use_feed_fetch_ops(False)
        predictor = paddle.inference.create_predictor(config)
        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handle = predictor.get_output_handle(
            predictor.get_output_names()[0])
        tokenizer = BertTokenizer.from_pretrained(
            os.path.dirname(args.model_path))

        return cls(predictor, input_handles, output_handle, tokenizer,
                   max_seq_length)

    def predict(self, data, label_map, batch_size=1):
        examples = []
        for text in data:
            input_ids, segment_ids = convert_example(
                text,
                self.tokenizer,
                label_list=label_map.values(),
                max_seq_length=self.max_seq_length)
            examples.append((input_ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"
                ),  # input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"
                ),  # segment
        ): fn(samples)

        # Seperates data into some batches.
        batches = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]

        outputs = []
        results = []
        for batch in batches:
            input_ids, segment_ids = batchify_fn(batch)
            self.input_handles[0].copy_from_cpu(input_ids)
            self.input_handles[1].copy_from_cpu(segment_ids)
            self.predictor.run()
            logits = self.output_handle.copy_to_cpu()
            probs = softmax(logits, axis=1)
            idx = np.argmax(probs, axis=1)
            idx = idx.tolist()
            labels = [label_map[i] for i in idx]
            outputs.extend(probs)
            results.extend(labels)
        return outputs, results


def main():
    args = parse_args()
    predictor = Predictor.create_predictor(args)

    data = [
        'against shimmering cinematography that lends the setting the ethereal beauty of an asian landscape painting',
        'the situation in a well-balanced fashion',
        'at achieving the modest , crowd-pleasing goals it sets for itself',
        'so pat it makes your teeth hurt',
        'this new jangle of noise , mayhem and stupidity must be a serious contender for the title .'
    ]
    label_map = {0: 'negative', 1: 'positive'}

    outputs, results = predictor.predict(data, label_map)
    for idx, text in enumerate(data):
        print(
            'Data: {} \n Label: {} \n Negative prob: {} \n Positive prob: {} \n '
            .format(text, results[idx], outputs[idx][0], outputs[idx][1]))


if __name__ == "__main__":
    main()
