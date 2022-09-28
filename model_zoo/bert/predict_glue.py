# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
from functools import partial

import paddle
from paddle import inference
from datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict

from run_glue import METRIC_CLASSES, MODEL_CLASSES, task_to_keys


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )
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
        choices=["gpu", "cpu", "xpu"],
        help="Device selected for inference.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.",
    )
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


class Predictor(object):

    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        config = paddle.inference.Config(args.model_path + ".pdmodel",
                                         args.model_path + ".pdiparams")
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        elif args.device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        config.switch_use_feed_fetch_ops(False)
        predictor = paddle.inference.create_predictor(config)
        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handles = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]
        return cls(predictor, input_handles, output_handles)

    def predict_batch(self, data):
        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field.numpy(
            ) if isinstance(input_field, paddle.Tensor) else input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def predict(self, dataset, collate_fn, batch_size=1):
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
        data_loader = paddle.io.DataLoader(dataset=dataset,
                                           batch_sampler=batch_sampler,
                                           collate_fn=collate_fn,
                                           num_workers=0,
                                           return_list=True)
        outputs = []
        for data in data_loader:
            output = self.predict_batch(data)
            outputs.append(output)
        return outputs


def main():
    args = parse_args()

    predictor = Predictor.create_predictor(args)

    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    test_ds = load_dataset('glue', args.task_name, split="test")
    tokenizer = tokenizer_class.from_pretrained(os.path.dirname(
        args.model_path))

    def preprocess_function(examples):
        # Tokenize the texts
        texts = ((examples[sentence1_key], ) if sentence2_key is None else
                 (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts, max_seq_len=args.max_seq_length)
        if "label" in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    test_ds = test_ds.map(preprocess_function)
    batchify_fn = lambda samples, fn=Dict({
        'input_ids':
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input
        'token_type_ids':
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"
            ),  # segment
    }): fn(samples)
    predictor.predict(test_ds,
                      batch_size=args.batch_size,
                      collate_fn=batchify_fn)


if __name__ == "__main__":
    main()
