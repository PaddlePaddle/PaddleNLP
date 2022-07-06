# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import json
from functools import partial
import numpy as np
import paddle
from paddlenlp.transformers import AutoTokenizer
from paddle import inference
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset

sys.path.append('./')
from data import create_dataloader, transform_fn_dict
from data import convert_example, convert_chid_example

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True,
                    help="The directory to static model.")
parser.add_argument("--task_name", type=str, default='tnews', required=True,
                    help="tnews task name.")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences "
                         "longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=15, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument('--use_tensorrt', default=False, type=eval, choices=[True, False],
                    help='Enable to use tensorrt to speed up.')
parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "fp16", "int8"],
                    help='The tensorrt precision.')
parser.add_argument('--cpu_threads', default=10, type=int,
                    help='Number of threads to predict when using cpu.')
parser.add_argument('--enable_mkldnn', default=False, type=eval, choices=[True, False],
                    help='Enable to use mkldnn to speed up when using cpu.')
parser.add_argument("--pattern_id", default=0, type=int, help="pattern id of pet")
args = parser.parse_args()
# yapf: enable


class Predictor(object):

    def __init__(self,
                 model_dir,
                 device="gpu",
                 max_seq_length=128,
                 batch_size=32,
                 use_tensorrt=False,
                 precision="fp32",
                 cpu_threads=10,
                 enable_mkldnn=False):
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        model_file = os.path.join(model_dir, 'inference.pdmodel')
        params_file = os.path.join(model_dir, 'inference.pdiparams')
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(params_file))
        config = paddle.inference.Config(model_file, params_file)

        if device == "gpu":
            # set GPU configs accordingly
            # such as intialize the gpu memory, enable tensorrt
            config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32,
                "int8": inference.PrecisionType.Int8
            }
            precision_mode = precision_map[precision]

            if args.use_tensorrt:
                config.enable_tensorrt_engine(max_batch_size=batch_size,
                                              min_subgraph_size=30,
                                              precision_mode=precision_mode)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        elif device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)

        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])

    def predict(self, tokenizer, data_loader, label_normalize_dict):
        """
        Predicts the data labels.
        Args:
            data (obj:`List(str)`): The batch data whose each element is a raw text.
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        Returns:
            results(obj:`dict`): All the predictions labels.
        """

        normed_labels = [
            normalized_lable
            for origin_lable, normalized_lable in label_normalize_dict.items()
        ]

        origin_labels = [
            origin_lable
            for origin_lable, normalized_lable in label_normalize_dict.items()
        ]

        label_length = len(normed_labels[0])

        y_pred_labels = []

        for batch in data_loader:
            src_ids, token_type_ids, masked_positions = batch

            max_len = src_ids.shape[1]
            new_masked_positions = []

            for bs_index, mask_pos in enumerate(masked_positions.numpy()):
                for pos in mask_pos:
                    new_masked_positions.append(bs_index * max_len + pos)
            new_masked_positions = paddle.to_tensor(
                np.array(new_masked_positions).astype('int64'))
            self.input_handles[0].copy_from_cpu(src_ids.numpy().astype('int64'))
            self.input_handles[1].copy_from_cpu(
                token_type_ids.numpy().astype('int64'))
            self.input_handles[2].copy_from_cpu(
                new_masked_positions.numpy().astype('int64'))
            self.predictor.run()

            logits = self.output_handle.copy_to_cpu()
            logits = paddle.to_tensor(logits)
            softmax_fn = paddle.nn.Softmax()
            prediction_probs = softmax_fn(logits)
            batch_size = len(src_ids)
            vocab_size = prediction_probs.shape[1]
            prediction_probs = paddle.reshape(
                prediction_probs, shape=[batch_size, -1, vocab_size]).numpy()

            label_ids = np.array([
                tokenizer(label)["input_ids"][1:-1] for label in normed_labels
            ])

            y_pred = np.ones(shape=[batch_size, len(label_ids)])

            # Calculate joint distribution of candidate labels

            for index in range(label_length):
                y_pred *= prediction_probs[:, index, label_ids[:, index]]

            # Get max probs label's index
            y_pred_index = np.argmax(y_pred, axis=-1)

            for index in y_pred_index:
                y_pred_labels.append(origin_labels[index])

        return y_pred_labels


if __name__ == "__main__":
    label_normalize_json = os.path.join("./label_normalized",
                                        args.task_name + ".json")
    label_norm_dict = None
    with open(label_normalize_json, 'r', encoding="utf-8") as f:
        label_norm_dict = json.load(f)

    # Load test_ds for tnews leaderboard
    test_ds = load_dataset("fewclue", name=args.task_name, splits=("test"))

    # Task related transform operations, eg: numbert label -> text_label, english -> chinese
    transform_fn = partial(transform_fn_dict[args.task_name],
                           label_normalize_dict=label_norm_dict,
                           is_test=True,
                           pattern_id=args.pattern_id)
    test_ds = test_ds.map(transform_fn, lazy=False)
    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

    batchify_test_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"
            ),  # token_type_ids
        Stack(dtype="int64"),  # masked_positions
    ): [data for data in fn(samples)]
    trans_test_func = partial(convert_example,
                              tokenizer=tokenizer,
                              max_seq_length=128,
                              is_test=True)

    test_data_loader = create_dataloader(test_ds,
                                         mode='eval',
                                         batch_size=args.batch_size,
                                         batchify_fn=batchify_test_fn,
                                         trans_fn=trans_test_func)
    p = Predictor(args.model_dir)
    y = p.predict(tokenizer, test_data_loader, label_norm_dict)
    print(y)
