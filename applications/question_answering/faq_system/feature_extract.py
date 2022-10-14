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

import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
from scipy.special import softmax

import paddle
from paddle import inference
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger

from model import SimCSE
from data import convert_example

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True,
    help="The directory to static model.")

parser.add_argument("--corpus_file", type=str, required=True,
    help="The corpus_file path.")

parser.add_argument("--max_seq_length", default=64, type=int,
    help="The maximum total input sequence length after tokenization. Sequences "
    "longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int,
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

args = parser.parse_args()
# yapf: enable


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    pad_to_max_seq_len=False):
    """
    Builds model inputs from a sequence.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """

    result = []
    for key, text in example.items():
        encoded_inputs = tokenizer(text=text,
                                   max_seq_len=max_seq_length,
                                   pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result


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

        model_file = model_dir + "/inference.get_pooled_embedding.pdmodel"
        params_file = model_dir + "/inference.get_pooled_embedding.pdiparams"
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

    def predict(self, data, tokenizer):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(str)`): The batch data whose each element is a raw text.
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.

        Returns:
            results(obj:`dict`): All the predictions labels.
        """

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'
                ),  # segment
        ): fn(samples)

        all_embeddings = []
        examples = []
        for idx, text in enumerate(tqdm(data)):
            input_ids, segment_ids = convert_example(
                text,
                tokenizer,
                max_seq_length=self.max_seq_length,
                pad_to_max_seq_len=True)
            examples.append((input_ids, segment_ids))
            if (len(examples) >= self.batch_size):
                input_ids, segment_ids = batchify_fn(examples)
                self.input_handles[0].copy_from_cpu(input_ids)
                self.input_handles[1].copy_from_cpu(segment_ids)
                self.predictor.run()
                logits = self.output_handle.copy_to_cpu()
                all_embeddings.append(logits)
                examples = []

        if (len(examples) > 0):
            input_ids, segment_ids = batchify_fn(examples)
            self.input_handles[0].copy_from_cpu(input_ids)
            self.input_handles[1].copy_from_cpu(segment_ids)
            self.predictor.run()
            logits = self.output_handle.copy_to_cpu()
            all_embeddings.append(logits)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        np.save('corpus_embedding', all_embeddings)


def read_text(file_path):
    file = open(file_path)
    id2corpus = {}
    for idx, data in enumerate(file.readlines()):
        id2corpus[idx] = data.strip()
    return id2corpus


if __name__ == "__main__":
    predictor = Predictor(args.model_dir, args.device, args.max_seq_length,
                          args.batch_size, args.use_tensorrt, args.precision,
                          args.cpu_threads, args.enable_mkldnn)

    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
    id2corpus = read_text(args.corpus_file)

    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    predictor.predict(corpus_list, tokenizer)
