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

import numpy as np
import paddle
from scipy.special import softmax
from scipy import spatial
from paddle import inference
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.utils.log import logger

sys.path.append('.')

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True,
    help="The directory to static model.")

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

parser.add_argument("--benchmark", type=eval, default=False,
    help="To log some information about environment and running.")
parser.add_argument("--save_log_path", type=str, default="./log_output/",
    help="The file path to save log.")
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

        model_file = model_dir + "/inference.pdmodel"
        params_file = model_dir + "/inference.pdiparams"
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

        if args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(model_name="ernie-3.0-medium-zh",
                                               model_precision=precision,
                                               batch_size=self.batch_size,
                                               data_shape="dynamic",
                                               save_path=args.save_log_path,
                                               inference_config=config,
                                               pids=pid,
                                               process_name=None,
                                               gpu_ids=0,
                                               time_keys=[
                                                   'preprocess_time',
                                                   'inference_time',
                                                   'postprocess_time'
                                               ],
                                               warmup=0,
                                               logger=logger)

    def extract_embedding(self, data, tokenizer):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(str)`): The batch data whose each element is a raw text.
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.

        Returns:
            results(obj:`dict`): All the feature vectors.
        """
        if args.benchmark:
            self.autolog.times.start()

        examples = []
        for text in data:
            input_ids, segment_ids = convert_example(text, tokenizer)
            examples.append((input_ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        ): fn(samples)

        if args.benchmark:
            self.autolog.times.stamp()

        input_ids, segment_ids = batchify_fn(examples)
        self.input_handles[0].copy_from_cpu(input_ids)
        self.input_handles[1].copy_from_cpu(segment_ids)
        self.predictor.run()
        logits = self.output_handle.copy_to_cpu()
        if args.benchmark:
            self.autolog.times.stamp()

        if args.benchmark:
            self.autolog.times.end(stamp=True)

        return logits

    def predict(self, data, tokenizer):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(str)`): The batch data whose each element is a raw text.
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.

        Returns:
            results(obj:`dict`): All the predictions probs.
        """
        if args.benchmark:
            self.autolog.times.start()

        examples = []
        for idx, text in enumerate(data):
            input_ids, segment_ids = convert_example({idx: text[0]}, tokenizer)
            title_ids, title_segment_ids = convert_example({idx: text[1]},
                                                           tokenizer)
            examples.append(
                (input_ids, segment_ids, title_ids, title_segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        ): fn(samples)

        if args.benchmark:
            self.autolog.times.stamp()

        query_ids, query_segment_ids, title_ids, title_segment_ids = batchify_fn(
            examples)
        self.input_handles[0].copy_from_cpu(query_ids)
        self.input_handles[1].copy_from_cpu(query_segment_ids)
        self.predictor.run()
        query_logits = self.output_handle.copy_to_cpu()

        self.input_handles[0].copy_from_cpu(title_ids)
        self.input_handles[1].copy_from_cpu(title_segment_ids)
        self.predictor.run()
        title_logits = self.output_handle.copy_to_cpu()

        if args.benchmark:
            self.autolog.times.stamp()

        if args.benchmark:
            self.autolog.times.end(stamp=True)
        result = [
            float(1 - spatial.distance.cosine(arr1, arr2))
            for arr1, arr2 in zip(query_logits, title_logits)
        ]
        return result


if __name__ == "__main__":
    # Define predictor to do prediction.
    predictor = Predictor(args.model_dir, args.device, args.max_seq_length,
                          args.batch_size, args.use_tensorrt, args.precision,
                          args.cpu_threads, args.enable_mkldnn)

    # ErnieTinyTokenizer is special for ernie-tiny pretained model.
    output_emb_size = 256
    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
    id2corpus = {0: '国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据'}
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    res = predictor.extract_embedding(corpus_list, tokenizer)
    print(res.shape)
    print(res)
    corpus_list = [['中西方语言与文化的差异', '中西方文化差异以及语言体现中西方文化,差异,语言体现'],
                   ['中西方语言与文化的差异', '飞桨致力于让深度学习技术的创新与应用更简单']]
    res = predictor.predict(corpus_list, tokenizer)
    print(res)
