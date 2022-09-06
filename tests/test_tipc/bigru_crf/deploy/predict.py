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
import time

import argparse
import numpy as np

import paddle
from paddlenlp.utils.log import logger
from paddle import inference
from paddlenlp.data import Stack, Tuple, Pad
# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_dir", type=str, default='./output', help="The path to parameters in static graph.")
parser.add_argument("--data_dir", type=str, default=None, help="The folder where the dataset is located.")
parser.add_argument("--batch_size", type=int, default=2, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu.")
parser.add_argument("--benchmark", type=eval, default=False, help="To log some information about environment and running.")
parser.add_argument("--save_log_path", type=str, default="./log_output/", help="The file path to save log.")
parser.add_argument('--use_tensorrt', default=False, type=eval, choices=[True, False], help='Enable to use tensorrt to speed up.')
parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "fp16"], help='The tensorrt precision.')
parser.add_argument('--cpu_threads', default=10, type=int, help='Number of threads to predict when using cpu.')
parser.add_argument('--enable_mkldnn', default=False, type=eval, choices=[True, False], help='Enable to use mkldnn to speed up when using cpu.')
args = parser.parse_args()
# yapf: enable


def normalize_token(token, normlize_vocab):
    """Normalize text from DBC case to SBC case"""
    if normlize_vocab:
        token = normlize_vocab.get(token, token)
    return token


def convert_tokens_to_ids(tokens,
                          vocab,
                          oov_replace_token=None,
                          normlize_vocab=None):
    """Convert tokens to token indexs"""
    token_ids = []
    oov_replace_token = vocab.get(
        oov_replace_token) if oov_replace_token else None
    for token in tokens:
        token = normalize_token(token, normlize_vocab)
        token_id = vocab.get(token, oov_replace_token)
        token_ids.append(token_id)

    return token_ids


def convert_example(tokens, max_seq_len, word_vocab, normlize_vocab=None):
    """Convert tokens of sequences to token ids"""
    tokens = tokens[:max_seq_len]

    token_ids = convert_tokens_to_ids(tokens,
                                      word_vocab,
                                      oov_replace_token="OOV",
                                      normlize_vocab=normlize_vocab)
    length = len(token_ids)
    return token_ids, length


def load_vocab(dict_path):
    """Load vocab from file"""
    vocab = {}
    reverse = None
    with open(dict_path, "r", encoding='utf8') as fin:
        for i, line in enumerate(fin):
            terms = line.strip("\n").split("\t")
            if len(terms) == 2:
                if reverse == None:
                    reverse = True if terms[0].isdigit() else False
                if reverse:
                    value, key = terms
                else:
                    key, value = terms
            elif len(terms) == 1:
                key, value = terms[0], i
            else:
                raise ValueError("Error line: %s in file: %s" %
                                 (line, dict_path))
            vocab[key] = value
    return vocab


def parse_result(words, preds, lengths, word_vocab, label_vocab):
    """ Parse padding result """
    batch_out = []
    id2word_dict = dict(zip(word_vocab.values(), word_vocab.keys()))
    id2label_dict = dict(zip(label_vocab.values(), label_vocab.keys()))
    for sent_index in range(len(lengths)):
        sent = [
            id2word_dict[index]
            for index in words[sent_index][:lengths[sent_index]]
        ]
        tags = [
            id2label_dict[index]
            for index in preds[sent_index][:lengths[sent_index]]
        ]

        sent_out = []
        tags_out = []
        parital_word = ""
        for ind, tag in enumerate(tags):
            # for the first word
            if parital_word == "":
                parital_word = sent[ind]
                tags_out.append(tag.split('-')[0])
                continue

            # for the beginning of word
            if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                sent_out.append(parital_word)
                tags_out.append(tag.split('-')[0])
                parital_word = sent[ind]
                continue

            parital_word += sent[ind]

        # append the last word, except for len(tags)=0
        if len(sent_out) < len(tags_out):
            sent_out.append(parital_word)

        batch_out.append([sent_out, tags_out])
    return batch_out


class Predictor(object):

    def __init__(self,
                 model_dir,
                 device="gpu",
                 max_seq_length=128,
                 batch_size=200,
                 use_tensorrt=False,
                 precision="fp32",
                 enable_mkldnn=False,
                 benchmark=False,
                 save_log_path=""):
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        model_file = os.path.join(model_dir, "inference.pdmodel")
        params_file = os.path.join(model_dir, "inference.pdiparams")
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(params_file))
        config = paddle.inference.Config(model_file, params_file)
        if device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": (inference.PrecisionType.Half, False),
                "fp32": (inference.PrecisionType.Float32, False),
                "int8": (inference.PrecisionType.Int8, True)
            }
            precision_mode, use_calib_mode = precision_map[precision]
            if use_tensorrt:
                config.enable_tensorrt_engine(max_batch_size=batch_size,
                                              min_subgraph_size=1,
                                              precision_mode=precision_mode,
                                              use_calib_mode=use_calib_mode)
                min_input_shape = {
                    # shape: [B, T, H]
                    "embedding_1.tmp_0": [batch_size, 1, 128],
                    # shape: [T, B, H]
                    "gru_0.tmp_0": [1, batch_size, 256],
                }
                max_input_shape = {
                    "embedding_1.tmp_0": [batch_size, 256, 128],
                    "gru_0.tmp_0": [256, batch_size, 256],
                }
                opt_input_shape = {
                    "embedding_1.tmp_0": [batch_size, 128, 128],
                    "gru_0.tmp_0": [128, batch_size, 256],
                }
                config.set_trt_dynamic_shape_info(min_input_shape,
                                                  max_input_shape,
                                                  opt_input_shape)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            if enable_mkldnn:
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
            kwargs = {
                "model_name":
                "bigru_crf",
                "model_precision":
                precision,
                "batch_size":
                self.batch_size,
                "data_shape":
                "dynamic",
                "save_path":
                save_log_path,
                "inference_config":
                config,
                "pids":
                pid,
                "process_name":
                None,
                "time_keys":
                ['preprocess_time', 'inference_time', 'postprocess_time'],
                "warmup":
                0,
                "logger":
                logger
            }
            if device == "gpu":
                kwargs["gpu_ids"] = 0
            self.autolog = auto_log.AutoLogger(**kwargs)

    def predict(self, data, word_vocab, label_vocab, normlize_vocab):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
                A Example object contains `text`(word_ids) and `seq_len`(sequence length).
            word_vocab(obj:`dict`): The word id (key) to word str (value) map.
            label_vocab(obj:`dict`): The label id (key) to label str (value) map.
            normlize_vocab(obj:`dict`): The fullwidth char (key) to halfwidth char (value) map.

        Returns:
            results(obj:`dict`): All the predictions labels.
        """
        if args.benchmark:
            self.autolog.times.start()
        examples = []

        for text in data:
            tokens = list(text.strip())
            token_ids, length = convert_example(tokens,
                                                self.max_seq_length,
                                                word_vocab=word_vocab,
                                                normlize_vocab=normlize_vocab)
            examples.append((token_ids, length))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=0),  # input
            Stack(axis=0),  # length
        ): fn(samples)

        batches = [
            examples[idx:idx + self.batch_size]
            for idx in range(0, len(examples), self.batch_size)
        ]

        results = []
        preds_list = []
        token_ids_list = []
        length_list = []

        for batch in batches:
            token_ids, length = batchify_fn(batch)
            token_ids_list.append(token_ids)
            length_list.append(length)
        # Preprocess time
        if args.benchmark:
            self.autolog.times.stamp()

        for token_ids, length in zip(token_ids_list, length_list):
            self.input_handles[0].copy_from_cpu(token_ids)
            self.input_handles[1].copy_from_cpu(length)
            self.predictor.run()
            preds = self.output_handle.copy_to_cpu()
            preds_list.append(preds)
        # inference time
        if args.benchmark:
            self.autolog.times.stamp()

        for token_ids, length, preds in zip(token_ids_list, length_list,
                                            preds_list):
            result = parse_result(token_ids, preds, length, word_vocab,
                                  label_vocab)
            results.extend(result)
        # Postprocess time
        if args.benchmark:
            self.autolog.times.end(stamp=True)
        return results, preds_list, length_list


if __name__ == "__main__":
    word_vocab = load_vocab(os.path.join(args.data_dir, 'word.dic'))
    label_vocab = load_vocab(os.path.join(args.data_dir, 'tag.dic'))
    normlize_vocab = load_vocab(os.path.join(args.data_dir, 'q2b.dic'))
    infer_ds = []
    with open(os.path.join(args.data_dir, 'infer.tsv'), "r",
              encoding="utf-8") as fp:
        for line in fp.readlines():
            infer_ds += [line.strip()]
    predictor = Predictor(args.model_dir, args.device, args.max_seq_len,
                          args.batch_size, args.use_tensorrt, args.precision,
                          args.enable_mkldnn)
    results, preds_list, length_list = predictor.predict(
        infer_ds, word_vocab, label_vocab, normlize_vocab)

    idx = 0
    for batch_preds, batch_length in zip(preds_list, length_list):
        for preds, length in zip(batch_preds, batch_length):
            print("{}\t{}".format(idx, preds[:length].tolist()), flush=True)
            idx += 1

    if args.benchmark:
        predictor.autolog.report()
