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
from functools import partial

import paddle
from paddle import inference
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_dir", type=str, default='./output', help="The path to parameters in static graph.")
parser.add_argument("--data_dir", type=str, default="./waybill_ie/data", help="The folder where the dataset is located.")
parser.add_argument("--batch_size", type=int, default=200, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu.")
parser.add_argument('--use_tensorrt', default=False, type=eval, choices=[True, False], help='Enable to use tensorrt to speed up.')
parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "fp16", "int8"], help='The tensorrt precision.')
parser.add_argument('--cpu_threads', default=10, type=int, help='Number of threads to predict when using cpu.')
parser.add_argument('--enable_mkldnn', default=False, type=eval, choices=[True, False], help='Enable to use mkldnn to speed up when using cpu.')
parser.add_argument("--benchmark", type=eval, default=False, help="To log some information about environment and running.")
parser.add_argument("--save_log_path", type=str, default="./log_output/", help="The file path to save log.")
args = parser.parse_args()
# yapf: enable


def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            key = line.strip('\n')
            vocab[key] = i
            i += 1
    return vocab


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


def parse_decodes(sentences, predictions, lengths, label_vocab):
    """Parse the padding result

    Args:
        sentences (list): the tagging sentences.
        predictions (list): the prediction tags.
        lengths (list): the valid length of each sentence.
        label_vocab (dict): the label vocab.

    Returns:
        outputs (list): the formatted output.
    """
    predictions = [x for batch in predictions for x in batch]
    lengths = [x for batch in lengths for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lengths):
        sent = sentences[idx][:end]
        tags = [id_label[x] for x in predictions[idx][:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split('-')[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


def convert_to_features(example, tokenizer):
    tokens = example[0]
    tokenized_input = tokenizer(tokens,
                                return_length=True,
                                is_split_into_words=True)
    # Token '[CLS]' and '[SEP]' will get label 'O'
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len']


def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as fp:
        next(fp)  # Skip header
        for line in fp.readlines():
            words, labels = line.strip('\n').split('\t')
            words = words.split('\002')
            labels = labels.split('\002')
            yield words, labels


class Predictor(object):

    def __init__(self,
                 model_dir,
                 device="gpu",
                 batch_size=200,
                 use_tensorrt=False,
                 precision="fp32",
                 enable_mkldnn=False,
                 benchmark=False,
                 save_log_path=""):
        self.batch_size = batch_size
        model_file = os.path.join(model_dir, "inference.pdmodel")
        param_file = os.path.join(model_dir, "inference.pdiparams")
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(param_file):
            raise ValueError("not find params file path {}".format(param_file))
        config = paddle.inference.Config(model_file, param_file)
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

            if use_tensorrt:
                config.enable_tensorrt_engine(max_batch_size=batch_size,
                                              min_subgraph_size=30,
                                              precision_mode=precision_mode)
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
            self.autolog = auto_log.AutoLogger(model_name="ernie-3.0-medium-zh",
                                               model_precision=precision,
                                               batch_size=self.batch_size,
                                               data_shape="dynamic",
                                               save_path=save_log_path,
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

    def predict(self, dataset, batchify_fn, tokenizer, label_vocab):
        if args.benchmark:
            self.autolog.times.start()
        all_preds = []
        all_lens = []
        num_of_examples = len(dataset)
        trans_func = partial(convert_to_features, tokenizer=tokenizer)
        start_idx = 0
        while start_idx < num_of_examples:
            end_idx = start_idx + self.batch_size
            end_idx = end_idx if end_idx < num_of_examples else num_of_examples
            batch_data = [
                trans_func(example) for example in dataset[start_idx:end_idx]
            ]

            if args.benchmark:
                self.autolog.times.stamp()
            input_ids, segment_ids, lens = batchify_fn(batch_data)
            self.input_handles[0].copy_from_cpu(input_ids)
            self.input_handles[1].copy_from_cpu(segment_ids)
            self.predictor.run()
            logits = self.output_handle.copy_to_cpu()

            if args.benchmark:
                self.autolog.times.stamp()
            preds = np.argmax(logits, axis=-1)
            # Drop CLS prediction
            preds = preds[:, 1:]
            all_preds.append(preds)
            all_lens.append(lens)

            start_idx += self.batch_size

        if args.benchmark:
            self.autolog.times.end(stamp=True)
        sentences = [example[0] for example in dataset.data]
        results = parse_decodes(sentences, all_preds, all_lens, label_vocab)
        return results


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
    test_ds = load_dataset(read,
                           data_path=os.path.join(args.data_dir, 'test.txt'),
                           lazy=False)
    label_vocab = load_dict(os.path.join(args.data_dir, 'tag.dic'))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # token_type_ids
        Stack(dtype='int64'),  # seq_len
    ): fn(samples)

    predictor = Predictor(args.model_dir, args.device, args.batch_size,
                          args.use_tensorrt, args.precision, args.enable_mkldnn,
                          args.benchmark, args.save_log_path)

    results = predictor.predict(test_ds, batchify_fn, tokenizer, label_vocab)
    print("\n".join(results))
    if args.benchmark:
        predictor.autolog.report()
