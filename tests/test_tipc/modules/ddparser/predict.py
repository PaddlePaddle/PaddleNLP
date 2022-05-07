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
import time
import copy
from functools import partial

import numpy as np
import paddle
from paddle import inference
import paddlenlp as ppnlp
from paddlenlp.data import Pad
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset

from data import create_dataloader, convert_example, load_vocab
from utils import flat_words, pad_sequence, istree, eisner

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True, help="The path to static model.")
parser.add_argument("--task_name", choices=["nlpcc13_evsam05_thu", "nlpcc13_evsam05_hit"], type=str, default="nlpcc13_evsam05_thu", help="Select the task.")
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
parser.add_argument("--batch_size", type=int, default=64, help="Numbers of examples a batch for training.")
parser.add_argument("--infer_output_file", type=str, default='infer_output.conll', help="The path to save infer results.")
parser.add_argument("--tree", type=bool, default=True, help="Ensure the output conforms to the tree structure.")
args = parser.parse_args()
# yapf: enable


def batchify_fn(batch):
    raw_batch = [raw for raw in zip(*batch)]
    batch = [pad_sequence(data) for data in raw_batch]
    return batch


def flat_words(words, pad_index=0):
    mask = words != pad_index
    lens = np.sum(mask.astype(int), axis=-1)
    position = np.cumsum(lens + (lens == 0).astype(int), axis=1) - 1
    lens = np.sum(lens, -1)
    words = words.ravel()[np.flatnonzero(words)]

    sequences = []
    idx = 0
    for l in lens:
        sequences.append(words[idx:idx + l])
        idx += l
    words = Pad(pad_val=pad_index)(sequences)

    max_len = words.shape[1]

    mask = (position >= max_len).astype(int)
    position = position * np.logical_not(mask) + mask * (max_len - 1)
    return words, position


def decode(s_arc, s_rel, mask, tree=True):

    lens = np.sum(mask.astype(int), axis=-1)
    arc_preds = np.argmax(s_arc, axis=-1)

    bad = [not istree(seq[:i + 1]) for i, seq in zip(lens, arc_preds)]
    if tree and any(bad):
        arc_preds[bad] = eisner(s_arc[bad], mask[bad])

    rel_preds = np.argmax(s_rel, axis=-1)
    rel_preds = [
        rel_pred[np.arange(len(arc_pred)), arc_pred]
        for arc_pred, rel_pred in zip(arc_preds, rel_preds)
    ]
    return arc_preds, rel_preds


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
        model_file = model_dir + "/inference.pdmodel"
        params_file = model_dir + "/inference.pdiparams"
        self.batch_size = batch_size
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(params_file))
        config = paddle.inference.Config(model_file, params_file)
        if device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32,
                "int8": inference.PrecisionType.Int8
            }
            precision_mode = precision_map[precision]
            if args.use_tensorrt:
                config.switch_ir_optim(True)
                config.enable_tensorrt_engine(
                    max_batch_size=batch_size,
                    min_subgraph_size=5,
                    precision_mode=precision_mode)
                min_batch_size, max_batch_size, opt_batch_size = 1, 32, 32
                min_seq_len, max_seq_len, opt_seq_len = 1, 128, 32
                min_input_shape = {
                    "input_ids": [min_batch_size, min_seq_len],
                    "tmp_4": [min_batch_size, min_seq_len],
                    "tmp_5": [min_batch_size, min_seq_len],
                    "words": [min_batch_size, min_seq_len],
                    "full_like_1.tmp_0": [min_batch_size, min_seq_len],
                    "token_type_ids": [min_batch_size, min_seq_len],
                    "unsqueeze2_0.tmp_0": [min_batch_size, 1, 1, min_seq_len]
                }
                max_input_shape = {
                    "input_ids": [max_batch_size, max_seq_len],
                    "tmp_4": [max_batch_size, max_seq_len],
                    "tmp_5": [max_batch_size, max_seq_len],
                    "words": [max_batch_size, max_seq_len],
                    "full_like_1.tmp_0": [max_batch_size, max_seq_len],
                    "token_type_ids": [max_batch_size, max_seq_len],
                    "unsqueeze2_0.tmp_0": [max_batch_size, 1, 1, max_seq_len]
                }
                opt_input_shape = {
                    "input_ids": [opt_batch_size, opt_seq_len],
                    "tmp_4": [opt_batch_size, opt_seq_len],
                    "tmp_5": [opt_batch_size, opt_seq_len],
                    "words": [opt_batch_size, opt_seq_len],
                    "full_like_1.tmp_0": [opt_batch_size, opt_seq_len],
                    "token_type_ids": [opt_batch_size, opt_seq_len],
                    "unsqueeze2_0.tmp_0": [opt_batch_size, 1, 1, opt_seq_len]
                }
                config.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)
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

        self.output_handle = [
            self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        ]

        if args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name='ddparser',
                model_precision=precision,
                batch_size=self.batch_size,
                data_shape="dynamic",
                save_path=args.save_log_path,
                inference_config=config,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def predict(self, data, vocabs):
        word_vocab, _, rel_vocab = vocabs
        word_pad_index = word_vocab.to_indices("[PAD]")
        word_bos_index = word_vocab.to_indices("[CLS]")
        word_eos_index = word_vocab.to_indices("[SEP]")
        examples = []
        for text in data:
            example = {
                "FORM": text["FORM"],
                "CPOS": text["CPOS"],
            }
            example = convert_example(
                example,
                vocabs=vocabs,
                mode="test", )
            examples.append(example)

        batches = [
            examples[idx:idx + args.batch_size]
            for idx in range(0, len(examples), args.batch_size)
        ]

        arcs, rels = [], []
        for batch in batches:
            words = batchify_fn(batch)[0]
            words, position = flat_words(words, word_pad_index)
            self.input_handles[0].copy_from_cpu(words)
            self.input_handles[1].copy_from_cpu(position)
            self.predictor.run()
            s_arc = self.output_handle[0].copy_to_cpu()
            s_rel = self.output_handle[1].copy_to_cpu()
            words = self.output_handle[2].copy_to_cpu()

            mask = np.logical_and(
                np.logical_and(words != word_pad_index,
                               words != word_bos_index),
                words != word_eos_index, )

            arc_preds, rel_preds = decode(s_arc, s_rel, mask, args.tree)

            arcs.extend([arc_pred[m] for arc_pred, m in zip(arc_preds, mask)])
            rels.extend([rel_pred[m] for rel_pred, m in zip(rel_preds, mask)])

        arcs = [[str(s) for s in seq] for seq in arcs]
        rels = [rel_vocab.to_tokens(seq) for seq in rels]
        return arcs, rels


if __name__ == "__main__":
    # Define predictor to do prediction.
    predictor = Predictor(args.model_dir, args.device)

    # Load vocabs from model file path
    vocabs = load_vocab(args.model_dir)

    test_ds = load_dataset(args.task_name, splits=["test"])
    test_ds_copy = copy.deepcopy(test_ds)

    pred_arcs, pred_rels = predictor.predict(test_ds, vocabs)

    with open(args.infer_output_file, 'w', encoding='utf-8') as out_file:
        for res, head, rel in zip(test_ds_copy, pred_arcs, pred_rels):
            res["HEAD"] = tuple(head)
            res["DEPREL"] = tuple(rel)
            res = '\n'.join('\t'.join(map(str, line))
                            for line in zip(*res.values())) + '\n'
            out_file.write("{}\n".format(res))
    out_file.close()
    print("Results saved!")
