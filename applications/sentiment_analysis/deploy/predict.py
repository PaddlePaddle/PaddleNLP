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

import sys

sys.path.append("../")

import os
import json
import copy
import argparse
import numpy as np
from functools import partial
from collections import defaultdict
import paddle
from paddle import inference
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import SkepTokenizer
from utils import decoding, read_test_file, load_dict
from extraction.data import convert_example_to_feature as convert_example_to_feature_ext
from classification.data import convert_example_to_feature as convert_example_to_feature_cls


class Predictor(object):

    def __init__(self, args):
        self.args = args
        self.ext_predictor, self.ext_input_handles, self.ext_output_hanle = self.create_predictor(
            args.ext_model_path)
        print(f"ext_model_path: {args.ext_model_path}, {self.ext_predictor}")
        self.cls_predictor, self.cls_input_handles, self.cls_output_hanle = self.create_predictor(
            args.cls_model_path)
        self.ext_label2id, self.ext_id2label = load_dict(args.ext_label_path)
        self.cls_label2id, self.cls_id2label = load_dict(args.cls_label_path)
        self.tokenizer = SkepTokenizer.from_pretrained(args.base_model_name)

    def create_predictor(self, model_path):
        model_file = model_path + ".pdmodel"
        params_file = model_path + ".pdiparams"
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(params_file))
        config = paddle.inference.Config(model_file, params_file)

        if self.args.device == "gpu":
            # set GPU configs accordingly
            # such as intialize the gpu memory, enable tensorrt
            config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32,
                "int8": inference.PrecisionType.Int8
            }
            precision_mode = precision_map[args.precision]

            if args.use_tensorrt:
                config.enable_tensorrt_engine(
                    max_batch_size=self.args.batch_size,
                    min_subgraph_size=30,
                    precision_mode=precision_mode)
        elif self.args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        elif self.args.device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)

        config.switch_use_feed_fetch_ops(False)
        predictor = paddle.inference.create_predictor(config)
        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handle = predictor.get_output_handle(
            predictor.get_output_names()[0])

        return predictor, input_handles, output_handle

    def predict_ext(self, args):
        ori_test_ds = load_dataset(read_test_file,
                                   data_path=args.test_path,
                                   lazy=False)
        trans_func = partial(convert_example_to_feature_ext,
                             tokenizer=self.tokenizer,
                             label2id=self.ext_label2id,
                             max_seq_len=args.ext_max_seq_len,
                             is_test=True)
        test_ds = copy.copy(ori_test_ds).map(trans_func, lazy=False)
        batch_list = [
            test_ds[idx:idx + args.batch_size]
            for idx in range(0, len(test_ds), args.batch_size)
        ]

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"),
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype="int64"
                ), Stack(dtype="int64")): fn(samples)

        results = []
        for bid, batch_data in enumerate(batch_list):
            input_ids, token_type_ids, seq_lens = batchify_fn(batch_data)
            self.ext_input_handles[0].copy_from_cpu(input_ids)
            self.ext_input_handles[1].copy_from_cpu(token_type_ids)
            self.ext_predictor.run()
            logits = self.ext_output_hanle.copy_to_cpu()

            predictions = logits.argmax(axis=2)
            for eid, (seq_len,
                      prediction) in enumerate(zip(seq_lens, predictions)):
                idx = bid * args.batch_size + eid
                tag_seq = [
                    self.ext_id2label[idx] for idx in prediction[:seq_len][1:-1]
                ]
                text = ori_test_ds[idx]["text"]
                aps = decoding(text[:args.ext_max_seq_len - 2], tag_seq)
                for aid, ap in enumerate(aps):
                    aspect, opinions = ap[0], list(set(ap[1:]))
                    aspect_text = self._concate_aspect_and_opinion(
                        text, aspect, opinions)
                    results.append({
                        "id": str(idx) + "_" + str(aid),
                        "aspect": aspect,
                        "opinions": opinions,
                        "text": text,
                        "aspect_text": aspect_text
                    })
        return results

    def predict_cls(self, args, ext_results):
        test_ds = MapDataset(ext_results)
        trans_func = partial(convert_example_to_feature_cls,
                             tokenizer=self.tokenizer,
                             label2id=self.cls_label2id,
                             max_seq_len=args.cls_max_seq_len,
                             is_test=True)
        test_ds = test_ds.map(trans_func, lazy=False)
        batch_list = [
            test_ds[idx:idx + args.batch_size]
            for idx in range(0, len(test_ds), args.batch_size)
        ]

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"),
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype="int64"
                ), Stack(dtype="int64")): fn(samples)

        results = []
        for batch_data in batch_list:
            input_ids, token_type_ids, _ = batchify_fn(batch_data)
            self.cls_input_handles[0].copy_from_cpu(input_ids)
            self.cls_input_handles[1].copy_from_cpu(token_type_ids)
            self.cls_predictor.run()
            logits = self.cls_output_hanle.copy_to_cpu()

            predictions = logits.argmax(axis=1).tolist()
            results.extend(predictions)

        return results

    def post_process(self, args, ext_results, cls_results):
        assert len(ext_results) == len(cls_results)

        collect_dict = defaultdict(list)
        for ext_result, cls_result in zip(ext_results, cls_results):
            ext_result["sentiment_polarity"] = self.cls_id2label[cls_result]
            eid, _ = ext_result["id"].split("_")
            collect_dict[eid].append(ext_result)

        sentiment_results = []
        for eid in collect_dict.keys():
            sentiment_result = {}
            ap_list = []
            for idx, single_ap in enumerate(collect_dict[eid]):
                if idx == 0:
                    sentiment_result["text"] = single_ap["text"]
                ap_list.append({
                    "aspect":
                    single_ap["aspect"],
                    "opinions":
                    single_ap["opinions"],
                    "sentiment_polarity":
                    single_ap["sentiment_polarity"]
                })
            sentiment_result["ap_list"] = ap_list
            sentiment_results.append(sentiment_result)

        with open(args.save_path, "w", encoding="utf-8") as f:
            for sentiment_result in sentiment_results:
                f.write(json.dumps(sentiment_result, ensure_ascii=False) + "\n")
        print(
            f"sentiment analysis results has been saved to path: {args.save_path}"
        )

    def predict(self, args):
        ext_results = self.predict_ext(args)
        cls_results = self.predict_cls(args, ext_results)
        self.post_process(args, ext_results, cls_results)

    def _concate_aspect_and_opinion(self, text, aspect, opinion_words):
        aspect_text = ""
        for opinion_word in opinion_words:
            if text.find(aspect) <= text.find(opinion_word):
                aspect_text += aspect + opinion_word + "，"
            else:
                aspect_text += opinion_word + aspect + "，"
        aspect_text = aspect_text[:-1]

        return aspect_text


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", default='skep_ernie_1.0_large_ch', type=str, help="Base model name, SKEP used by default", )
    parser.add_argument("--ext_model_path", type=str, default=None, help="The path of extraction model path that you want to load.")
    parser.add_argument("--cls_model_path", type=str, default=None, help="The path of classification model path that you want to load.")
    parser.add_argument("--ext_label_path", type=str, default=None, help="The path of extraction label dict.")
    parser.add_argument("--cls_label_path", type=str, default=None, help="The path of classification label dict.")
    parser.add_argument('--test_path', type=str, default=None, help="The path of test set that you want to predict.")
    parser.add_argument('--save_path', type=str, required=True, default=None, help="The saving path of predict results.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--ext_max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization for extraction model.")
    parser.add_argument("--cls_max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization for classification model.")
    parser.add_argument("--use_tensorrt", action='store_true', help="Whether to use inference engin TensorRT.")
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "fp16", "int8"],help='The tensorrt precision.')
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "xpu"], help="Device selected for inference.")
    parser.add_argument('--cpu_threads', default=10, type=int, help='Number of threads to predict when using cpu.')
    parser.add_argument('--enable_mkldnn', default=False, type=eval, choices=[True, False], help='Enable to use mkldnn to speed up when using cpu.')
    args = parser.parse_args()
    # yapf: enbale

    predictor = Predictor(args)
    predictor.predict(args)
