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
from functools import partial
import numpy as np
import time
import paddle
from paddle import inference
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import SkepModel, SkepTokenizer
from paddlenlp.metrics import AccuracyAndF1
from seqeval.metrics.sequence_labeling import get_entities


def decoding(text, tag_seq):
    assert len(text) == len(
        tag_seq), f"text len: {len(text)}, tag_seq len: {len(tag_seq)}"

    puncs = list(",.?;!，。？；！")
    splits = [idx for idx in range(len(text)) if text[idx] in puncs]

    prev = 0
    sub_texts, sub_tag_seqs = [], []
    for i, split in enumerate(splits):
        sub_tag_seqs.append(tag_seq[prev:split])
        sub_texts.append(text[prev:split])
        prev = split
    sub_tag_seqs.append(tag_seq[prev:])
    sub_texts.append((text[prev:]))

    ents_list = []
    for sub_text, sub_tag_seq in zip(sub_texts, sub_tag_seqs):
        ents = get_entities(sub_tag_seq, suffix=False)
        ents_list.append((sub_text, ents))

    aps = []
    no_a_words = []
    for sub_tag_seq, ent_list in ents_list:
        sub_aps = []
        sub_no_a_words = []
        # print(ent_list)
        for ent in ent_list:
            ent_name, start, end = ent
            if ent_name == "Aspect":
                aspect = sub_tag_seq[start:end + 1]
                sub_aps.append([aspect])
                if len(sub_no_a_words) > 0:
                    sub_aps[-1].extend(sub_no_a_words)
                    sub_no_a_words.clear()
            else:
                ent_name == "Opinion"
                opinion = sub_tag_seq[start:end + 1]
                if len(sub_aps) > 0:
                    sub_aps[-1].append(opinion)
                else:
                    sub_no_a_words.append(opinion)

        if sub_aps:
            aps.extend(sub_aps)
            if len(no_a_words) > 0:
                aps[-1].extend(no_a_words)
                no_a_words.clear()
        elif sub_no_a_words:
            if len(aps) > 0:
                aps[-1].extend(sub_no_a_words)
            else:
                no_a_words.extend(sub_no_a_words)

    if no_a_words:
        no_a_words.insert(0, "None")
        aps.append(no_a_words)

    return aps


class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.ext_predictor, self.ext_input_handles, self.ext_output_hanle = self.create_predictor(
            args.ext_model_path)
        print(f"ext_model_path: {args.ext_model_path}, {self.ext_predictor}")
        self.cls_predictor, self.cls_input_handles, self.cls_output_hanle = self.create_predictor(
            args.cls_model_path)
        self.ext_label2id, self.ext_id2label = self._load_dict(
            args.ext_label_path)
        self.cls_label2id, self.cls_id2label = self._load_dict(
            args.cls_label_path)
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
        output_handle = predictor.get_output_handle(predictor.get_output_names()
                                                    [0])

        return predictor, input_handles, output_handle

    def predict(self, text):
        # extract aspect and opinion words
        ext_input_ids, ext_token_type_ids = self.convert_example_to_feature(
            text, max_seq_len=args.max_seq_len)
        self.ext_input_handles[0].copy_from_cpu(ext_input_ids)
        self.ext_input_handles[1].copy_from_cpu(ext_token_type_ids)
        self.ext_predictor.run()
        ext_logits = self.ext_output_hanle.copy_to_cpu()

        # extract aspect and opinion words
        predictions = ext_logits.argmax(axis=2)[0]
        tag_seq = [self.ext_id2label[idx] for idx in predictions][1:-1]
        aps = decoding(text, tag_seq)

        # predict sentiment for aspect with cls_model
        results = []
        for ap in aps:
            aspect = ap[0]
            opinion_words = list(set(ap[1:]))
            aspect_text = self._concate_aspect_and_opinion(text, aspect,
                                                           opinion_words)
            cls_input_ids, cls_token_type_ids = self.convert_example_to_feature(
                text=aspect_text, text_pair=text, max_seq_len=args.max_seq_len)

            self.cls_input_handles[0].copy_from_cpu(cls_input_ids)
            self.cls_input_handles[1].copy_from_cpu(cls_token_type_ids)
            self.cls_predictor.run()
            cls_logits = self.cls_output_hanle.copy_to_cpu()

            pred_id = cls_logits.argmax(axis=1)[0]
            result = {
                "aspect": aspect,
                "opinions": opinion_words,
                "sentiment": self.cls_id2label[pred_id]
            }
            results.append(result)

        self._format_print(results)

    def convert_example_to_feature(self, text, text_pair=None, max_seq_len=256):
        if text_pair is None:
            encoded_inputs = self.tokenizer(
                list(text), is_split_into_words=True, max_seq_len=max_seq_len)
        else:
            encoded_inputs = self.tokenizer(
                text, text_pair=text_pair, max_seq_len=max_seq_len)

        input_ids = np.array([encoded_inputs["input_ids"]])
        token_type_ids = np.array([encoded_inputs["token_type_ids"]])

        return input_ids, token_type_ids

    def _load_dict(self, dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            words = [word.strip() for word in f.readlines()]
            word2id = dict(zip(words, range(len(words))))
            id2word = dict((v, k) for k, v in word2id.items())

            return word2id, id2word

    def _concate_aspect_and_opinion(self, text, aspect, opinion_words):
        aspect_text = ""
        for opinion_word in opinion_words:
            if self._is_aspect_first(text, aspect, opinion_word):
                aspect_text += aspect + opinion_word + "，"
            else:
                aspect_text += opinion_word + aspect + "，"
        aspect_text = aspect_text[:-1]

        return aspect_text

    def _is_aspect_first(self, text, aspect, opinion_word):
        return text.find(aspect) <= text.find(opinion_word)

    def _format_print(self, results):
        for result in results:
            aspect, opinions, sentiment = result["aspect"], result[
                "opinions"], result["sentiment"]
            print(
                f"aspect: {aspect}, opinions: {opinions}, sentiment: {sentiment}"
            )
        print()


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", default='skep_ernie_1.0_large_ch', type=str, help="Base model name, SKEP used by default", )
    parser.add_argument("--ext_model_path", type=str, default=None, help="The path of extraction model path that you want to load.")
    parser.add_argument("--cls_model_path", type=str, default=None, help="The path of classification model path that you want to load.")
    parser.add_argument("--ext_label_path", type=str, default=None, help="The path of extraction label dict.")
    parser.add_argument("--cls_label_path", type=str, default=None, help="The path of classification label dict.")
    parser.add_argument("--max_seq_len", default=256, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--use_tensorrt", action='store_true', help="Whether to use inference engin TensorRT.")
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "fp16", "int8"],help='The tensorrt precision.')
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "xpu"], help="Device selected for inference.")
    parser.add_argument('--cpu_threads', default=10, type=int, help='Number of threads to predict when using cpu.')
    parser.add_argument('--enable_mkldnn', default=False, type=eval, choices=[True, False], help='Enable to use mkldnn to speed up when using cpu.')
    args = parser.parse_args()
    # yapf: enbale

    predictor = Predictor(args)

    # start to predict with ext_ and cls_ model
    input_text = "蛋糕味道不错，环境也很好"
    print("default input_text:", input_text)
    predictor.predict(input_text.strip().replace(" ", ""))

    while True:
        input_text = input("input text: \n")
        if not input_text:
            continue
        if input_text == "quit":
            break
        predictor.predict(input_text.strip().replace(" ", ""))
