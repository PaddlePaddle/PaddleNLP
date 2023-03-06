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
import copy
import json
import os
import re
from collections import defaultdict
from functools import partial

import paddle
from datasets import Dataset, load_dataset
from paddle import inference
from seqeval.metrics.sequence_labeling import get_entities

from paddlenlp.data import DataCollatorForTokenClassification, DataCollatorWithPadding
from paddlenlp.transformers import SkepTokenizer


def load_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word2id = dict(zip(words, range(len(words))))
        id2word = dict((v, k) for k, v in word2id.items())

        return word2id, id2word


def read_test_file(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().replace(" ", "")
            yield {"text": line}


def decoding(text, tag_seq):
    assert len(text) == len(tag_seq), f"text len: {len(text)}, tag_seq len: {len(tag_seq)}"

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
        for ent in ent_list:
            ent_name, start, end = ent
            if ent_name == "Aspect":
                aspect = sub_tag_seq[start : end + 1]
                sub_aps.append([aspect])
                if len(sub_no_a_words) > 0:
                    sub_aps[-1].extend(sub_no_a_words)
                    sub_no_a_words.clear()
            else:
                ent_name == "Opinion"
                opinion = sub_tag_seq[start : end + 1]
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


def convert_example_to_feature_ext(example, tokenizer, label2id, max_seq_len=512, is_test=False):
    example = example["text"].rstrip().split("\t")
    text = list(example[0])
    if not is_test:
        label = example[1].split(" ")
        assert len(text) == len(label)
        new_text = []
        new_label = []
        for text_ch, label_ch in zip(text, label):
            if text_ch.strip():
                new_text.append(text_ch)
                new_label.append(label_ch)
        new_label = (
            [label2id["O"]] + [label2id[label_term] for label_term in new_label][: (max_seq_len - 2)] + [label2id["O"]]
        )
        encoded_inputs = tokenizer(new_text, is_split_into_words="token", max_seq_len=max_seq_len, return_length=True)
        encoded_inputs["labels"] = new_label
        assert len(encoded_inputs["input_ids"]) == len(
            new_label
        ), f"input_ids: {len(encoded_inputs['input_ids'])}, label: {len(new_label)}"
    else:
        new_text = [text_ch for text_ch in text if text_ch.strip()]
        encoded_inputs = tokenizer(new_text, is_split_into_words="token", max_seq_len=max_seq_len, return_length=True)

    return encoded_inputs


def convert_example_to_feature_cls(example, tokenizer, label2id, max_seq_len=512, is_test=False):
    example = example["text"].rstrip().split("\t")
    if not is_test:
        label = int(example[0])
        aspect_text = example[1]
        text = example[2]
        encoded_inputs = tokenizer(aspect_text, text_pair=text, max_seq_len=max_seq_len, return_length=True)
        encoded_inputs["label"] = label
    else:
        aspect_text = example[0]
        text = example[1]
        encoded_inputs = tokenizer(aspect_text, text_pair=text, max_seq_len=max_seq_len, return_length=True)

    return encoded_inputs


def remove_blanks(example):
    example["text"] = re.sub(" +", "", example["text"])
    return example


class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.ext_predictor, self.ext_input_handles, self.ext_output_hanle = self.create_predictor(args.ext_model_path)
        print(f"ext_model_path: {args.ext_model_path}, {self.ext_predictor}")
        self.cls_predictor, self.cls_input_handles, self.cls_output_hanle = self.create_predictor(args.cls_model_path)
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
            # such as initialize the gpu memory, enable tensorrt
            config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32,
                "int8": inference.PrecisionType.Int8,
            }
            precision_mode = precision_map[args.precision]

            if args.use_tensorrt:
                config.enable_tensorrt_engine(
                    max_batch_size=self.args.batch_size, min_subgraph_size=30, precision_mode=precision_mode
                )
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
        input_handles = [predictor.get_input_handle(name) for name in predictor.get_input_names()]
        output_handle = predictor.get_output_handle(predictor.get_output_names()[0])

        return predictor, input_handles, output_handle

    def predict_ext(self, args):
        datasets = load_dataset("text", data_files={"test": args.test_path})
        datasets["test"] = datasets["test"].map(remove_blanks)
        trans_func = partial(
            convert_example_to_feature_ext,
            tokenizer=self.tokenizer,
            label2id=self.ext_label2id,
            max_seq_len=args.ext_max_seq_len,
            is_test=True,
        )
        test_ds = copy.copy(datasets["test"]).map(trans_func, batched=False, remove_columns=["text"])
        data_collator = DataCollatorForTokenClassification(self.tokenizer, label_pad_token_id=self.ext_label2id["O"])
        test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = paddle.io.DataLoader(test_ds, batch_sampler=test_batch_sampler, collate_fn=data_collator)

        results = []
        for bid, batch_data in enumerate(test_loader):
            input_ids, token_type_ids, seq_lens = (
                batch_data["input_ids"],
                batch_data["token_type_ids"],
                batch_data["seq_len"],
            )
            self.ext_input_handles[0].copy_from_cpu(input_ids.numpy())
            self.ext_input_handles[1].copy_from_cpu(token_type_ids.numpy())
            self.ext_predictor.run()
            logits = self.ext_output_hanle.copy_to_cpu()

            predictions = logits.argmax(axis=2)
            for eid, (seq_len, prediction) in enumerate(zip(seq_lens, predictions)):
                idx = bid * args.batch_size + eid
                tag_seq = [self.ext_id2label[idx] for idx in prediction[:seq_len][1:-1]]
                text = datasets["test"][idx]["text"]
                aps = decoding(text[: args.ext_max_seq_len - 2], tag_seq)
                for aid, ap in enumerate(aps):
                    aspect, opinions = ap[0], list(set(ap[1:]))
                    aspect_text = self._concate_aspect_and_opinion(text, aspect, opinions)
                    results.append(
                        {
                            "id": str(idx) + "_" + str(aid),
                            "aspect": aspect,
                            "opinions": opinions,
                            "text": text,
                            "aspect_text": aspect_text,
                        }
                    )

        return results

    def predict_cls(self, args, ext_results):
        text_list = []
        for result in ext_results:
            example = result["aspect_text"] + "\t" + result["text"]
            text_list.append(example)
        ext_results = {"text": text_list}

        dataset = Dataset.from_dict(ext_results)
        trans_func = partial(
            convert_example_to_feature_cls,
            tokenizer=self.tokenizer,
            label2id=self.cls_label2id,
            max_seq_len=args.cls_max_seq_len,
            is_test=True,
        )

        test_ds = dataset.map(trans_func, batched=False, remove_columns=["text"])
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = paddle.io.DataLoader(test_ds, batch_sampler=test_batch_sampler, collate_fn=data_collator)

        results = []
        for batch_data in test_loader:
            input_ids, token_type_ids = batch_data["input_ids"], batch_data["token_type_ids"]
            self.cls_input_handles[0].copy_from_cpu(input_ids.numpy())
            self.cls_input_handles[1].copy_from_cpu(token_type_ids.numpy())
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
                ap_list.append(
                    {
                        "aspect": single_ap["aspect"],
                        "opinions": single_ap["opinions"],
                        "sentiment_polarity": single_ap["sentiment_polarity"],
                    }
                )
            sentiment_result["ap_list"] = ap_list
            sentiment_results.append(sentiment_result)

        with open(args.save_path, "w", encoding="utf-8") as f:
            for sentiment_result in sentiment_results:
                f.write(json.dumps(sentiment_result, ensure_ascii=False) + "\n")
        print(f"sentiment analysis results has been saved to path: {args.save_path}")

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
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "fp16", "int8"], help='The tensorrt precision.')
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "xpu"], help="Device selected for inference.")
    parser.add_argument('--cpu_threads', default=10, type=int, help='Number of threads to predict when using cpu.')
    parser.add_argument('--enable_mkldnn', default=False, type=eval, choices=[True, False], help='Enable to use mkldnn to speed up when using cpu.')
    args = parser.parse_args()
    # yapf: enbale

    predictor = Predictor(args)
    predictor.predict(args)
