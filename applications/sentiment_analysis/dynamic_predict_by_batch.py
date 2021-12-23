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

import os
import copy
import json
import argparse
from collections import defaultdict
from functools import partial
import paddle
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepModel, SkepTokenizer
from extraction.data import convert_example_to_feature as convert_example_to_feature_ext
from extraction.model import SkepForTokenClassification
from classification.model import SkepForSequenceClassification
from classification.data import load_dict
from classification.data import convert_example_to_feature as convert_example_to_feature_cls

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


def is_aspect_first(text, aspect, opinion):
    return text.find(aspect) <= text.find(opinion)


def concate_aspect_and_opinion(text, aspect, opinions):
    aspect_text = ""
    for opinion in opinions:
        if is_aspect_first(text, aspect, opinion):
            aspect_text += aspect + opinion + "，"
        else:
            aspect_text += opinion + aspect + "，"
    aspect_text = aspect_text[:-1]

    return aspect_text


def read_ext(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().replace(" ", "")
            yield {"text": line}


def read_cls(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            example = json.loads(line)
            yield example


def predict_ext(ext_model_path, ext_label_path, test_path):
    # load dict
    model_name = "skep_ernie_1.0_large_ch"
    ext_label2id, ext_id2label = load_dict(args.ext_label_path)

    tokenizer = SkepTokenizer.from_pretrained(model_name)
    ori_test_ds = load_dataset(read_ext, data_path=test_path, lazy=False)
    trans_func = partial(
        convert_example_to_feature_ext,
        tokenizer=tokenizer,
        label2id=ext_label2id,
        max_seq_len=args.max_seq_len,
        is_test=True)
    test_ds = copy.copy(ori_test_ds).map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="int64"), ): fn(samples)

    test_batch_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = paddle.io.DataLoader(
        test_ds, batch_sampler=test_batch_sampler, collate_fn=batchify_fn)
    print("test data loaded.")

    # load ext model
    ext_state_dict = paddle.load(args.ext_model_path)
    ext_skep = SkepModel.from_pretrained(model_name)
    ext_model = SkepForTokenClassification(
        ext_skep, num_classes=len(ext_label2id))
    ext_model.load_dict(ext_state_dict)
    print("extraction model loaded.")

    ext_model.eval()
    results = []
    for bid, batch_data in enumerate(test_loader):
        input_ids, token_type_ids, seq_lens = batch_data
        logits = ext_model(input_ids, token_type_ids=token_type_ids)

        predictions = logits.argmax(axis=2).numpy()
        for eid, (seq_len, prediction) in enumerate(zip(seq_lens, predictions)):
            idx = bid * args.batch_size + eid
            tag_seq = [ext_id2label[idx] for idx in prediction[:seq_len][1:-1]]
            text = ori_test_ds[idx]["text"]
            aps = decoding(text, tag_seq)
            for aid, ap in enumerate(aps):
                aspect, opinions = ap[0], list(set(ap[1:]))
                aspect_text = concate_aspect_and_opinion(text, aspect, opinions)
                results.append({
                    "id": str(idx) + "_" + str(aid),
                    "aspect": aspect,
                    "opinions": opinions,
                    "text": text,
                    "target_text": aspect_text
                })

    with open(args.save_ext_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def predict_cls(cls_model_path, cls_label_path, test_path):
    # load dict
    model_name = "skep_ernie_1.0_large_ch"
    cls_label2id, cls_id2label = load_dict(args.cls_label_path)

    tokenizer = SkepTokenizer.from_pretrained(model_name)
    test_ds = load_dataset(read_cls, data_path=test_path, lazy=False)
    # examples = copy.copy(test_ds)
    trans_func = partial(
        convert_example_to_feature_cls,
        tokenizer=tokenizer,
        label2id=cls_label2id,
        max_seq_len=args.max_seq_len,
        is_test=True)
    test_ds = test_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="int64"), ): fn(samples)

    # set shuffle is False
    test_batch_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = paddle.io.DataLoader(
        test_ds, batch_sampler=test_batch_sampler, collate_fn=batchify_fn)
    print("test data loaded.")

    # load cls model
    cls_state_dict = paddle.load(args.cls_model_path)
    cls_skep = SkepModel.from_pretrained(model_name)
    cls_model = SkepForSequenceClassification(
        cls_skep, num_classes=len(cls_label2id))
    cls_model.load_dict(cls_state_dict)
    print("classification model loaded.")

    cls_model.eval()

    results = []
    for bid, batch_data in enumerate(test_loader):
        input_ids, token_type_ids, seq_lens = batch_data
        logits = cls_model(input_ids, token_type_ids=token_type_ids)

        predictions = logits.argmax(axis=1).numpy().tolist()
        results.extend(predictions)

    with open(args.save_cls_path, "w", encoding="utf-8") as f:
        for line_id, pred_id in enumerate(results):
            f.write(
                json.dumps(
                    {
                        "line_id": line_id,
                        "sentiment_polarity": cls_id2label[pred_id]
                    },
                    ensure_ascii=False) + "\n")


def post_process():
    ext_results, cls_results = [], []

    with open(args.save_ext_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            ext_results.append(json.loads(line))

    with open(args.save_cls_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            cls_results.append(json.loads(line))

    assert len(ext_results) == len(cls_results)

    collect_dict = defaultdict(list)
    for ext_result, cls_result in zip(ext_results, cls_results):
        ext_result["sentiment_polarity"] = cls_result["sentiment_polarity"]
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
                "aspect": single_ap["aspect"],
                "opinions": single_ap["opinions"],
                "sentiment_polarity": single_ap["sentiment_polarity"]
            })
        sentiment_result["ap_list"] = ap_list
        sentiment_results.append(sentiment_result)

    with open(args.save_path, "w", encoding="utf-8") as f:
        for sentiment_result in sentiment_results:
            f.write(json.dumps(sentiment_result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--ext_model_path", type=str, default=None, help="The path of extraction model path that you want to load.")
    parser.add_argument("--cls_model_path", type=str, default=None, help="The path of classification model path that you want to load.")
    parser.add_argument("--ext_label_path", type=str, default=None, help="The path of extraction label dict.")
    parser.add_argument("--cls_label_path", type=str, default=None, help="The path of classification label dict.")
    parser.add_argument('--test_path', type=str, default=None, help="The path of test set that you want to predict.")
    parser.add_argument('--save_path', type=str, required=True, default=None, help="The saving path of predict results.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    args = parser.parse_args()
    # yapf: enbale

    # process save_path
    ppath, whole_file_name = os.path.split(args.save_path)
    file_name, suffix = whole_file_name.rsplit(".", maxsplit=1)
    args.save_ext_path = os.path.join(ppath, file_name+"_ext."+suffix)
    args.save_cls_path = os.path.join(ppath, file_name+"_cls."+suffix)

    # predict with ext model
    predict_ext(args.ext_model_path, args.ext_label_path, args.test_path)
    print(f"ext prediction results has been saved to path: {args.save_ext_path}")

    # predict with cls model
    predict_cls(args.cls_model_path, args.cls_label_path, args.save_ext_path)
    print(f"cls prediction results has been saved to path: {args.save_cls_path}")

    # post_process prediction results 
    post_process()
    print(f"sentiment analysis results has been saved to path: {args.save_path}")
