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
from functools import partial
from collections import defaultdict
import paddle
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import SkepTokenizer, SkepForTokenClassification, SkepForSequenceClassification
from utils import decoding, load_dict, read_test_file
from extraction.data import convert_example_to_feature as convert_example_to_feature_ext
from classification.data import convert_example_to_feature as convert_example_to_feature_cls


def concate_aspect_and_opinion(text, aspect, opinions):
    aspect_text = ""
    for opinion in opinions:
        if text.find(aspect) <= text.find(opinion):
            aspect_text += aspect + opinion + "，"
        else:
            aspect_text += opinion + aspect + "，"
    aspect_text = aspect_text[:-1]

    return aspect_text


def predict_ext(args):
    # load dict
    model_name = "skep_ernie_1.0_large_ch"
    ext_label2id, ext_id2label = load_dict(args.ext_label_path)

    tokenizer = SkepTokenizer.from_pretrained(model_name)
    ori_test_ds = load_dataset(read_test_file,
                               data_path=args.test_path,
                               lazy=False)
    trans_func = partial(convert_example_to_feature_ext,
                         tokenizer=tokenizer,
                         label2id=ext_label2id,
                         max_seq_len=args.ext_max_seq_len,
                         is_test=True)
    test_ds = copy.copy(ori_test_ds).map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
        Stack(dtype="int64"),
    ): fn(samples)

    test_batch_sampler = paddle.io.BatchSampler(test_ds,
                                                batch_size=args.batch_size,
                                                shuffle=False)
    test_loader = paddle.io.DataLoader(test_ds,
                                       batch_sampler=test_batch_sampler,
                                       collate_fn=batchify_fn)
    print("test data loaded.")

    # load ext model
    ext_state_dict = paddle.load(args.ext_model_path)
    ext_model = SkepForTokenClassification.from_pretrained(
        model_name, num_classes=len(ext_label2id))
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
            aps = decoding(text[:args.ext_max_seq_len - 2], tag_seq)
            for aid, ap in enumerate(aps):
                aspect, opinions = ap[0], list(set(ap[1:]))
                aspect_text = concate_aspect_and_opinion(text, aspect, opinions)
                results.append({
                    "id": str(idx) + "_" + str(aid),
                    "aspect": aspect,
                    "opinions": opinions,
                    "text": text,
                    "aspect_text": aspect_text
                })

    return results


def predict_cls(args, ext_results):
    # load dict
    model_name = "skep_ernie_1.0_large_ch"
    cls_label2id, cls_id2label = load_dict(args.cls_label_path)

    tokenizer = SkepTokenizer.from_pretrained(model_name)
    test_ds = MapDataset(ext_results)
    trans_func = partial(convert_example_to_feature_cls,
                         tokenizer=tokenizer,
                         label2id=cls_label2id,
                         max_seq_len=args.cls_max_seq_len,
                         is_test=True)
    test_ds = test_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id), Stack(dtype="int64")
    ): fn(samples)

    # set shuffle is False
    test_batch_sampler = paddle.io.BatchSampler(test_ds,
                                                batch_size=args.batch_size,
                                                shuffle=False)
    test_loader = paddle.io.DataLoader(test_ds,
                                       batch_sampler=test_batch_sampler,
                                       collate_fn=batchify_fn)
    print("test data loaded.")

    # load cls model
    cls_state_dict = paddle.load(args.cls_model_path)
    cls_model = SkepForSequenceClassification.from_pretrained(
        model_name, num_classes=len(cls_label2id))
    cls_model.load_dict(cls_state_dict)
    print("classification model loaded.")

    cls_model.eval()

    results = []
    for bid, batch_data in enumerate(test_loader):
        input_ids, token_type_ids, seq_lens = batch_data
        logits = cls_model(input_ids, token_type_ids=token_type_ids)

        predictions = logits.argmax(axis=1).numpy().tolist()
        results.extend(predictions)

    results = [cls_id2label[pred_id] for pred_id in results]
    return results


def post_process(ext_results, cls_results):
    assert len(ext_results) == len(cls_results)

    collect_dict = defaultdict(list)
    for ext_result, cls_result in zip(ext_results, cls_results):
        ext_result["sentiment_polarity"] = cls_result
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
    parser.add_argument("--ext_max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization for extraction model.")
    parser.add_argument("--cls_max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization for classification model.")
    args = parser.parse_args()
    # yapf: enbale

    # predict with ext model
    ext_results = predict_ext(args)
    print("predicting with extraction model done!")

    # predict with cls model
    cls_results = predict_cls(args, ext_results)
    print("predicting with classification model done!")

    # post_process prediction results
    post_process(ext_results, cls_results)
    print(f"sentiment analysis results has been saved to path: {args.save_path}")
