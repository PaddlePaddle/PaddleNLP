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
import paddle
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import ErnieCtmNptagModel, ErnieCtmTokenizer

from data import convert_example, create_dataloader, read_custom_data
from utils import construct_dict_map, decode, search, find_topk

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default="./output/model_100/model_state.pdparams", required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--data_dir", type=str, default="./data", help="The input data dir, should contain name_category_map.json.")
parser.add_argument("--max_seq_len", type=int, default=64, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def do_predict(data,
               model,
               tokenizer,
               batch_size=1,
               max_cls_len=5,
               summary_num=2):
    examples = []
    for text in data:
        example = {"text": text}
        input_ids, token_type_ids, label_indices = convert_example(
            example, tokenizer, max_seq_len=args.max_seq_len, is_test=True)
        examples.append((input_ids, token_type_ids, label_indices))

    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # token_type_ids
        Stack(dtype='int64'),  # label_indices
    ): fn(samples)

    name_dict, bk_tree, id_vocabs, vocab_ids = construct_dict_map(
        tokenizer, os.path.join(args.data_dir, "name_category_map.json"))

    all_scores_can = []
    all_preds_can = []
    pred_ids = []

    model.eval()
    for batch in batches:
        input_ids, token_type_ids, label_indices = batchify_fn(batch)

        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids).numpy()
        for i, l in zip(label_indices, logits):
            score = l[i[0]:i[-1] + 1, vocab_ids]
            # Find topk candidates of scores and predicted indices.
            score_can, pred_id_can = find_topk(score, k=4, axis=-1)

            all_scores_can.extend([score_can.tolist()])
            all_preds_can.extend([pred_id_can.tolist()])
            pred_ids.extend([pred_id_can[:, 0].tolist()])

    results = []
    for i, d in enumerate(data):
        label = decode(pred_ids[i], id_vocabs)

        result = {
            'text': d,
            'label': label,
        }

        if label not in name_dict:
            scores_can = all_scores_can[i]
            pred_ids_can = all_preds_can[i]
            labels_can = search(scores_can, pred_ids_can, 0, [], 0)
            labels_can.sort(key=lambda d: -d[1])
            for labels in labels_can:
                cls_label_can = decode(labels[0], id_vocabs)
                if cls_label_can in name_dict:
                    result['label'] = cls_label_can
                    break
                else:
                    labels_can = bk_tree.search_similar_word(label)
                    if len(labels_can) != 0:
                        result['label'] = labels_can[0][0]

        if result['label'] in name_dict:
            result['category'] = name_dict[result['label']]
        results.append(result)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    data = [
        '刘德华',
        '快乐薯片',
        '自适应共振理论映射',
    ]

    model = ErnieCtmNptagModel.from_pretrained("nptag")
    tokenizer = ErnieCtmTokenizer.from_pretrained("nptag")

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = do_predict(data, model, tokenizer, batch_size=args.batch_size)
    print(results)
