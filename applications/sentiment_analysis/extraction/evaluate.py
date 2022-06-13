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

import argparse
from tqdm import tqdm
from functools import partial
import paddle
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepForTokenClassification, SkepTokenizer
from data import read, load_dict, convert_example_to_feature


def evaluate(model, data_loader, metric):

    model.eval()
    metric.reset()
    for idx, batch_data in tqdm(enumerate(data_loader)):
        input_ids, token_type_ids, seq_lens, labels = batch_data
        logits = model(input_ids, token_type_ids=token_type_ids)

        # count metric
        predictions = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            seq_lens, predictions, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(),
                      num_correct_chunks.numpy())

    precision, recall, f1 = metric.accumulate()

    return precision, recall, f1


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument('--test_path', type=str, default=None, help="The path of test set.")
    parser.add_argument("--label_path", type=str, default=None, help="The path of label dict.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    args = parser.parse_args()
    # yapf: enbale

    # load dev data
    model_name = "skep_ernie_1.0_large_ch"
    label2id, id2label = load_dict(args.label_path)
    test_ds = load_dataset(read, data_path=args.test_path, lazy=False)

    tokenizer = SkepTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, label2id=label2id, max_seq_len=args.max_seq_len)
    test_ds = test_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
        Stack(dtype="int64"),
        Pad(axis=0, pad_val=-1, dtype="int64")
    ): fn(samples)

    test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = paddle.io.DataLoader(test_ds, batch_sampler=test_batch_sampler, collate_fn=batchify_fn)

    # load model
    loaded_state_dict = paddle.load(args.model_path)
    model = SkepForTokenClassification.from_pretrained(model_name, num_classes=len(label2id))
    model.load_dict(loaded_state_dict)

    metric = ChunkEvaluator(label2id.keys())

    # evalute on dev data
    precision, recall, f1  = evaluate(model, test_loader,  metric)
    print(f'evalution result: precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')
