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
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Dict, Pad
from datasets import load_dataset
from paddlenlp.transformers import SkepCrfForTokenClassification, SkepModel, SkepTokenizer

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def convert_example_to_feature(example,
                               tokenizer,
                               label_map,
                               max_seq_len=512,
                               is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence has the following format:
    ::
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence pair mask has the following format:
    ::

        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label dict that convert label to label_id.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`list[int]`, optional): The input label if not test data.
    """
    text = example['text_a']
    label = example['label']
    tokenized_input = tokenizer(list(text),
                                return_length=True,
                                is_split_into_words=True,
                                max_seq_len=max_seq_len)

    input_ids = np.array(tokenized_input['input_ids'], dtype="int64")
    token_type_ids = np.array(tokenized_input['token_type_ids'], dtype="int64")
    seq_len = tokenized_input['seq_len']

    if is_test:
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "seq_len": seq_len
        }
    else:
        # processing label
        start_idx = text.find(label)
        encoded_label = [label_map['O']] * len(text)
        if start_idx != -1:
            encoded_label[start_idx] = label_map["B"]
            for idx in range(start_idx + 1, start_idx + len(label)):
                encoded_label[idx] = label_map["I"]
        encoded_label = encoded_label[:(max_seq_len - 2)]
        encoded_label = np.array([label_map["O"]] + encoded_label +
                                 [label_map["O"]],
                                 dtype="int64")

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "seq_len": seq_len,
            "label": encoded_label
        }


@paddle.no_grad()
def predict(model, data_loader, id2label):
    """
    Given a prediction dataset, it gives the prediction results.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        id2label(obj:`dict`): The label id (key) to label str (value) map.
    """
    model.eval()
    results = []
    for input_ids, token_type_ids, seq_lens in data_loader:
        preds = model(input_ids, token_type_ids, seq_lens=seq_lens)
        tags = parse_predict_result(preds.numpy(), seq_lens.numpy(), id2label)
        results.extend(tags)
    return results


def parse_predict_result(predictions, seq_lens, id2label):
    """
    Parses the prediction results to the label tag.
    """
    pred_tag = []
    for idx, pred in enumerate(predictions):
        seq_len = seq_lens[idx]
        # drop the "[CLS]" and "[SEP]" token
        tag = [id2label[i] for i in pred[1:seq_len - 1]]
        pred_tag.append(tag)
    return pred_tag


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)


if __name__ == "__main__":
    paddle.set_device(args.device)

    test_ds, = load_dataset("cote", "dp", split=[
        'test',
    ])
    label_list = ["B", "I", "O"]
    # The COTE_DP dataset labels with "BIO" schema.
    label_map = {label: idx for idx, label in enumerate(label_list)}
    id2label = dict([(v, k) for k, v in label_map.items()])

    skep = SkepModel.from_pretrained('skep_ernie_1.0_large_ch')
    model = SkepCrfForTokenClassification(skep, num_classes=len(label_list))
    tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    trans_func = partial(convert_example_to_feature,
                         tokenizer=tokenizer,
                         label_map=label_map,
                         max_seq_len=args.max_seq_len,
                         is_test=True)

    batchify_fn = lambda samples, fn=Dict({
        "input_ids":
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input ids
        "token_type_ids":
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        "seq_len":
        Stack(dtype='int64')  # sequence lens
    }): fn(samples)

    test_data_loader = create_dataloader(test_ds,
                                         mode='test',
                                         batch_size=args.batch_size,
                                         batchify_fn=batchify_fn,
                                         trans_fn=trans_func)

    results = predict(model, test_data_loader, id2label)
    for idx, example in enumerate(test_ds):
        print(len(example['text_a']), len(results[idx]))
        print('Data: {} \t Label: {}'.format(example, results[idx]))
