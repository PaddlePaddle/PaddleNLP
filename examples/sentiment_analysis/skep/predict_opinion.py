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
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepCrfForTokenClassification, SkepModel, SkepTokenizer

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
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
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask. 
    """
    tokens = example["tokens"]
    encoded_inputs = tokenizer(tokens,
                               return_length=True,
                               is_split_into_words=True,
                               max_seq_len=max_seq_length)
    input_ids = np.array(encoded_inputs["input_ids"], dtype="int64")
    token_type_ids = np.array(encoded_inputs["token_type_ids"], dtype="int64")
    seq_len = np.array(encoded_inputs["seq_len"], dtype="int64")

    return input_ids, token_type_ids, seq_len


@paddle.no_grad()
def predict(model, data_loader, label_map):
    """
    Given a prediction dataset, it gives the prediction results.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
    """
    model.eval()
    results = []
    for input_ids, token_type_ids, seq_lens in data_loader:
        preds = model(input_ids, token_type_ids, seq_lens=seq_lens)
        tags = parse_predict_result(preds.numpy(), seq_lens.numpy(), label_map)
        results.extend(tags)
    return results


def parse_predict_result(predictions, seq_lens, label_map):
    """
    Parses the prediction results to the label tag.
    """
    pred_tag = []
    for idx, pred in enumerate(predictions):
        seq_len = seq_lens[idx]
        # drop the "[CLS]" and "[SEP]" token
        tag = [label_map[i] for i in pred[1:seq_len - 1]]
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

    test_ds = load_dataset("cote", "dp", splits=['test'])
    # The COTE_DP dataset labels with "BIO" schema.
    label_map = {0: "B", 1: "I", 2: "O"}
    # `no_entity_label` represents that the token isn't an entity.
    no_entity_label_idx = 2

    skep = SkepModel.from_pretrained('skep_ernie_1.0_large_ch')
    model = SkepCrfForTokenClassification(skep,
                                          num_classes=len(test_ds.label_list))
    tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]
            ),  # token type ids
        Stack(dtype='int64'),  # sequence lens
    ): [data for data in fn(samples)]

    test_data_loader = create_dataloader(test_ds,
                                         mode='test',
                                         batch_size=args.batch_size,
                                         batchify_fn=batchify_fn,
                                         trans_fn=trans_func)

    results = predict(model, test_data_loader, label_map)
    for idx, example in enumerate(test_ds.data):
        print(len(example['tokens']), len(results[idx]))
        print('Data: {} \t Label: {}'.format(example, results[idx]))
