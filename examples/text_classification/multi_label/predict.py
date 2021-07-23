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
import os
from functools import partial

import numpy as np
import pandas as pd
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset

from data import convert_example, create_dataloader, read_custom_data
from metric import F1Score

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=200, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--data_path", type=str, default="./data", help="The path of datasets to be loaded")
args = parser.parse_args()
# yapf: enable

def write_results(filename, results, results_dict):
    """write_results"""
    cols = ["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    data = pd.read_csv(filename)
    qids = [line[0] for line in data.values]
    results_dict["id"] = qids
    results = list(map(list, zip(*results)))
    for key in results_dict:
        if key != "id":
            for result in results:
                results_dict[key] = result
    df = pd.DataFrame(results_dict)
    df.to_csv("sample_test.csv", index=False)
    print("Test results saved")

def predict(model, data_loader, tokenizer, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """

    results = []
    model.eval()
    for step, batch in enumerate(data_loader, start=1):
        input_ids, token_type_ids = batch
        logits = model(input_ids, token_type_ids)
        probs = F.sigmoid(logits)
        print(probs)
        exit()
        preds = probs.tolist()
        results.extend(preds)
        if step % 100 == 0:
            print("step %d, %d samples processed" % (step, step * batch_size))
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    # Load train dataset.
    dataset_name = 'test.csv'
    test_ds = load_dataset(read_custom_data, filename=os.path.join(args.data_path, dataset_name), is_test=True, lazy=False)

    # Init the results template.
    results_dict = {"toxic": [], "severe_toxic": [], "obscene": [], "threat": [], "insult": [], "identity_hate": []}

    # If you wanna use bert/roberta/electra pretrained model,
    model = ppnlp.transformers.BertForMultiLabelTextClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(results_dict))

    # If you wanna use bert/roberta/electra pretrained model,
    tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): [data for data in fn(samples)]
    test_data_loader = create_dataloader(
        test_ds,
        mode='test',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = predict(model, test_data_loader, tokenizer, args.batch_size)
    filename = os.path.join(args.data_path, dataset_name)

    write_results(filename, results, results_dict)
