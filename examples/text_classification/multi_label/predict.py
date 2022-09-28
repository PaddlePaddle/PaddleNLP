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

import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.data import Tuple, Pad
from paddlenlp.datasets import load_dataset

from data import convert_example, create_dataloader, read_custom_data, write_test_results
from model import MultiLabelClassifier

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, default='./checkpoint/model_800/model_state.pdparams', help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--data_path", type=str, default="./data", help="The path of datasets to be loaded")
args = parser.parse_args()
# yapf: enable


def predict(model, data_loader, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
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
        probs = probs.tolist()
        results.extend(probs)
        if step % 100 == 0:
            print("step %d, %d samples processed" % (step, step * batch_size))
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    # Load train dataset.
    file_name = 'test.csv'
    test_ds = load_dataset(read_custom_data,
                           filename=os.path.join(args.data_path, file_name),
                           is_test=True,
                           lazy=False)

    # The dataset labels
    label_info = [
        'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    ]

    # Load pretrained model
    pretrained_model = AutoModel.from_pretrained("bert-base-uncased")

    # Load bert tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    model = MultiLabelClassifier(pretrained_model, num_labels=len(label_info))

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         is_test=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): [data for data in fn(samples)]
    test_data_loader = create_dataloader(test_ds,
                                         mode='test',
                                         batch_size=args.batch_size,
                                         batchify_fn=batchify_fn,
                                         trans_fn=trans_func)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = predict(model, test_data_loader, args.batch_size)
    filename = os.path.join(args.data_path, file_name)

    # Write test result into csv file
    write_test_results(filename, results, label_info)
