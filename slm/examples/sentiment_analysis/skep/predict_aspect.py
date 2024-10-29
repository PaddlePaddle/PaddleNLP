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
from functools import partial

import paddle
import paddle.nn.functional as F
from tqdm import tqdm

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    choices=["skep_ernie_1.0_large_ch", "skep_ernie_2.0_large_en"],
    default="skep_ernie_1.0_large_ch",
    help="Select which model to train, defaults to skep_ernie_1.0_large_ch.",
)
parser.add_argument("--ckpt_dir", type=str, default=None, help="The directory of saved model checkpoint.")
parser.add_argument(
    "--max_seq_len",
    default=400,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument("--batch_size", default=6, type=int, help="Batch size per GPU/CPU for prediction.")
parser.add_argument(
    "--device",
    choices=["cpu", "gpu", "xpu"],
    default="gpu",
    help="Select which device to train model, defaults to gpu.",
)
args = parser.parse_args()


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
    for batch in tqdm(data_loader):
        input_ids, token_type_ids = batch["input_ids"], batch["token_type_ids"]
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


def convert_example_to_feature(example, tokenizer, max_seq_len=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens.

    Args:
        example(obj:`dict`): Dict of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): The list of token_type_ids.
        label(obj:`int`, optional): The input label if not is_test.
    """
    encoded_inputs = tokenizer(text=example["text"], text_pair=example["text_pair"], max_seq_len=max_seq_len)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if is_test:
        return {"input_ids": input_ids, "token_type_ids": token_type_ids}
    else:
        label = example["label"]
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "labels": label}


def create_dataloader(dataset, mode="train", batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)


if __name__ == "__main__":

    test_ds = load_dataset("seabsa16", "phns", splits=["test"])
    label_map = {0: "negative", 1: "positive"}

    tokenizer = SkepTokenizer.from_pretrained(args.model_name)
    model = SkepForSequenceClassification.from_pretrained(args.ckpt_dir, num_labels=len(label_map))

    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, max_seq_len=args.max_seq_len, is_test=True)
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    test_data_loader = create_dataloader(
        test_ds, mode="test", batch_size=args.batch_size, batchify_fn=data_collator, trans_fn=trans_func
    )

    results = predict(model, test_data_loader, label_map)
    for idx, text in enumerate(test_ds.data):
        print("Data: {} \t Label: {}".format(text, results[idx]))
