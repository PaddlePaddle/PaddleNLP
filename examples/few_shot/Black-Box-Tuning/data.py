# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import label2text

from paddlenlp.datasets import MapDataset


def convert_example(example, n_prompt_tokens, tokenizer, max_seq_length):
    if n_prompt_tokens > 0:  # use randomly selected words as initial prompt
        offset = 1000
        prompt = tokenizer.decode(list(range(offset, offset + n_prompt_tokens)))
        example["input_text"] = "%s . %s . It was %s ." % (prompt, example["text"], tokenizer.mask_token)
        example["target_text"] = label2text[example["labels"]]
    else:
        example["input_text"] = "%s . It was %s ." % (example["text"], tokenizer.mask_token)
        example["target_text"] = label2text[example["labels"]]

    input_encodings = tokenizer.encode(example["input_text"], return_attention_mask=True, max_seq_len=max_seq_length)
    target_encodings = tokenizer.encode(example["target_text"], add_special_tokens=False, max_seq_len=max_seq_length)
    mask_pos = input_encodings["input_ids"].index(tokenizer.mask_token_id)
    # encodings = {
    #     'input_ids': input_encodings['input_ids'],
    #     'attention_mask': input_encodings['attention_mask'],
    #     'mask_pos': mask_pos,
    #     'labels': target_encodings['input_ids'],
    # }

    return input_encodings["input_ids"], input_encodings["attention_mask"], mask_pos, target_encodings["input_ids"][0]


def load_data(train_path=None, dev_path=None, dev_size=0.2, test_path=None):

    """
    加载数据集，数据集是csv/xlsx格式，列名是text，label
    test_size：测试集比例
    random_state：随机种子，默认是0，每次拆分结果都不一样
    """

    if test_path:
        test = pd.read_csv(test_path)
        test.columns = ["text", "label"]
        return process_data(test)

    train = pd.read_csv(train_path, sep="\t")
    train.columns = ["text", "label"]
    if dev_path:
        dev = pd.read_csv(dev_path)
        dev.columns = ["text", "label"]
        return process_data(train), process_data(dev)
    else:
        train, dev = train_test_split(train, test_size=dev_size)
        return process_data(train), process_data(dev)


def process_data(dataframe):
    x = list(dataframe["text"])
    y = list(dataframe["label"])
    final = []
    for i in range(len(x)):
        temp = {}
        temp["text"] = x[i]
        temp["label"] = str(y[i])
        final.append(temp)
    return MapDataset(final, label_list=list(set(y)))


example_label2text = {
    0: "bad",
    1: "great",
}


def transform_data(example, is_test=False, label_length=None):
    if is_test:
        example["label_length"] = label_length
        return example
    else:
        example["labels"] = example["label"]
        del example["label"]

        return example


def create_dataloader(dataset, mode="train", batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)
