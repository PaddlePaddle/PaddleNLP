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

import paddle
import paddle.nn.functional as F

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_seq_len",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument("--ckpt_dir", type=str, default=None, help="The directory of saved model checkpoint.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--device",
    choices=["cpu", "gpu", "xpu"],
    default="gpu",
    help="Select which device to train model, defaults to gpu.",
)
parser.add_argument(
    "--model_name",
    choices=["skep_ernie_1.0_large_ch", "skep_ernie_2.0_large_en"],
    default="skep_ernie_1.0_large_ch",
    help="Select which model to train, defaults to skep_ernie_1.0_large_ch.",
)
args = parser.parse_args()


def convert_example_to_feature(example, tokenizer, max_seq_len=512):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens.

    Args:
        example(obj:`str`): The input text to sentiment analysis.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        dataset_name((obj:`str`, defaults to "chnsenticorp"): The dataset name, "chnsenticorp" or "sst-2".

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): The list of token_type_ids.
        label(obj:`int`, optional): The input label if not is_test.
    """
    encoded_inputs = tokenizer(text=example, max_seq_len=max_seq_len)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    return {"input_ids": input_ids, "token_type_ids": token_type_ids}


@paddle.no_grad()
def predict(model, data, tokenizer, label_map, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`list`): All the predictions labels.
    """
    examples = []
    for text in data:
        encoded_inputs = convert_example_to_feature(text, tokenizer, max_seq_len=args.max_seq_len)
        examples.append(encoded_inputs)

    # Separates data into some batches.
    batches = [examples[idx : idx + batch_size] for idx in range(0, len(examples), batch_size)]

    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    results = []
    model.eval()
    for raw_batch in batches:
        batch = data_collator(raw_batch)
        input_ids, token_type_ids = batch["input_ids"], batch["token_type_ids"]
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy().tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    # These data samples is in Chinese.
    # If you use the english model, you should change the test data in English.
    data = [
        "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般",
        "怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片",
        "作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。",
    ]
    label_map = {0: "negative", 1: "positive"}

    tokenizer = SkepTokenizer.from_pretrained(args.model_name)
    model = SkepForSequenceClassification.from_pretrained(args.ckpt_dir, num_labels=len(label_map))
    print("Loaded model from %s" % args.ckpt_dir)

    results = predict(model, data, tokenizer, label_map, batch_size=args.batch_size)
    for idx, text in enumerate(data):
        print("Data: {} \t Label: {}".format(text, results[idx]))
