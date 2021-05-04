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
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
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
        label_list(obj:`list[str]`): All the labels that the data has.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask. 
    """
    encoded_inputs = tokenizer(
        text=example[0], text_pair=example[1], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    return input_ids, token_type_ids


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
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text in data:
        input_ids, token_type_ids = convert_example(
            text,
            tokenizer,
            label_list=label_map.values(),
            max_seq_length=args.max_seq_length,
            is_test=True)
        examples.append((input_ids, token_type_ids))

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token type ids
    ): [data for data in fn(samples)]

    results = []
    model.eval()
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    data = [
        ('phone#design_features',
         'K860入手一天感受前天晚上京东下单，昨天早上就送到公司了，同事代收的货。昨天下午四点多机器到手玩到今天，一天多点时间，'
         '初步印象：1、机身很大，刚开始极不习惯，好像拿个山寨机的感觉。适应了一整天稍微好点了。不过机身还是很薄的，放在裤子口袋无压力。'
         '2、屏幕漂亮，看电子书极爽。后摄像头突出容易手摸到，也容易磨损，迫切需要手机套保护。3、昨晚放进移动SIM卡开始使用，一直到今天早上'
         '九点都没有电话进来，这才觉得有点奇怪，遂尝试拨打电话无效，取出卡再插入解决，这算是接触不良吗？4、外放音乐效果差强人意，只有高音'
         '没有重低音，完败于前一个手机华为U8800。5、游戏没怎么尝试，4核应该没什么压力吧。6、摄像头是重点，随手拍了二三十张照片，效果上佳，'
         '昏暗环境下ISO自动上到1200，噪点居然也不是很多。7、待机：昨晚充满电，到今晚八点剩余60%，期间打电话12个，拍照二十几张，看电子书10'
         '分钟，上网查资料半小时。估计轻度使用应该可以撑2天。'),
        ('display#quality',
         'K860入手一天感受前天晚上京东下单，昨天早上就送到公司了，同事代收的货。昨天下午四点多机器到手玩到今天，一天多点时间，'
         '初步印象：1、机身很大，刚开始极不习惯，好像拿个山寨机的感觉。适应了一整天稍微好点了。不过机身还是很薄的，放在裤子口袋无压力。'
         '2、屏幕漂亮，看电子书极爽。后摄像头突出容易手摸到，也容易磨损，迫切需要手机套保护。3、昨晚放进移动SIM卡开始使用，一直到今天早上'
         '九点都没有电话进来，这才觉得有点奇怪，遂尝试拨打电话无效，取出卡再插入解决，这算是接触不良吗？4、外放音乐效果差强人意，只有高音'
         '没有重低音，完败于前一个手机华为U8800。5、游戏没怎么尝试，4核应该没什么压力吧。6、摄像头是重点，随手拍了二三十张照片，效果上佳，'
         '昏暗环境下ISO自动上到1200，噪点居然也不是很多。7、待机：昨晚充满电，到今晚八点剩余60%，期间打电话12个，拍照二十几张，看电子书10'
         '分钟，上网查资料半小时。估计轻度使用应该可以撑2天。'),
    ]
    label_map = {0: 'negative', 1: 'positive'}

    model = ppnlp.transformers.SkepForSequenceClassification.from_pretrained(
        "skep_ernie_1.0_large_ch", num_classes=len(label_map))
    tokenizer = ppnlp.transformers.SkepTokenizer.from_pretrained(
        "skep_ernie_1.0_large_ch")

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = predict(
        model, data, tokenizer, label_map, batch_size=args.batch_size)
    for idx, text in enumerate(data):
        print('Data: {} \t Label: {}'.format(text, results[idx]))
