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
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
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
    tokens = list(example)
    encoded_inputs = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    seq_len = encoded_inputs["seq_len"]

    return input_ids, token_type_ids, seq_len


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
        input_ids, token_type_ids, seq_len = convert_example(
            text, tokenizer, max_seq_length=args.max_seq_length, is_test=True)
        examples.append((input_ids, token_type_ids, seq_len))

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # token type ids
        Stack(dtype="int64")  # seq lens
    ): [data for data in fn(samples)]

    results = []
    model.eval()
    for batch in batches:
        input_ids, token_type_ids, seq_lens = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        seq_lens = paddle.to_tensor(seq_lens)
        preds = model(input_ids, token_type_ids, seq_lens=seq_lens)
        tags = parse_predict_result(preds.numpy(), seq_lens.numpy(), label_map)
        results.extend(tags)
    return results


def parse_predict_result(predictions, seq_lens, label_map):
    """按需解析模型预测出来的结果
    :param predict_result: 模型预测出来的结果
    :return:
    """
    pred_tag = []
    for idx, pred in enumerate(predictions):
        seq_len = seq_lens[idx]
        # drop the "[CLS]" and "[SEP]" token
        tag = [label_map[i] for i in pred[1:seq_len - 1]]
        pred_tag.append(tag)
    return pred_tag


if __name__ == "__main__":
    paddle.set_device(args.device)

    # These data samples is in Chinese.
    # If you use the english model, you should change the test data in English.
    data = [
        '还是第一次进星巴克店里吃东西,那会儿第一次喝咖啡还是外带的',
        '阿春粤菜馆普君新城店在普君新城的二楼，从进入地下停车场开始就一直有明确的指示牌指引停车方向，'
        '汽车直达负二层E区停车后再搭乘手扶电梯沿路跟着指示牌步行就可以找到阿春粤菜馆了',
        '去三亚的时候去吃了大东海的拾味馆.得到了全家的一致好评.没想到学校附近也有一家.'
        '果断和室友约着看电影的时候我去吃.由于对椰香骨汤印象很深刻.浓浓的骨汤头里还有着椰子的清香味，'
        '喝完口也不会有很干的感觉，推荐.凉粉中规中矩，有点偏咸，总体还是不错的.香糯的椰子饭值得一试.'
        '在三亚时海南四大名菜就东山羊没能吃到，在这里终于凑齐了，东山羊刚入口时完全吃不出有羊的膻味，搭配蘸酱吃更好吃了，'
        '不过吃到后来膻味就出来了.整体来说还是不错的.不过觉得没三亚的那家氛围好.',
    ]
    # The COTE_DP dataset labels with "BIO" schema.
    label_map = {"B": 0, "I": 1, "O": 2}
    inversed_label_map = {value: key for key, value in label_map.items()}
    # `no_entity_label` represents that the token isn't an entity. 
    no_entity_label = "O"
    # `ignore_label` is using to pad input labels.
    ignore_label = -1

    skep = ppnlp.transformers.SkepModel.from_pretrained(
        'skep_ernie_1.0_large_ch')
    model = ppnlp.transformers.SkepCrfForTokenClassification(
        skep, num_classes=len(label_map))
    tokenizer = ppnlp.transformers.SkepTokenizer.from_pretrained(
        'skep_ernie_1.0_large_ch')

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = predict(
        model, data, tokenizer, inversed_label_map, batch_size=args.batch_size)
    for idx, text in enumerate(data):
        print(len(text), len(results[idx]))
        print('Data: {} \t Label: {}'.format(text, results[idx]))
