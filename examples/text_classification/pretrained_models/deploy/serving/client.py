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
import numpy as np
import sys

import paddle
from paddlenlp.data import Tuple, Pad
from paddlenlp.transformers import ErnieTinyTokenizer
from paddle_serving_client import Client
from scipy.special import softmax

sys.path.append('./')

from utils import convert_example

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--client_config_file", type=str, default="./serving_client/serving_client_conf.prototxt", help="Client prototxt config file.")
parser.add_argument("--server_ip_port", type=str, default="127.0.0.1:8090", help="The ip address and port of the server.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU/CPU for training.")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
args = parser.parse_args()
# yapf: enable


def predict(data, label_map, batch_size):
    """
    Args:
        sentences (list[str]): each string is a sentence. If have sentences then no need paths
        paths (list[str]): The paths of file which contain sentences. If have paths then no need sentences
    Returns:
        res (list(numpy.ndarray)): The result of sentence, indicate whether each word is replaced, same shape with sentences.
    """
    # TODO: Text tokenization which is done in the serving end not the client end may be better.
    tokenizer = ErnieTinyTokenizer.from_pretrained("ernie-tiny")
    examples = []
    for text in data:
        example = {"text": text}
        input_ids, token_type_ids = convert_example(
            example,
            tokenizer,
            max_seq_length=args.max_seq_length,
            is_test=True)
        examples.append((input_ids, token_type_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'
            ),  # token type ids
    ): fn(samples)

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]

    # initialize client
    client = Client()
    client.load_client_config(args.client_config_file)
    client.connect([args.server_ip_port])

    results = []
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        fetch_map = client.predict(feed={
            "input_ids": input_ids,
            "token_type_ids": token_type_ids
        },
                                   fetch=["save_infer_model/scale_0.tmp_1"],
                                   batch=True)
        output_data = np.array(fetch_map["save_infer_model/scale_0.tmp_1"])
        probs = softmax(output_data, axis=1)
        idx = np.argmax(probs, axis=1)
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)

    return results


if __name__ == '__main__':
    paddle.enable_static()
    data = [
        '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般',
        '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
        '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    ]
    label_map = {0: 'negative', 1: 'positive'}
    results = predict(data, label_map, args.batch_size)
    for idx, text in enumerate(data):
        print('Data: {} \t Label: {}'.format(text, results[idx]))
