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

import os
import argparse

import numpy as np
import paddle
from paddlenlp.layers.crf import LinearChainCrf
from paddlenlp.utils.tools import compare_version
if compare_version(paddle.version.full_version, "2.2.0") >= 0:
    # paddle.text.ViterbiDecoder is supported by paddle after version 2.2.0
    from paddle.text import ViterbiDecoder
else:
    from paddlenlp.layers.crf import ViterbiDecoder
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer

from data import transfer_str_to_example, convert_example, load_dict
from utils import decode, reset_offset

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default="./output/model_300/model_state.pdparams", required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--data_dir", type=str, default="./data", help="The input data dir, should contain name_category_map.json.")
parser.add_argument("--max_seq_len", type=int, default=64, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def do_predict(data,
               model,
               tokenizer,
               viterbi_decoder,
               tags_to_idx,
               idx_to_tags,
               batch_size=1,
               summary_num=2):

    examples = []
    for text in data:
        example = {"tokens": list(text)}
        input_ids, token_type_ids, seq_len = convert_example(example,
                                                             tokenizer,
                                                             args.max_seq_len,
                                                             is_test=True)

        examples.append((input_ids, token_type_ids, seq_len))

    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # token_type_ids
        Stack(dtype='int64'),  # seq_len
    ): fn(samples)

    all_pred_tags = []

    model.eval()
    for batch in batches:
        input_ids, token_type_ids, seq_len = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        seq_len = paddle.to_tensor(seq_len)
        pred_tags = model(input_ids, token_type_ids, lengths=seq_len)
        all_pred_tags.extend(pred_tags.numpy().tolist())
    results = decode(data, all_pred_tags, summary_num, idx_to_tags)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    data = [
        '美人鱼是周星驰执导的一部电影',
    ]

    tags_to_idx = load_dict(os.path.join(args.data_dir, "tags.txt"))
    idx_to_tags = dict(zip(*(tags_to_idx.values(), tags_to_idx.keys())))

    model = ErnieCtmWordtagModel.from_pretrained("wordtag",
                                                 num_tag=len(tags_to_idx))
    tokenizer = ErnieCtmTokenizer.from_pretrained("wordtag")

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = do_predict(data,
                         model,
                         tokenizer,
                         model.viterbi_decoder,
                         tags_to_idx,
                         idx_to_tags,
                         batch_size=args.batch_size)
    print(results)
