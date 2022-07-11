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

import argparse
import json
import time
from functools import partial

import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoTokenizer, AutoModel

from utils import Preprocessor, set_seed, convert_example
from model import TPLinkerPlus, HandshakingTaggingScheme
from metric import MetricsCalculator


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(args.seed)

    train_data = json.load(open(args.train_path, "r", encoding="utf-8"))
    valid_data = json.load(open(args.dev_path, "r", encoding="utf-8"))

    encoder = AutoModel.from_pretrained("ernie-3.0-base-zh")
    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer(text,
                                                   return_token_type_ids=None,
                                                   return_offsets_mapping=True,
                                                   add_special_tokens=False)[
                                                       "offset_mapping"]

    preprocessor = Preprocessor(
        tokenize_func=tokenize,
        get_tok2char_span_map_func=get_tok2char_span_map)

    max_tok_num = 0
    all_data = train_data + valid_data

    for sample in all_data:
        tokens = tokenize(sample["text"])
        max_tok_num = max(max_tok_num, len(tokens))

    if max_tok_num > args.max_seq_len:
        train_data = preprocessor.split_into_short_samples(
            train_data, args.max_seq_len, sliding_len=args.sliding_len)
        valid_data = preprocessor.split_into_short_samples(
            valid_data, args.max_seq_len, sliding_len=args.sliding_len)

    max_seq_len = min(max_tok_num, args.max_seq_len)
    rel2id = json.load(open(args.rel2id_path, "r", encoding="utf-8"))
    ent2id = json.load(open(args.ent2id_path, "r", encoding="utf-8"))
    handshaking_tagger = HandshakingTaggingScheme(rel2id, max_seq_len, ent2id)
    tag_size = handshaking_tagger.get_tag_size()

    train_ds = MapDataset(train_data)
    dev_ds = MapDataset(valid_data)

    train_ds = train_ds.map(
        partial(convert_example,
                tokenizer=tokenizer,
                shaking_tagger=handshaking_tagger,
                max_seq_len=args.max_seq_len))

    dev_ds = dev_ds.map(
        partial(convert_example,
                tokenizer=tokenizer,
                shaking_tagger=handshaking_tagger,
                max_seq_len=args.max_seq_len))

    train_batch_sampler = paddle.io.BatchSampler(dataset=train_ds,
                                                 batch_size=args.batch_size,
                                                 shuffle=True)
    train_data_loader = paddle.io.DataLoader(dataset=train_ds,
                                             batch_sampler=train_batch_sampler,
                                             return_list=True)

    dev_batch_sampler = paddle.io.BatchSampler(dataset=dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    dev_data_loader = paddle.io.DataLoader(dataset=dev_ds,
                                           batch_sampler=dev_batch_sampler,
                                           return_list=True)

    model = TPLinkerPlus(encoder, tag_size)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    metric = MetricsCalculator(handshaking_tagger)
    loss_func = lambda y_pred, y_true: metric.loss_func(
        y_pred, y_true, ghm=False)

    optimizer = paddle.optimizer.AdamW(learning_rate=args.learning_rate,
                                       parameters=model.parameters())

    loss_list = []
    global_step = 0
    best_step = 0
    best_f1 = 0
    tic_train = time.time()

    for epoch in range(1, args.num_epochs + 1):
        for batch in train_data_loader:
            input_ids, token_type_ids, att_mask, shaking_tags = batch
            pred_small_shaking_outputs, sampled_tok_pair_indices = model(
                input_ids, att_mask, token_type_ids)


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--rel2id_path", default="./data/rel2id.json", type=str, help="The file path of the mappings of relations.")
    parser.add_argument("--ent2id_path", default="./data/ent2id.json", type=str, help="The file path of the mappings of entities.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length. "
        "Sequences longer than this will be split automatically.")
    parser.add_argument("--sliding_len", default=50, type=int, help="")
    parser.add_argument("--num_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=10, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=100, type=int, help="The interval steps to evaluate model performance.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--model", choices=["uie-base", "uie-tiny", "uie-medium", "uie-mini", "uie-micro", "uie-nano"], default="uie-base", type=str, help="Select the pretrained model for few-shot learning.")
    parser.add_argument("--init_from_ckpt", default=None, type=str, help="The path of model parameters for initialization.")

    args = parser.parse_args()
    # yapf: enable

    do_train()
