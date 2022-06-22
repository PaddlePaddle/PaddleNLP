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

from args import parse_args

import paddle
import paddle.nn as nn
from paddlenlp.metrics import Perplexity

from seq2seq_attn import Seq2SeqAttnModel, CrossEntropyCriterion
from data import create_train_loader


def do_train(args):
    device = paddle.set_device(args.device)

    # Define dataloader
    train_loader, eval_loader, src_vocab_size, tgt_vocab_size, eos_id = create_train_loader(
        args)

    model = paddle.Model(
        Seq2SeqAttnModel(src_vocab_size, tgt_vocab_size, args.hidden_size,
                         args.hidden_size, args.num_layers, args.dropout,
                         eos_id))

    grad_clip = nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate,
                                      parameters=model.parameters(),
                                      grad_clip=grad_clip)

    ppl_metric = Perplexity()
    model.prepare(optimizer, CrossEntropyCriterion(), ppl_metric)

    print(args)
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    benchmark_logger = paddle.callbacks.ProgBarLogger(log_freq=args.log_freq,
                                                      verbose=3)

    model.fit(train_data=train_loader,
              eval_data=eval_loader,
              epochs=args.max_epoch,
              eval_freq=1,
              save_freq=1,
              save_dir=args.model_path,
              callbacks=[benchmark_logger])


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
