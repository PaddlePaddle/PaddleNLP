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

import math
import time

import paddle
from args import parse_args, print_args
from dataset import OneBillionWordDataset, load_vocab
from elmo import ELMo, ELMoLoss
from paddle.io import DataLoader


@paddle.no_grad()
def eval(args):
    paddle.set_device(args.device)

    if not args.init_from_ckpt:
        raise ValueError("init_from_ckpt should be set when eval.")
    vocab = load_vocab(args.vocab_file, args.max_characters_per_token)

    elmo = ELMo(
        args.batch_size,
        args.char_embed_dim,
        args.projection_dim,
        vocab.size,
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_highways=args.num_highways,
        char_vocab_size=vocab.char_size,
    )
    elmo.eval()

    elmo_loss = ELMoLoss()

    # Loads pre-trained parameters.
    weight_state_dict = paddle.load(args.init_from_ckpt + ".pdparams")
    elmo.set_state_dict(weight_state_dict)
    print("Loaded checkpoint from %s" % args.init_from_ckpt)

    dev_dataset = OneBillionWordDataset(
        args.dev_data_path, vocab, args.batch_size, args.unroll_steps, mode="test", shuffle=False, seed=args.seed
    )

    dev_dataloader = DataLoader(dev_dataset, return_list=True, batch_size=None)

    total_step = total_loss = 0
    total_time = 0.0
    batch_start_time = time.time()
    for step, inputs in enumerate(dev_dataloader, start=1):
        ids, next_ids, ids_reverse, next_ids_reverse = inputs
        outputs = elmo([ids, ids_reverse])
        loss = elmo_loss(outputs, [next_ids, next_ids_reverse])
        ppl = paddle.exp(loss)

        total_loss += float(loss)
        total_step += 1

        total_time += time.time() - batch_start_time
        if step % args.log_freq == 0:
            print(
                "Eval step %d - loss: %.4f - Perplexity: %.4f - %.3fs/step"
                % (step, float(loss) * args.unroll_steps, float(ppl), total_time / args.log_freq)
            )
            total_time = 0.0
        batch_start_time = time.time()

    avg_loss = total_loss / total_step
    avg_ppl = math.exp(avg_loss)
    print("Eval - average loss: %.4f - average Perplexity: %.4f" % (avg_loss * args.unroll_steps, avg_ppl))


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    eval(args)
