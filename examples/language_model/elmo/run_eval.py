import os
import time
import math
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddle.distributed as dist

from args import parse_args, print_args
from elmo import ELMo, ELMoLoss
from dataset import load_vocab, OneBillionWordDataset


@paddle.no_grad()
def eval(args):
    paddle.set_device(args.device)

    if not args.init_from_ckpt:
        raise ValueError('init_from_ckpt should be set when eval.')
    vocab = load_vocab(args.vocab_file, args.max_characters_per_token)

    elmo = ELMo(args.batch_size,
                args.char_embed_dim,
                args.projection_dim,
                vocab.size,
                dropout=args.dropout,
                num_layers=args.num_layers,
                num_highways=args.num_highways,
                char_vocab_size=vocab.char_size)
    elmo.eval()

    elmo_loss = ELMoLoss()

    # Loads pre-trained parameters.
    weight_state_dict = paddle.load(args.init_from_ckpt + '.pdparams')
    elmo.set_state_dict(weight_state_dict)
    print("Loaded checkpoint from %s" % args.init_from_ckpt)

    dev_dataset = OneBillionWordDataset(args.dev_data_path,
                                        vocab,
                                        args.batch_size,
                                        args.unroll_steps,
                                        mode='test',
                                        shuffle=False,
                                        seed=args.seed)

    dev_dataloader = DataLoader(dev_dataset, return_list=True, batch_size=None)

    total_step = total_loss = 0
    total_time = 0.0
    batch_start_time = time.time()
    for step, inputs in enumerate(dev_dataloader, start=1):
        ids, next_ids, ids_reverse, next_ids_reverse = inputs
        outputs = elmo([ids, ids_reverse])
        loss = elmo_loss(outputs, [next_ids, next_ids_reverse])
        ppl = paddle.exp(loss)

        total_loss += loss.numpy()[0]
        total_step += 1

        total_time += (time.time() - batch_start_time)
        if step % args.log_freq == 0:
            print("Eval step %d - loss: %.4f - Perplexity: %.4f - %.3fs/step" %
                  (step, loss.numpy()[0] * args.unroll_steps, ppl.numpy()[0],
                   total_time / args.log_freq))
            total_time = 0.0
        batch_start_time = time.time()

    avg_loss = total_loss / total_step
    avg_ppl = math.exp(avg_loss)
    print("Eval - average loss: %.4f - average Perplexity: %.4f" %
          (avg_loss * args.unroll_steps, avg_ppl))


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    eval(args)
