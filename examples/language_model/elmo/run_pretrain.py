import os
import time
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddle.distributed as dist

from args import parse_args, print_args
from elmo import ELMo, ELMoLoss
from dataset import load_vocab, OneBillionWordDataset


def save_params(elmo, optimizer, save_dir, name):
    elmo_ckpt = os.path.join(save_dir, '{}.pdparams'.format(name))
    opt_ckpt = os.path.join(save_dir, '{}.pdopt'.format(name))
    paddle.save(elmo.state_dict(), elmo_ckpt)
    paddle.save(optimizer.state_dict(), opt_ckpt)


def train(args):
    paddle.set_device(args.device)
    n_procs = dist.get_world_size()
    rank = dist.get_rank()

    if n_procs > 1:
        dist.init_parallel_env()

    vocab = load_vocab(args.vocab_file, args.max_characters_per_token)

    elmo = ELMo(args.batch_size,
                args.char_embed_dim,
                args.projection_dim,
                vocab.size,
                dropout=args.dropout,
                num_layers=args.num_layers,
                num_highways=args.num_highways,
                char_vocab_size=vocab.char_size)
    if n_procs > 1:
        elmo = paddle.DataParallel(elmo)
    elmo.train()

    gloabl_norm_clip = nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.Adagrad(learning_rate=args.lr,
                                         parameters=elmo.parameters(),
                                         initial_accumulator_value=1.0,
                                         grad_clip=gloabl_norm_clip)
    elmo_loss = ELMoLoss()

    # Loads pre-trained parameters.
    if args.init_from_ckpt:
        weight_state_dict = paddle.load(args.init_from_ckpt + '.pdparams')
        opt_state_dict = paddle.load(args.init_from_ckpt + '.pdopt')
        elmo.set_state_dict(weight_state_dict)
        optimizer.set_state_dict(opt_state_dict)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    train_dataset = OneBillionWordDataset(args.train_data_path,
                                          vocab,
                                          args.batch_size,
                                          args.unroll_steps,
                                          n_procs=n_procs,
                                          rank=rank,
                                          mode='train',
                                          shuffle=True,
                                          seed=args.seed)

    train_dataloader = DataLoader(train_dataset,
                                  return_list=True,
                                  batch_size=None)

    n_tokens_per_batch = args.batch_size * args.unroll_steps * n_procs
    n_steps_per_epoch = int(train_dataset.number_of_tokens / n_tokens_per_batch)
    n_steps_total = args.epochs * n_steps_per_epoch
    print("Training for %s epochs and %s steps" % (args.epochs, n_steps_total))

    total_time = 0.0
    batch_start_time = time.time()
    for step, inputs in enumerate(train_dataloader, start=1):
        ids, next_ids, ids_reverse, next_ids_reverse = inputs
        outputs = elmo([ids, ids_reverse])
        loss = elmo_loss(outputs, [next_ids, next_ids_reverse])
        ppl = paddle.exp(loss)
        loss *= args.unroll_steps
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        total_time += (time.time() - batch_start_time)
        if step % args.log_freq == 0:
            print("step %d/%d - loss: %.4f - Perplexity: %.4f - %.3fs/step" %
                  (step, n_steps_total, loss.numpy()[0], ppl.numpy()[0],
                   total_time / args.log_freq))
            total_time = 0.0
        if rank == 0 and step % args.save_freq == 0:
            save_params(elmo, optimizer, args.save_dir, step)
        if step == n_steps_total:
            # training done
            if rank == 0:
                save_params(elmo, optimizer, args.save_dir, 'final')
            break
        batch_start_time = time.time()


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    train(args)
