import os
import time
import yaml
import logging
import argparse
import numpy as np
from pprint import pprint
from attrdict import AttrDict

import paddle
import paddle.nn as nn
import paddle.distributed as dist

from mem_transformer import MemTransformerLM
from reader import get_lm_vocab, get_lm_data_loader

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="./configs/enwik8.yaml",
                        type=str,
                        help="Path of the config file. ")
    args = parser.parse_args()
    return args


def do_train(args):
    if args.use_gpu:
        rank = dist.get_rank()
        trainer_count = dist.get_world_size()
    else:
        rank = 0
        trainer_count = 1
        paddle.set_device("cpu")

    if trainer_count > 1:
        dist.init_parallel_env()

    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    vocab = get_lm_vocab(args)
    train_loader = get_lm_data_loader(args, vocab, "train")
    eval_loader = get_lm_data_loader(args, vocab, "valid")

    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [20000, 40000, 200000]
            tie_projs += [True] * len(cutoffs)
        elif args.dataset == 'lm1b':
            cutoffs = [60000, 100000, 640000]
            tie_projs += [False] * len(cutoffs)

    mem_transformer = MemTransformerLM(args.ntokens,
                                       args.n_layer,
                                       args.n_head,
                                       args.d_model,
                                       args.d_head,
                                       args.d_inner_hid,
                                       args.dropout,
                                       args.attn_dropout,
                                       tie_weight=args.tie_weight,
                                       d_embed=args.d_model,
                                       div_val=args.div_val,
                                       tie_projs=tie_projs,
                                       normalize_before=args.normalize_before,
                                       tgt_len=args.tgt_len,
                                       ext_len=args.ext_len,
                                       mem_len=args.mem_len,
                                       cutoffs=cutoffs,
                                       same_length=args.same_length,
                                       attn_type=args.attn_type,
                                       clamp_len=args.clamp_len,
                                       sample_softmax=args.sample_softmax)

    if args.scheduler == 'cosine':
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=args.learning_rate,
            T_max=args.max_step,
            eta_min=args.eta_min)
    elif args.scheduler == 'noam':
        scheduler = paddle.optimizer.lr.NoamDecay(
            d_model=args.d_model,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate)
    elif args.scheduler == 'dev_perf':
        paddle.optimizer.lr.ReduceOnPlateau(learning_rate=args.learning_rate,
                                            factor=args.decay_rate,
                                            patience=args.patience,
                                            min_lr=args.lr_min)
    elif args.scheduler == 'constant':
        scheduler = args.learning_rate

    clip = paddle.nn.ClipGradByGlobalNorm(args.clip)
    if args.optim.lower() == 'momentum':
        optimizer = paddle.optimizer.Momentum(
            learning_rate=scheduler,
            parameters=mem_transformer.parameters(),
            momentum=args.mom,
            grad_clip=clip)
    elif args.optim.lower() == 'adam':
        optimizer = paddle.optimizer.Adam(
            learning_rate=scheduler,
            parameters=mem_transformer.parameters(),
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=eval(args.eps),
            grad_clip=clip)
    elif args.optim.lower() == 'adagrad':
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=scheduler,
            parameters=mem_transformer.parameters(),
            grad_clip=clip)

    # Init from some checkpoint, to resume the previous training
    if args.init_from_checkpoint:
        model_dict = paddle.load(
            os.path.join(args.init_from_checkpoint, "mem_transformer.pdparams"))
        opt_dict = paddle.load(
            os.path.join(args.init_from_checkpoint, "mem_transformer.pdopt"))
        mem_transformer.set_state_dict(model_dict)
        optimizer.set_state_dict(opt_dict)
        print("loaded from checkpoint.")
    # Init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        model_dict = paddle.load(
            os.path.join(args.init_from_pretrain_model,
                         "mem_transformer.pdparams"))
        mem_transformer.set_state_dict(model_dict)
        print("loaded from pre-trained model.")

    if trainer_count > 1:
        mem_transformer = paddle.DataParallel(mem_transformer)

    step_idx = 0
    train_loss = 0.0

    log_start_time = time.time()

    for pass_id in range(args.epoch):
        batch_id = 0

        mems = tuple()
        for input_data in train_loader:
            (src, target, seq_len) = input_data
            ret = mem_transformer(src, target, *mems)
            loss = ret[0]
            mems = ret[1:]
            train_loss += loss.numpy()

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if step_idx > 0 and step_idx % args.print_step == 0 and rank == 0:
                cur_loss = train_loss / args.print_step
                elapsed = time.time() - log_start_time
                if args.scheduler == "constant":
                    lr = optimizer.get_lr()
                else:
                    lr = scheduler.get_lr()
                logger_info = "step_idx: %d, epoch: %d, batch: %d, learning rate: %.8f, " \
                              "speed: %f ms/batch, loss: %f" % \
                              (step_idx, pass_id, batch_id, lr,
                               elapsed * 1000.0 / args.print_step, cur_loss)
                if args.dataset in ["enwik8", "text8"]:
                    logger_info = logger_info + ", bpc: %f" % (cur_loss /
                                                               np.log(2))
                else:
                    logger_info = logger_info + ", ppl: %f" % (np.exp(cur_loss))

                logger.info(logger_info)
                train_loss = 0.0
                log_start_time = time.time()

            if step_idx % args.save_step == 0 and step_idx != 0:
                # Do validation.
                mem_transformer.eval()

                # TODO(FrostML): simplify this.
                if args.mem_len == 0:
                    if dist.get_world_size() == 1:
                        mem_transformer.reset_length(tgt_len=args.eval_tgt_len,
                                                     ext_len=args.ext_len +
                                                     args.tgt_len -
                                                     args.eval_tgt_len,
                                                     mem_len=args.mem_len)
                    else:
                        mem_transformer._layers.reset_length(
                            tgt_len=args.eval_tgt_len,
                            ext_len=args.ext_len + args.tgt_len -
                            args.eval_tgt_len,
                            mem_len=args.mem_len)
                else:
                    if dist.get_world_size() == 1:
                        mem_transformer.reset_length(tgt_len=args.eval_tgt_len,
                                                     ext_len=args.ext_len,
                                                     mem_len=args.mem_len +
                                                     args.tgt_len -
                                                     args.eval_tgt_len)
                    else:
                        mem_transformer._layers.reset_length(
                            tgt_len=args.eval_tgt_len,
                            ext_len=args.ext_len,
                            mem_len=args.mem_len + args.tgt_len -
                            args.eval_tgt_len)

                total_len, total_loss = 0, 0.

                eval_mems = tuple()
                with paddle.no_grad():
                    for i, (src, target, seq_len) in enumerate(eval_loader):
                        if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                            break
                        ret = mem_transformer(src, target, *eval_mems)
                        loss, eval_mems = ret[0], ret[1:]
                        eval_cur_loss = seq_len * loss.numpy()
                        total_loss += eval_cur_loss
                        total_len += seq_len
                    eval_loss = total_loss / total_len

                logger_info = "Validation, step_idx: %d, validation loss: %f" % \
                            (step_idx, eval_loss)
                if args.dataset in ['enwik8', 'text8']:
                    logger_info = logger_info + ", bpc: %f" % (eval_loss /
                                                               np.log(2))
                else:
                    logger_info = logger_info + ", ppl: %f" % (
                        np.exp(eval_loss))
                logger.info(logger_info)

                if args.save_model and rank == 0:
                    model_dir = os.path.join(args.save_model,
                                             "step_" + str(step_idx))
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    paddle.save(
                        mem_transformer.state_dict(),
                        os.path.join(model_dir, "mem_transformer.pdparams"))
                    paddle.save(
                        optimizer.state_dict(),
                        os.path.join(model_dir, "mem_transformer.pdopt"))
                    f = open(
                        os.path.join(args.save_model, "step_" + str(step_idx),
                                     "evaluation_loss_" + str(eval_loss)), "w")
                    f.close()

                if args.scheduler == 'dev_perf':
                    scheduler.step(eval_loss)

                # TODO(FrostML): simplify this.
                if dist.get_world_size() == 1:
                    mem_transformer.reset_length(tgt_len=args.tgt_len,
                                                 ext_len=args.ext_len,
                                                 mem_len=args.mem_len)
                else:
                    mem_transformer._layers.reset_length(tgt_len=args.tgt_len,
                                                         ext_len=args.ext_len,
                                                         mem_len=args.mem_len)

                mem_transformer.train()

            if step_idx >= args.max_step:
                return
            step_idx += 1
            batch_id += 1
            if args.scheduler in ['cosine', 'dev_perf']:
                if step_idx < args.warmup_steps:
                    curr_lr = args.learning_rate * step_idx / args.warmup_steps
                    scheduler.base_lr = curr_lr
                else:
                    if args.scheduler == 'cosine':
                        scheduler.step()
            elif args.scheduler == 'constant':
                if step_idx < args.warmup_steps:
                    curr_lr = args.learning_rate * step_idx / args.warmup_steps
                    optimizer.set_lr(curr_lr)
            elif args.scheduler == 'noam':
                scheduler.step()

    if args.save_model and rank == 0:
        model_dir = os.path.join(args.save_model, "step_final")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        paddle.save(mem_transformer.state_dict(),
                    os.path.join(model_dir, "mem_transformer.pdparams"))
        paddle.save(optimizer.state_dict(),
                    os.path.join(model_dir, "mem_transformer.pdopt"))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_train(args)
