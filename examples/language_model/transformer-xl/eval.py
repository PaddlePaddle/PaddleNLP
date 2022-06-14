import os
import time
import yaml
import logging
import argparse
import numpy as np
from pprint import pprint
from attrdict import AttrDict

import paddle

from reader import get_lm_vocab, get_lm_data_loader
from mem_transformer import MemTransformerLM

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


def do_eval(args):
    assert args.ext_len >= 0, 'Extended context length must be no less than 0'

    def _evaluate(loader):
        total_len, total_loss = 0, 0.

        eval_mems = tuple()
        for i, (src, target, seq_len) in enumerate(loader):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = mem_transformer(src, target, *eval_mems)
            loss, eval_mems = ret[0], ret[1:]
            eval_cur_loss = seq_len * loss.numpy()
            total_loss += eval_cur_loss
            total_len += seq_len
        return total_loss / total_len

    def _logger(loss):
        if args.dataset in ['enwik8', 'text8']:
            logger_info = "loss: %f, bpc: %f" % \
                          (loss, loss / np.log(2))
        else:
            logger_info = "loss: %f, ppl: %.2f" % \
                          (loss, np.exp(loss))
        return logger_info

    if not args.use_gpu:
        paddle.set_device("cpu")

    vocab = get_lm_vocab(args)
    eval_loader = get_lm_data_loader(args, vocab, "valid")
    test_loader = get_lm_data_loader(args, vocab, "test")

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

    assert args.init_from_params, (
        "Please set init_from_params to load the infer model.")

    model_dict = paddle.load(
        os.path.join(args.init_from_params, "mem_transformer.pdparams"))
    mem_transformer.load_dict(model_dict)

    logger.info(
        "Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}".
        format(args.eval_batch_size, args.tgt_len, args.ext_len, args.mem_len,
               args.clamp_len))

    mem_transformer.reset_length(args.tgt_len, args.ext_len, args.mem_len)

    test_loss = None
    valid_loss = None
    if args.mode == 'all':
        test_loss = _evaluate(test_loader)
        valid_loss = _evaluate(eval_loader)
    elif args.mode == 'valid':
        valid_loss = _evaluate(eval_loader)
    elif args.mode == 'test':
        test_loss = _evaluate(test_loader)

    logger_info = ''
    if valid_loss is not None:
        logger_info = logger_info + "validation loss: " + _logger(
            valid_loss) + " | "
    if test_loss is not None:
        logger_info = logger_info + "test loss: " + _logger(test_loss) + " | "
    logger.info(logger_info)


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_eval(args)
