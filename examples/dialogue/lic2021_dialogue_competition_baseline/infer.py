import os
import time
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader

from args import parse_args, print_args
from model import UnifiedTransformer
from data import DialogueDataset, Vocabulary, select_response


def repeat_tensor(x, times):
    repeat_times = [times] + [1] * (len(x.shape) - 1)
    return paddle.tile(x, repeat_times)


def load_ckpt(init_from_ckpt, model):
    params_state_dict = paddle.load(init_from_ckpt + '.pdparams')
    model.set_state_dict(params_state_dict)
    print('Loaded checkpoint from %s' % init_from_ckpt)


def main(args):
    paddle.set_device('gpu' if args.n_gpus else 'cpu')
    paddle.seed(args.seed)

    vocab = Vocabulary(args.vocab_file)

    model = UnifiedTransformer(
        args.num_layers,
        args.d_model,
        args.nhead,
        args.dropout,
        args.activation,
        args.normalize_before,
        vocab.size,
        args.type_size,
        args.max_seq_len,
        vocab.unk_id,
        vocab.bos_id,
        vocab.eos_id,
        vocab.mask_id,
        vocab.pad_id,
        is_infer=True)

    test_dataset = DialogueDataset(
        args.test_data_path, vocab, args.infer_batch_size, mode='test')
    test_dataloader = DataLoader(
        test_dataset, return_list=True, batch_size=None)

    if args.init_from_ckpt:
        load_ckpt(args.init_from_ckpt, model)
    else:
        raise ValueError('"init_from_ckpt" must be set when doing infer.')

    infer(model, test_dataloader, vocab)


@paddle.no_grad()
def infer(model, data_loader, vocab):
    print('\nInfer begin...')
    model.eval()
    total_time = 0.0
    start_time = time.time()
    responses = []
    for step, inputs in enumerate(data_loader, 1):
        token_ids, type_ids, pos_ids, generation_mask = inputs
        ids, scores = model.generate(
            input_ids=token_ids,
            token_type_ids=type_ids,
            position_ids=pos_ids,
            attention_mask=generation_mask,
            max_length=args.max_dec_len,
            min_length=args.min_dec_len,
            decode_strategy=args.decode_strategy,
            top_k=args.topk,
            num_return_sequences=args.num_samples)

        total_time += (time.time() - start_time)
        if step % args.logging_steps == 0:
            print('step %d - %.3fs/step' %
                  (step, total_time / args.logging_steps))
            total_time = 0.0
        results = select_response(ids, scores, vocab, args.max_dec_len,
                                  args.num_samples)
        responses.extend(results)

        start_time = time.time()
        #break

    with open(args.output_path, 'w') as fout:
        for response in responses:
            fout.write(response + '\n')
    print('Save inference result into: %s' % args.output_path)


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    main(args)
