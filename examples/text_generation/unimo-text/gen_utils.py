import random
from functools import partial

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
from paddlenlp.data import Pad


def print_args(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better.
    paddle.seed(seed + dist.get_rank())


def convert_example(example,
                    tokenizer,
                    max_seq_len=512,
                    max_target_len=128,
                    max_title_len=256,
                    mode='train'):
    """Convert all examples into necessary features."""
    source = example['source']
    title = None
    if 'title' in example.keys():
        title = example['title']

    if mode != 'test':
        tokenized_example = tokenizer.gen_encode(source,
                                                 title=title,
                                                 target=example['target'],
                                                 max_seq_len=max_seq_len,
                                                 max_target_len=max_target_len,
                                                 max_title_len=max_title_len,
                                                 return_position_ids=True,
                                                 return_length=True)
        target_start = tokenized_example['input_ids'].index(
            tokenizer.cls_token_id, 1)
        target_end = tokenized_example['seq_len']
        # Use to gather the logits corresponding to the labels during training
        tokenized_example['masked_positions'] = list(
            range(target_start, target_end - 1))
        tokenized_example['labels'] = tokenized_example['input_ids'][
            target_start + 1:target_end]

        return tokenized_example
    else:
        tokenized_example = tokenizer.gen_encode(
            source,
            title=title,
            max_seq_len=max_seq_len,
            max_title_len=max_title_len,
            add_start_token_for_decoding=True,
            return_position_ids=True)

        if 'target' in example and example['target']:
            tokenized_example['target'] = example['target']
        return tokenized_example


def batchify_fn(batch_examples, pad_val, mode):

    def pad_mask(batch_attention_mask):
        batch_size = len(batch_attention_mask)
        max_len = max(map(len, batch_attention_mask))
        attention_mask = np.ones(
            (batch_size, max_len, max_len), dtype='float32') * -1e9
        for i, mask_data in enumerate(attention_mask):
            seq_len = len(batch_attention_mask[i])
            mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i],
                                                       dtype='float32')
        # In order to ensure the correct broadcasting mechanism, expand one
        # dimension to the second dimension (n_head of Transformer).
        attention_mask = np.expand_dims(attention_mask, axis=1)
        return attention_mask

    pad_func = Pad(pad_val=pad_val, pad_right=False, dtype='int64')

    input_ids = pad_func([example['input_ids'] for example in batch_examples])
    token_type_ids = pad_func(
        [example['token_type_ids'] for example in batch_examples])
    position_ids = pad_func(
        [example['position_ids'] for example in batch_examples])

    attention_mask = pad_mask(
        [example['attention_mask'] for example in batch_examples])

    if mode != 'test':
        max_len = max([example['seq_len'] for example in batch_examples])
        masked_positions = np.concatenate([
            np.array(example['masked_positions']) +
            (max_len - example['seq_len']) + i * max_len
            for i, example in enumerate(batch_examples)
        ])
        labels = np.concatenate([
            np.array(example['labels'], dtype='int64')
            for example in batch_examples
        ])
        return input_ids, token_type_ids, position_ids, attention_mask, masked_positions, labels
    else:
        return input_ids, token_type_ids, position_ids, attention_mask


def create_data_loader(dataset, tokenizer, args, mode):
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_len=args.max_seq_len,
                         max_target_len=args.max_target_len,
                         max_title_len=args.max_title_len,
                         mode=mode)
    dataset = dataset.map(trans_func, lazy=True)
    if mode == 'train':
        batch_sampler = DistributedBatchSampler(dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True)
    else:
        batch_sampler = BatchSampler(dataset,
                                     batch_size=args.batch_size // 2,
                                     shuffle=False)
    collate_fn = partial(batchify_fn, pad_val=tokenizer.pad_token_id, mode=mode)
    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=collate_fn,
                             return_list=True)
    return dataset, data_loader


def post_process_sum(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    special_tokens = ['[UNK]']
    tokens = [token for token in tokens if token not in special_tokens]
    return token_ids, tokens


def select_sum(ids,
               scores,
               tokenizer,
               max_dec_len=None,
               num_return_sequences=1):
    results = []
    group = []
    tmp = []
    if scores is not None:
        ids = ids.numpy()
        scores = scores.numpy()

        if len(ids) != len(scores) or (len(ids) % num_return_sequences) != 0:
            raise ValueError(
                "the length of `ids` is {}, but the `num_return_sequences` is {}"
                .format(len(ids), num_return_sequences))

        for pred, score in zip(ids, scores):
            pred_token_ids, pred_tokens = post_process_sum(pred, tokenizer)
            num_token = len(pred_token_ids)

            target = "".join(pred_tokens)

            # not ending
            if max_dec_len is not None and num_token >= max_dec_len:
                score -= 1e3

            tmp.append([target, score])
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []

        for preds in group:
            preds = sorted(preds, key=lambda x: -x[1])
            results.append(preds[0][0])
    else:
        ids = ids.numpy()

        for pred in ids:
            pred_token_ids, pred_tokens = post_process_sum(pred, tokenizer)
            num_token = len(pred_token_ids)
            response = "".join(pred_tokens)

            # TODO: Support return scores in FT.
            tmp.append([response])
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []

        for preds in group:
            results.append(preds[0][0])

    return results
