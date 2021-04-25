import random
import argparse
from functools import partial

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
from paddlenlp.data import Pad


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_name_or_path', type=str, default='unified_transformer-12L-cn-luge', help='The path or shortcut name of the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='The directory where the checkpoints will be saved.')
    parser.add_argument('--output_path', type=str, default='./predict.txt', help='The file path where the infer result will be saved.')
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every X updates steps.')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for initialization.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--lr', type=float, default=5e-5, help='The initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='The weight decay for optimizer.')
    parser.add_argument('--epochs', type=int, default=3, help='Total number of training epochs to perform.')
    parser.add_argument('--warmup_steps', type=int, default=2500, help='The number of warmup steps.')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='The max value of grad norm.')
    parser.add_argument('--max_seq_len', type=int, default=512, help='The maximum sequence length of training.')
    parser.add_argument('--max_response_len', type=int, default=128, help='The maximum response sequence length of training.')
    parser.add_argument('--max_knowledge_len', type=int, default=256, help='The maximum knowledge sequence length of training.')
    parser.add_argument('--min_dec_len', type=int, default=1, help='The minimum sequence length of generation.')
    parser.add_argument('--max_dec_len', type=int, default=64, help='The maximum sequence length of generation.')
    parser.add_argument('--num_samples', type=int, default=1, help='The decode numbers in generation.')
    parser.add_argument('--decode_strategy', type=str, default='sampling', help='The decode strategy in generation.')
    parser.add_argument('--top_k', type=int, default=0, help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--temperature', type=float, default=1.0, help='The value used to module the next token probabilities.')
    parser.add_argument('--top_p', type=float, default=1.0, help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams', type=int, default=0, help='The number of beams for beam search.')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--early_stopping', type=eval, default=False, help='Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.')
    parser.add_argument('--device', type=str, default='gpu', help='The device to select for training the model.')

    args = parser.parse_args()
    return args
# yapf: enable


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


def preprocess_examples(examples, mode='train'):
    """
    For training set and dev set, treat each utterance of the first speaker as 
    the response, and concatenate the goal, knowledge and the dialog’s previous 
    utterances as the history. In this way, multiple history-response pairs 
    are constructed.
    """
    if mode == 'test':
        return examples
    new_examples = []
    for example in examples:
        conversation = example['conversation']
        for i in range(0, len(conversation), 2):
            new_examples.append({
                'goal': example['goal'],
                'knowledge': example['knowledge'],
                'history': conversation[:i],
                'response': conversation[i]
            })
    return new_examples


def convert_example(example,
                    tokenizer,
                    max_seq_len=512,
                    max_response_len=128,
                    max_knowledge_len=256,
                    mode='train'):
    """Convert all examples into necessary features."""
    goal = example['goal']
    knowledge = example['knowledge']
    goal_knowledge = ' '.join([' '.join(lst) for lst in goal + knowledge])

    if mode != 'test':
        tokenized_example = tokenizer.dialogue_encode(
            example['history'],
            response=example['response'],
            knowledge=goal_knowledge,
            task_type='knowledge',
            max_seq_len=max_seq_len,
            max_response_len=max_response_len,
            max_knowledge_len=max_knowledge_len,
            return_length=True)
        response_start = tokenized_example['input_ids'].index(
            tokenizer.cls_token_id, 1)
        response_end = tokenized_example['seq_len']
        # Use to gather the logits corresponding to the labels during training
        tokenized_example['masked_positions'] = list(
            range(response_start, response_end - 1))
        tokenized_example['labels'] = tokenized_example['input_ids'][
            response_start + 1:response_end]
        return tokenized_example
    else:
        tokenized_example = tokenizer.dialogue_encode(
            example['history'],
            knowledge=goal_knowledge,
            task_type='knowledge',
            max_seq_len=max_seq_len,
            max_knowledge_len=max_knowledge_len,
            add_start_token_as_response=True)

        if 'response' in example:
            tokenized_example['response'] = example['response']
        return tokenized_example


def batchify_fn(batch_examples, pad_val, mode):
    def pad_mask(batch_attention_mask):
        batch_size = len(batch_attention_mask)
        max_len = max(map(len, batch_attention_mask))
        attention_mask = np.ones(
            (batch_size, max_len, max_len), dtype='float32') * -1e9
        for i, mask_data in enumerate(attention_mask):
            seq_len = len(batch_attention_mask[i])
            mask_data[-seq_len:, -seq_len:] = np.array(
                batch_attention_mask[i], dtype='float32')
        # In order to ensure the correct broadcasting mechanism, expand one 
        # dimension to the second dimension (n_head of Transformer).
        attention_mask = np.expand_dims(attention_mask, axis=1)
        return attention_mask

    pad_func = Pad(pad_val=pad_val, pad_right=False)

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
        labels = np.concatenate(
            [np.array(example['labels']) for example in batch_examples])
        return input_ids, token_type_ids, position_ids, attention_mask, masked_positions, labels
    else:
        return input_ids, token_type_ids, position_ids, attention_mask


def create_data_loader(dataset, tokenizer, args, mode):
    trans_func1 = partial(preprocess_examples, mode=mode)
    trans_func2 = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_response_len=args.max_response_len,
        max_knowledge_len=args.max_knowledge_len,
        mode=mode)
    dataset = dataset.map(trans_func1, batched=True).map(trans_func2, lazy=True)
    if mode == 'train':
        batch_sampler = DistributedBatchSampler(
            dataset, batch_size=args.batch_size, shuffle=True)
    else:
        batch_sampler = BatchSampler(
            dataset, batch_size=args.batch_size, shuffle=False)
    collate_fn = partial(batchify_fn, pad_val=tokenizer.pad_token_id, mode=mode)
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        return_list=True)
    return dataset, data_loader


def post_process_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return token_ids, tokens


def get_in_turn_repetition(pred, is_cn=False):
    """Get in-turn repetition."""
    if len(pred) == 0:
        return 1.0
    if isinstance(pred[0], str):
        pred = [tok.lower() for tok in pred]
        if is_cn:
            pred = "".join(pred)
    tri_grams = set()
    for i in range(len(pred) - 2):
        tri_gram = tuple(pred[i:i + 3])
        if tri_gram in tri_grams:
            return True
        tri_grams.add(tri_gram)
    return False


def select_response(ids, scores, tokenizer, max_dec_len=None, num_samples=1):
    ids = ids.numpy().tolist()
    scores = scores.numpy()

    if len(ids) != len(scores) or (len(ids) % num_samples) != 0:
        raise ValueError(
            "the length of `ids` is {}, but the `num_samples` is {}".format(
                len(ids), num_samples))

    group = []
    tmp = []
    for pred, score in zip(ids, scores):
        pred_token_ids, pred_tokens = post_process_response(pred, tokenizer)
        num_token = len(pred_token_ids)
        response = " ".join(pred_tokens)

        in_turn_repetition = get_in_turn_repetition(
            pred_tokens, True) or get_in_turn_repetition(pred_token_ids)
        # not ending
        if max_dec_len is not None and num_token >= max_dec_len:
            score -= 1e3
        elif in_turn_repetition:
            score -= 1e3

        tmp.append([response, score])
        if len(tmp) == num_samples:
            group.append(tmp)
            tmp = []

    results = []
    for preds in group:
        preds = sorted(preds, key=lambda x: -x[1])
        results.append(preds[0][0])
    return results
