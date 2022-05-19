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

import os
import random
import argparse
from functools import partial
import multiprocessing
import numpy as np
from tqdm import tqdm

import paddle
from paddlenlp.transformers import (GPTLMHeadModel, GPTTokenizer)
from datasets import load_dataset, load_metric
from paddlenlp.data import Pad
from paddlenlp.datasets import MapDataset
from paddle.io import DataLoader, BatchSampler

MODEL_CLASSES = {"gpt2": (GPTLMHeadModel, GPTTokenizer)}


def tokenize_input(tokenizer, texts):
    input_ids = []
    max_len = 0
    for text in texts:
        ids = tokenizer(text)['input_ids']
        max_len = max(max_len, len(ids))
        input_ids.append(ids)
    for i in range(len(input_ids)):
        if len(input_ids[i]) < max_len:
            input_ids[i] += [tokenizer.pad_token_id] * (
                max_len - len(input_ids[i]))
    input_ids = paddle.to_tensor(input_ids, dtype="int32")
    return input_ids


def convert_example(example, tokenizer):
    """Convert all examples into necessary features."""
    tokenized = tokenizer(example, return_position_ids=True)
    input_ids = tokenized['input_ids']
    tokenized['attention_mask'] = np.triu(
        np.ones(
            (len(input_ids), len(input_ids)), dtype='float32') * -1e4, 1)
    return tokenized


def batchify_fn(examples, pad_val):
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

    pad_func = Pad(pad_val=pad_val, pad_right=False, dtype='int64')
    input_ids = pad_func([example['input_ids'] for example in examples])
    position_ids = pad_func([example['position_ids'] for example in examples])
    attention_mask = pad_mask(
        [example['attention_mask'] for example in examples])

    return input_ids, position_ids, attention_mask


def create_data_loader(dataset, tokenizer, args):
    trans_func = partial(convert_example, tokenizer=tokenizer)
    dataset = dataset.map(trans_func, lazy=True)
    batch_sampler = BatchSampler(
        dataset, batch_size=args.batch_size, shuffle=False)
    collate_fn = partial(batchify_fn, pad_val=tokenizer.pad_token_id)
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        return_list=True)
    return data_loader


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        '--model_type',
        default='gpt2-en',
        type=str,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument(
        '--model_name_or_path',
        default='gpt2-en',
        type=str,
        help="The path or shortcut name of the pre-trained model.")

    parser.add_argument(
        "--vocab_file",
        type=str,
        default='./data_tools/code-vocab.json',
        help="Path to the vocab file")
    parser.add_argument(
        "--merge_file",
        type=str,
        default='./data_tools/code-merges.txt',
        help="Path to the BPE merge file (if necessary).", )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size per GPU/CPU for predict.')
    parser.add_argument(
        "--end_token",
        default="\n",
        type=str,
        help="The end token. Defaults to \n. ")
    parser.add_argument(
        '--decode_strategy',
        type=str,
        default='sampling',
        help='The decode strategy in generation.')
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='The number of highest probability vocabulary tokens to keep for top-k sampling.'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='The value used to module the next token probabilities.')
    parser.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help='The cumulative probability for top-p sampling.')
    parser.add_argument(
        '--num_beams',
        type=int,
        default=0,
        help='The number of beams for beam search.')
    parser.add_argument(
        '--length_penalty',
        type=float,
        default=1.0,
        help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument(
        '--early_stopping',
        type=eval,
        default=False,
        help='Whether to stop the beam search when at least `num_beams` '
        'sentences are finished per batch or not.')
    parser.add_argument(
        '--min_dec_len',
        type=int,
        default=1,
        help='The minimum sequence length of generation.')
    parser.add_argument(
        '--max_dec_len',
        type=int,
        default=512,
        help='The maximum sequence length of generation.')
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='The number of output sequences to generation.')
    parser.add_argument(
        '--seed', type=int, default=123, help='Random seed for initialization.')
    parser.add_argument(
        '--device',
        type=str,
        default='gpu',
        help='The device to select for training the model.')

    args = parser.parse_args()
    return args


def print_args(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 or length > max_sequence_length:
        length = max_sequence_length
    return length


def postprocess_response(seq, eos_idx):
    """Post-process the decoded sequence."""
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [idx for idx in seq[:eos_pos + 1] if idx != eos_idx]
    return seq


def main(args):
    paddle.set_device(args.device)
    if args.seed:
        set_seed(args.seed)

    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError(
            "The `model_type` must be selected in the list: {}. But received: {}.".
            format(MODEL_CLASSES.keys(), args.model_type))

    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class(args.vocab_file, args.merge_file)
    model.eval()

    args.max_dec_len = adjust_length_to_model(args.max_dec_len,
                                              model.max_position_embeddings)

    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    code_eval_metric = load_metric("code_eval")
    human_eval = load_dataset("openai_humaneval")
    n_tasks = len(human_eval["test"])
    prompts = []
    for task in range(n_tasks):
        prompt = human_eval["test"][task]["prompt"].strip()
        prompts.extend([prompt] * args.num_samples)
    ds = MapDataset(prompts)
    data_loader = create_data_loader(ds, tokenizer, args)
    generated_sequences = []
    for batch in data_loader:
        input_ids, position_ids, attention_mask = batch
        ids, scores = model.generate(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_length=args.max_dec_len,
            min_length=args.min_dec_len,
            decode_strategy=args.decode_strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            early_stopping=args.early_stopping,
            eos_token_id=tokenizer.eos_token_id)
        for i, generated_ids in enumerate(ids):
            generated_ids = generated_ids.numpy().tolist()
            generated_ids = postprocess_response(generated_ids,
                                                 model.pad_token_id)
            # Decode text
            text = tokenizer.convert_ids_to_string(generated_ids)
            generated_sequences.append(text)

    generations, references = [], []
    for task in tqdm(range(n_tasks)):
        task_generations = []
        prompt = human_eval["test"][task]["prompt"]
        for sample in generated_sequences[task * \
            args.num_samples:(task + 1) * args.num_samples]:
            task_generations.append(prompt + sample)
        generations.append(task_generations)
        test_func = human_eval["test"][task]["test"]
        entry_point = f"check({human_eval['test'][task]['entry_point']})"
        references.append("\n" + test_func + "\n" + entry_point)

    # Evaluate completions with "code_eval" metric
    pass_at_k, _ = code_eval_metric.compute(
        references=references,
        predictions=generations,
        num_workers=multiprocessing.cpu_count() - 1,
        k=[1, 10, 100])
    print(f"Results: {pass_at_k}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
