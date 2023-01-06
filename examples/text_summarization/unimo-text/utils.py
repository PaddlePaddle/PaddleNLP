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

import random
from functools import partial

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from rouge import Rouge

from paddlenlp.data import Pad
from paddlenlp.metrics import BLEU


def print_args(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better.
    paddle.seed(seed + dist.get_rank())


def compute_metrics(preds, targets):
    assert len(preds) == len(targets), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(preds), len(targets))
    )
    rouge = Rouge()
    bleu4 = BLEU(n_size=4)
    scores = []
    for pred, target in zip(preds, targets):
        try:
            score = rouge.get_scores(" ".join(pred), " ".join(target))
            scores.append([score[0]["rouge-1"]["f"], score[0]["rouge-2"]["f"], score[0]["rouge-l"]["f"]])
        except ValueError:
            scores.append([0, 0, 0])
        bleu4.add_inst(pred, [target])
    rouge1 = np.mean([i[0] for i in scores])
    rouge2 = np.mean([i[1] for i in scores])
    rougel = np.mean([i[2] for i in scores])
    print("\n" + "*" * 15)
    print("The auto evaluation result is:")
    print("rouge-1:", round(rouge1, 4))
    print("rouge-2:", round(rouge2, 4))
    print("rouge-L:", round(rougel, 4))
    print("BLEU-4:", round(bleu4.score(), 4))


def convert_example(example, tokenizer, max_seq_len=512, max_target_len=128, mode="train"):
    """Convert all examples into necessary features."""
    source = example["content"]
    if mode != "test":
        tokenized_example = tokenizer.gen_encode(
            source,
            target=example["title"],
            max_seq_len=max_seq_len,
            max_target_len=max_target_len,
            return_position_ids=True,
            return_length=True,
        )
        target_start = tokenized_example["input_ids"].index(tokenizer.cls_token_id, 1)
        target_end = tokenized_example["seq_len"]
        # Use to gather the logits corresponding to the labels during training
        tokenized_example["masked_positions"] = list(range(target_start, target_end - 1))
        tokenized_example["labels"] = tokenized_example["input_ids"][target_start + 1 : target_end]

        return tokenized_example
    else:
        tokenized_example = tokenizer.gen_encode(
            source, max_seq_len=max_seq_len, add_start_token_for_decoding=True, return_position_ids=True
        )

        if "title" in example and example["title"]:
            tokenized_example["title"] = example["title"]
        return tokenized_example


def batchify_fn(batch_examples, pad_val, mode):
    def pad_mask(batch_attention_mask):
        batch_size = len(batch_attention_mask)
        max_len = max(map(len, batch_attention_mask))
        attention_mask = np.ones((batch_size, max_len, max_len), dtype="float32") * -1e9
        for i, mask_data in enumerate(attention_mask):
            seq_len = len(batch_attention_mask[i])
            mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i], dtype="float32")
        # In order to ensure the correct broadcasting mechanism, expand one
        # dimension to the second dimension (n_head of Transformer).
        attention_mask = np.expand_dims(attention_mask, axis=1)
        return attention_mask

    pad_func = Pad(pad_val=pad_val, pad_right=False, dtype="int64")

    input_ids = pad_func([example["input_ids"] for example in batch_examples])
    token_type_ids = pad_func([example["token_type_ids"] for example in batch_examples])
    position_ids = pad_func([example["position_ids"] for example in batch_examples])

    attention_mask = pad_mask([example["attention_mask"] for example in batch_examples])

    if mode != "test":
        max_len = max([example["seq_len"] for example in batch_examples])
        masked_positions = np.concatenate(
            [
                np.array(example["masked_positions"]) + (max_len - example["seq_len"]) + i * max_len
                for i, example in enumerate(batch_examples)
            ]
        )
        labels = np.concatenate([np.array(example["labels"], dtype="int64") for example in batch_examples])
        return input_ids, token_type_ids, position_ids, attention_mask, masked_positions, labels
    else:
        return input_ids, token_type_ids, position_ids, attention_mask


def create_data_loader(dataset, tokenizer, args, mode):
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_target_len=args.max_target_len,
        mode=mode,
    )
    dataset = dataset.map(trans_func, lazy=True)
    if mode == "train":
        batch_sampler = DistributedBatchSampler(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        batch_sampler = BatchSampler(dataset, batch_size=args.batch_size // 2, shuffle=False)
    collate_fn = partial(batchify_fn, pad_val=tokenizer.pad_token_id, mode=mode)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, return_list=True)
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
    special_tokens = ["[UNK]"]
    tokens = [token for token in tokens if token not in special_tokens]
    return token_ids, tokens


def select_sum(ids, scores, tokenizer, max_dec_len=None, num_return_sequences=1):
    results = []
    group = []
    tmp = []
    if scores is not None:
        ids = ids.numpy()
        scores = scores.numpy()

        if len(ids) != len(scores) or (len(ids) % num_return_sequences) != 0:
            raise ValueError(
                "the length of `ids` is {}, but the `num_return_sequences` is {}".format(
                    len(ids), num_return_sequences
                )
            )

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
