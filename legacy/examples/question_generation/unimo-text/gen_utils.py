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

from paddlenlp.data import Pad


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


def convert_example(
    example, tokenizer, max_seq_len=512, max_target_len=128, max_title_len=256, mode="train", template=0
):
    """Convert all examples into necessary features."""
    if mode == "pretrain" or mode == "pretrain_test":
        context = example["context"]
        answer = example["answer"]
        target = example["target"]
        source = "答案：" + answer + tokenizer.sep_token + "上下文：" + context
        title = None

    elif mode == "train" or mode == "test":
        target = None
        title = None
        if "source" in example and "title" in example:
            source = example["source"]
            if "title" in example.keys():
                title = example["title"]
        elif "context" in example and "answer" in example:
            source = example["context"]
            if "answer" in example.keys():
                title = example["answer"]
        else:
            assert False, "Source and title are not in the input dictionary, nor are context and answer."
        if "target" in example.keys():
            target = example["target"]
        elif "question" in example.keys():
            target = example["question"]

        if template == 1:
            source = "答案：" + title + tokenizer.sep_token + "上下文：" + source
            title = None
            if target:
                target = "问题：" + target
        elif template == 2:
            source = "答案：" + title + tokenizer.sep_token + "上下文：" + source
            title = None
            if target:
                target = "在已知答案的前提下，问题：" + target
        elif template == 3:
            source = "这是一个问题生成任务，根据提供的答案和上下文，来生成问题。" + title + tokenizer.sep_token + "上下文：" + source
            title = None
            if target:
                target = "问题：" + target
        elif template == 4:
            prompt_common = example["prompt_common"]
            prompt_domain = example["prompt_domain"]
            source = (
                prompt_common
                + " "
                + tokenizer.sep_token
                + " "
                + "".join(
                    [" " + tokenizer.cls_token + " " + one + " " + tokenizer.sep_token + " " for one in prompt_domain]
                )
                + " "
                + tokenizer.cls_token
                + " "
                + "答案："
                + title
                + " "
                + tokenizer.sep_token
                + " "
                + tokenizer.cls_token
                + "上下文："
                + source
            )

            title = None
            if target:
                target = "问题：" + target

    if mode == "train" or mode == "pretrain":
        tokenized_example = tokenizer.gen_encode(
            source,
            title=title,
            target=target,
            max_seq_len=max_seq_len,
            max_target_len=max_target_len,
            max_title_len=max_title_len,
            return_position_ids=True,
            return_length=True,
        )
        temp_tokens = tokenizer.convert_ids_to_tokens(tokenized_example["input_ids"])
        index_list = []
        count = tokenized_example["input_ids"].count(tokenizer.cls_token_id)
        # If template==4, count must be equal to 7, otherwise count must be equal to 2
        assert count == 7 or count == 2, (
            str(count) + " is not in [2, 7], temp_tokens: " + " ".join(temp_tokens) + "source: " + source
        )
        index = -1
        for i in range(0, count):
            index = tokenized_example["input_ids"].index(tokenizer.cls_token_id, index + 1)
            index_list.append(index)
        if template == 4:
            tokenized_example["token_type_ids"] = (
                [2] * (index_list[1] - index_list[0])
                + [3] * (index_list[4] - index_list[1])
                + [0] * (index_list[6] - index_list[4])
                + [1] * (len(tokenized_example["input_ids"]) - index_list[6])
            )
        target_start = index_list[-1]
        target_end = tokenized_example["seq_len"]
        # Use to gather the logits corresponding to the labels during training
        tokenized_example["masked_positions"] = list(range(target_start, target_end - 1))
        tokenized_example["labels"] = tokenized_example["input_ids"][target_start + 1 : target_end]
        if template == 4:
            tokenized_example["token_type_ids"]
        return tokenized_example

    elif mode == "test" or mode == "pretrain_test":
        tokenized_example = tokenizer.gen_encode(
            source,
            title=title,
            max_seq_len=max_seq_len,
            max_title_len=max_title_len,
            add_start_token_for_decoding=True,
            return_position_ids=True,
        )

        if template == 4:
            # temp_tokens = tokenizer.convert_ids_to_tokens(tokenized_example['input_ids'])
            index_list = []
            count = tokenized_example["input_ids"].count(tokenizer.cls_token_id)
            assert count == 7, str(count) + " is not in [7]"
            index = -1
            for i in range(0, count):
                index = tokenized_example["input_ids"].index(tokenizer.cls_token_id, index + 1)
                index_list.append(index)
            tokenized_example["token_type_ids"] = (
                [2] * (index_list[1] - index_list[0])
                + [3] * (index_list[4] - index_list[1])
                + [0] * (index_list[6] - index_list[4])
                + [1] * (len(tokenized_example["input_ids"]) - index_list[6])
            )

        if "target" in example and example["target"]:
            tokenized_example["target"] = example["target"]
        elif "question" in example and example["question"]:
            tokenized_example["target"] = example["question"]
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

    if mode == "train" or mode == "pretrain":
        max_len = max([example["seq_len"] for example in batch_examples])
        masked_positions = np.concatenate(
            [
                np.array(example["masked_positions"]) + (max_len - example["seq_len"]) + i * max_len
                for i, example in enumerate(batch_examples)
            ]
        )
        labels = np.concatenate([np.array(example["labels"], dtype="int64") for example in batch_examples])
        return input_ids, token_type_ids, position_ids, attention_mask, masked_positions, labels
    elif mode == "test" or mode == "pretrain_test":
        return input_ids, token_type_ids, position_ids, attention_mask


def create_data_loader(dataset, tokenizer, args, mode):
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_target_len=args.max_target_len,
        max_title_len=args.max_title_len,
        mode=mode,
        template=args.template,
    )
    dataset = dataset.map(trans_func, lazy=True)
    if mode == "pretrain":
        batch_sampler = DistributedBatchSampler(dataset, batch_size=args.batch_size, shuffle=True)
    elif mode == "train":
        batch_sampler = DistributedBatchSampler(dataset, batch_size=args.batch_size, shuffle=True)
    elif mode == "test" or mode == "pretrain_test":
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


def remove_template(instr):
    """Remove template prefix of decoded sequence."""
    outstr = instr.strip("问题：")
    outstr = instr.strip("在已知答案的前提下，问题：")
    return outstr


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
            target = remove_template(target)

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
            response = remove_template(response)

            # TODO: Support return scores in FT.
            tmp.append([response])
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []

        for preds in group:
            results.append(preds[0][0])

    return results
