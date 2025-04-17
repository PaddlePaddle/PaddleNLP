# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import random
import time
from functools import partial

import numpy as np
import paddle

from paddlenlp.data import DataCollatorForTokenClassification
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepCrfForTokenClassification, SkepTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    choices=["skep_ernie_1.0_large_ch", "skep_ernie_2.0_large_en"],
    default="skep_ernie_1.0_large_ch",
    help="Select which model to train, defaults to skep_ernie_1.0_large_ch.",
)
parser.add_argument(
    "--save_dir",
    default="./checkpoints",
    type=str,
    help="The output directory where the model checkpoints will be written.",
)
parser.add_argument(
    "--max_seq_len",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-7, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument(
    "--device",
    choices=["cpu", "gpu", "xpu"],
    default="gpu",
    help="Select which device to train model, defaults to gpu.",
)
args = parser.parse_args()


def set_seed(seed):
    """Sets random seed."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def convert_example_to_feature(example, tokenizer, max_seq_len=512, no_entity_label="O", is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens.

    Args:
        example(obj:`dict`): Dict of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        no_entity_label(obj:`int`): The label to pad label sequence by default.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): The list of token_type_ids.
        label(obj:`List[int]`, optional): The input label if not is_test.
    """
    tokens = example["tokens"]
    labels = example["labels"]
    assert len(tokens) == len(labels)

    # 1. tokenize the tokens into sub-tokens, and align the length of tokens and labels
    new_labels, new_tokens = [no_entity_label], [tokenizer.cls_token]
    for index, token in enumerate(tokens):
        sub_tokens = tokenizer.tokenize(token)
        if not sub_tokens:
            sub_tokens = [tokenizer.unk_token]

        # repeate the labels n-times
        new_labels.extend([labels[index]] * len(sub_tokens))
        new_tokens.extend(sub_tokens)

    # 2. check the max-length of tokens and labels
    new_tokens = new_tokens[: max_seq_len - 1]
    new_labels = new_labels[: max_seq_len - 1]

    # 3. construct the input data
    new_labels.append(no_entity_label)
    new_tokens.append(tokenizer.sep_token)
    input_ids = [tokenizer.convert_tokens_to_ids(token) for token in new_tokens]
    token_type_ids = [0] * len(input_ids)
    seq_len = len(input_ids)

    if is_test:
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "seq_lens": seq_len}
    else:
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "seq_lens": seq_len, "labels": new_labels}


def create_dataloader(dataset, mode="train", batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)


if __name__ == "__main__":
    set_seed(args.seed)

    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    train_ds = load_dataset("cote", "dp", splits=["train"])
    label_list = train_ds.label_list
    # The COTE_DP dataset labels with "BIO" schema.
    label_map = {label: idx for idx, label in enumerate(label_list)}
    # `no_entity_label` represents that the token isn't an entity.
    no_entity_label_idx = label_map.get("O", 2)

    tokenizer = SkepTokenizer.from_pretrained(args.model_name)
    model = SkepCrfForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))

    trans_func = partial(
        convert_example_to_feature,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        no_entity_label=no_entity_label_idx,
        is_test=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=no_entity_label_idx)

    train_data_loader = create_dataloader(
        train_ds, mode="train", batch_size=args.batch_size, batchify_fn=data_collator, trans_fn=trans_func
    )

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    global_step = 0
    tic_train = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            # print(batch)
            input_ids, token_type_ids, seq_lens, labels = (
                batch["input_ids"],
                batch["token_type_ids"],
                batch["seq_lens"],
                batch["labels"],
            )
            loss = model(input_ids, token_type_ids, seq_lens=seq_lens, labels=labels)
            avg_loss = paddle.mean(loss)
            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss, 10 / (time.time() - tic_train))
                )
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if global_step % 100 == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # Need better way to get inner model of DataParallel
                model._layers.save_pretrained(save_dir)
                print("Model saved to: {}.".format(save_dir))
