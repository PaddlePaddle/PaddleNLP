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
from functools import partial

import paddle
from tqdm import tqdm

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
parser.add_argument("--ckpt_dir", type=str, default=None, help="The directory of saved model checkpoint.")
parser.add_argument(
    "--max_seq_len",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--device",
    choices=["cpu", "gpu", "xpu"],
    default="gpu",
    help="Select which device to train model, defaults to gpu.",
)
args = parser.parse_args()


@paddle.no_grad()
def predict(model, data_loader, label_map):
    """
    Given a prediction dataset, it gives the prediction results.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
    """
    model.eval()
    results = []
    for batch in tqdm(data_loader):
        input_ids, token_type_ids, seq_lens = batch["input_ids"], batch["token_type_ids"], batch["seq_lens"]
        preds = model(input_ids, token_type_ids, seq_lens=seq_lens)
        tags = parse_predict_result(preds.numpy(), seq_lens.numpy(), label_map)
        results.extend(tags)
    return results


def convert_example_to_feature(example, tokenizer, max_seq_len=512):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens.

    Args:
        example(obj:`dict`): Dict of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): The list of token_type_ids.
    """
    tokens = example["tokens"]
    new_tokens = [tokenizer.cls_token]

    for index, token in enumerate(tokens):
        sub_tokens = tokenizer.tokenize(token)
        if not sub_tokens:
            sub_tokens = [tokenizer.unk_token]
        new_tokens.extend(sub_tokens)

    new_tokens = new_tokens[: max_seq_len - 1]
    new_tokens.append(tokenizer.sep_token)

    input_ids = [tokenizer.convert_tokens_to_ids(token) for token in new_tokens]
    token_type_ids = [0] * len(input_ids)
    seq_len = len(input_ids)

    return {"input_ids": input_ids, "token_type_ids": token_type_ids, "seq_lens": seq_len}


def parse_predict_result(predictions, seq_lens, label_map):
    """
    Parses the prediction results to the label tag.
    """
    pred_tag = []
    for idx, pred in enumerate(predictions):
        seq_len = seq_lens[idx]
        # drop the "[CLS]" and "[SEP]" token
        tag = [label_map[i] for i in pred[1 : seq_len - 1]]
        pred_tag.append(tag)
    return pred_tag


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
    paddle.set_device(args.device)

    test_ds = load_dataset("cote", "dp", splits=["test"])
    label_list = test_ds.label_list
    # The COTE_DP dataset labels with "BIO" schema.
    label_map = {0: "B", 1: "I", 2: "O"}
    # `no_entity_label` represents that the token isn't an entity.
    no_entity_label_idx = 2

    tokenizer = SkepTokenizer.from_pretrained(args.model_name)
    model = SkepCrfForTokenClassification.from_pretrained(args.ckpt_dir, num_labels=len(label_list))
    print("Loaded model from %s" % args.ckpt_dir)

    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=no_entity_label_idx)

    test_data_loader = create_dataloader(
        test_ds, mode="test", batch_size=args.batch_size, batchify_fn=data_collator, trans_fn=trans_func
    )

    results = predict(model, test_data_loader, label_map)
    for idx, example in enumerate(test_ds.data):
        print(len(example["tokens"]), len(results[idx]))
        print("Data: {} \t Label: {}".format(example, results[idx]))
