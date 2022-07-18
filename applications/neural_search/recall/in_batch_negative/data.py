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

import os
import paddle
from paddlenlp.utils.log import logger


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)
    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    pad_to_max_seq_len=False):
    """
    Builds model inputs from a sequence.

    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """

    result = []
    for key, text in example.items():
        encoded_inputs = tokenizer(text=text,
                                   max_seq_len=max_seq_length,
                                   pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result


def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 2:
                continue
            yield {'text_a': data[0], 'text_b': data[1]}


def read_text_triplet(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 3:
                continue
            yield {
                'text': data[0],
                'pos_sample': data[1],
                'neg_sample': data[2]
            }


# ANN - active learning ------------------------------------------------------
def get_latest_checkpoint(args):
    """
        Return: (latest_checkpint_path, global_step)
    """
    if not os.path.exists(args.save_dir):
        return args.init_from_ckpt, 0

    subdirectories = list(next(os.walk(args.save_dir))[1])

    def valid_checkpoint(checkpoint):
        chk_path = os.path.join(args.save_dir, checkpoint)
        scheduler_path = os.path.join(chk_path, "model_state.pdparams")
        succeed_flag_file = os.path.join(chk_path, "succeed_flag_file")
        return os.path.exists(scheduler_path) and os.path.exists(
            succeed_flag_file)

    trained_steps = [int(s) for s in subdirectories if valid_checkpoint(s)]

    if len(trained_steps) > 0:
        return os.path.join(args.save_dir, str(max(trained_steps)),
                            "model_state.pdparams"), max(trained_steps)

    return args.init_from_ckpt, 0


# ANN - active learning ------------------------------------------------------
def get_latest_ann_data(ann_data_dir):
    if not os.path.exists(ann_data_dir):
        return None, -1

    subdirectories = list(next(os.walk(ann_data_dir))[1])

    def valid_checkpoint(step):
        ann_data_file = os.path.join(ann_data_dir, step, "new_ann_data")
        # succed_flag_file is an empty file that indicates ann data has been generated
        succeed_flag_file = os.path.join(ann_data_dir, step,
                                         "succeed_flag_file")
        return os.path.exists(succeed_flag_file) and os.path.exists(
            ann_data_file)

    ann_data_steps = [int(s) for s in subdirectories if valid_checkpoint(s)]

    if len(ann_data_steps) > 0:
        latest_ann_data_file = os.path.join(ann_data_dir,
                                            str(max(ann_data_steps)),
                                            "new_ann_data")
        logger.info(
            "Using lateset ann_data_file:{}".format(latest_ann_data_file))
        return latest_ann_data_file, max(ann_data_steps)

    logger.info("no new ann_data, return (None, -1)")
    return None, -1


def gen_id2corpus(corpus_file):
    id2corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            id2corpus[idx] = line.rstrip()
    return id2corpus


def gen_text_file(similar_text_pair_file):
    text2similar_text = {}
    texts = []
    with open(similar_text_pair_file, 'r', encoding='utf-8') as f:
        for line in f:
            splited_line = line.rstrip().split("\t")
            if len(splited_line) != 2:
                continue

            text, similar_text = line.rstrip().split("\t")

            if not text or not similar_text:
                continue

            text2similar_text[text] = similar_text
            texts.append({"text": text})
    return texts, text2similar_text
