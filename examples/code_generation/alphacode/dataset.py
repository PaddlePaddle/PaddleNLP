# Copyright (c) 2020, NVIDIA CORPORATION.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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

import time
import os
import sys
import numpy as np
import paddle
from paddle.io import DataLoader, Dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger
from paddlenlp.utils.batch_sampler import DistributedBatchSampler

# Used to load data_tools path.
sys.path.insert(0, "../")


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(
            round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def create_pretrained_dataset(
        args,
        input_path,
        tokenizer,
        masked_lm_prob=0.15,
        encoder_max_seq_len=1536,
        decoder_max_seq_len=768, ):

    assert len(input_path) == 1, "GPT only support one dataset for now."

    input_prefix = input_path[0]

    if os.path.isfile(input_prefix + "_ids.npz"):
        logger.warning(
            "You are using compatible dataset, please make new dataset as the readme!"
        )
        process_datas = np.load(
            input_prefix + "_ids.npz", mmap_mode="r+", allow_pickle=True)
        sample_ids = process_datas["ids"]
        sample_lens = process_datas["lens"].astype("int32")
    else:
        for suffix in ["_ids.npy", "_idx.npz"]:
            if not os.path.isfile(input_prefix + suffix):
                raise ValueError("File Not found, %s" % (input_prefix + suffix))

        sample_ids = np.load(
            input_prefix + "_ids.npy", mmap_mode="r", allow_pickle=True)
        # All documment ids, extend as 1-D array.

        process_datas = np.load(input_prefix + "_idx.npz")
        # The len(sample_lens) num of docs
        # The sum(sample_lens) should equal len(sample_ids)
        sample_lens = process_datas["lens"]

    splits = get_train_valid_test_split_(args.split, len(sample_lens))
    assert len(sample_lens) >= splits[
        -1], "The document nums should larger than max of splits, but %s < %s" % (
            len(sample_lens), splits[-1])

    def build_dataset(index, name):
        dataset = Seq2SeqDataset(
            sample_ids=sample_ids,
            sample_lens=sample_lens,
            tokenizer=tokenizer,
            documents=np.arange(splits[index], splits[index + 1]),
            masked_lm_prob=masked_lm_prob,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
            seed=args.seed)
        batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=args.micro_batch_size,
            shuffle=False if name != 'train' else True,
            drop_last=True)

        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()),
            return_list=False)
        return data_loader

    # Note, data should be broardcast to all devices.
    train_data_loader = build_dataset(0, "train")
    valid_data_loader = build_dataset(1, "valid")
    test_data_loader = build_dataset(2, "test")

    return train_data_loader, valid_data_loader, test_data_loader


class Seq2SeqDataset(paddle.io.Dataset):
    def __init__(self,
                 sample_ids,
                 sample_lens,
                 tokenizer,
                 documents=None,
                 masked_lm_prob=0.15,
                 encoder_max_seq_len=1536,
                 decoder_max_seq_len=768,
                 seed=1234):
        self.encoder_max_seq_len = encoder_max_seq_len
        self.decoder_max_seq_len = decoder_max_seq_len
        self.sample_ids = sample_ids
        self.sample_lens = sample_lens
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob

        if documents is None:
            self.document_ids = np.arange(0, self.sample_lens.shape[0])
        else:
            self.document_ids = documents

        self.vocab_id_list = list(tokenizer.decoder.keys())
        self.vocab_id_to_token_dict = tokenizer.decoder
        self.vocab_token_to_id_dict = tokenizer.encoder
        # maybe choose another mask token
        self.mask_id = tokenizer.encoder['MASK']
        self.pad_id = tokenizer.pad_token_id
        self.cumsum_lens = [0] + np.cumsum(sample_lens).tolist()

    def _construct_sample(self, tokens, idx):
        origin_tokens = np.array(tokens).astype("int64").tolist()
        pivot = np.random.choice(range(len(origin_tokens)))

        encoder_tokens = origin_tokens[:pivot][:self.encoder_max_seq_len]
        decoder_tokens = origin_tokens[pivot:][:self.decoder_max_seq_len]

        labels = decoder_tokens[1:]
        tokens = decoder_tokens[:-1]

        pad_length = self.encoder_max_seq_len - len(tokens)
        pad_tokens = [self.pad_id] * pad_length
        tokens += pad_tokens
        labels += pad_tokens
        seq_length = len(tokens)

        # Attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape((1, seq_length,
                                                                 seq_length))

        # The pad and eos tokens do not contribute the loss
        loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[np.where(np.array(tokens) == self.pad_id)] = 0.0
        position_ids = list(range(len(tokens) - pad_length)) + pad_tokens
        position_ids = np.array(position_ids, dtype="int64")

        attention_mask = (attention_mask - 1.0) * 1e9
        attention_mask = attention_mask.astype("float32")
        labels = np.array(labels, dtype="int64")

        from data_tools.dataset_utils import create_masked_lm_predictions, pad_and_convert_to_numpy

        max_num_tokens = len(encoder_tokens)
        max_predictions_per_seq = self.masked_lm_prob * max_num_tokens
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))
        tokentypes = len(masked_tokens) * [0]
        (masked_tokens, masked_positions, masked_labels, _,
         _) = create_masked_lm_predictions(
             encoder_tokens,
             self.vocab_id_list,
             self.vocab_id_to_token_dict,
             self.masked_lm_prob,
             None,
             None,
             self.mask_id,
             max_predictions_per_seq,
             np_rng,
             vocab_token_to_id_dict=self.vocab_token_to_id_dict,
             to_chinese_char=False,
             inplace_random_mask=False)
        # Padding.
        mlm_tokens, mlm_tokentypes, mlm_labels, mlm_padding_mask, mlm_loss_mask \
            = pad_and_convert_to_numpy(masked_tokens, tokentypes, masked_positions,
                                       masked_labels, self.pad_id, self.encoder_max_seq_len)

        return {
            'encoder': [
                mlm_tokens, mlm_tokentypes, mlm_labels, mlm_padding_mask,
                mlm_loss_mask, masked_positions
            ],
            'decoder':
            [tokens, loss_mask, attention_mask, position_ids, labels]
        }

    def __getitem__(self, index):
        start_pos = self.cumsum_lens[index]
        if index < len(self.sample_ids) - 1:
            end_pos = self.cumsum_lens[index + 1]
            tokens = self.sample_ids[start_pos:end_pos]
        else:
            tokens = self.sample_ids[start_pos:]
        return self._construct_sample(tokens, index)

    def __len__(self):
        return len(self.document_ids)
