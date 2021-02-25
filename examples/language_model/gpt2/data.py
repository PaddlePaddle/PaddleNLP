# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import numpy as np
import paddle


def construct_samples_and_shuffle_data(name, data_prefix, documents, sizes,
                                       num_samples, seq_length, seed,
                                       worker_index):
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'
    # Build the indexed mapping if not exist.
    if worker_index == 0:
        if (not os.path.isfile(doc_idx_filename)) or \
           (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):
            if num_epochs == 1:
                separate_last_epoch = False
            else:
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                last_epoch_num_samples = num_samples - \
                                         num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, \
                    'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                assert last_epoch_num_samples < (num_samples_per_epoch + 1), \
                    'last epoch number of samples exceeded max value.'
                separate_last_epoch = (
                    last_epoch_num_samples < int(0.80 * num_samples_per_epoch))
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            # sample-idx.
            assert doc_idx.dtype == np.int32
            sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
                                           num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
    else:
        while True:
            if (not os.path.isfile(doc_idx_filename)) or \
               (not os.path.isfile(sample_idx_filename)) or \
               (not os.path.isfile(shuffle_idx_filename)):
                time.sleep(3)
            else:
                break

    # Load mappings.
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    shuffle_idx = np.load(
        shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(lens):
    """Total number of tokens in the dataset."""
    return np.sum(lens)


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        # np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([int(num_samples) + 1, 2], dtype=np.int32)

    sample_index = 0
    doc_idx_index = 0
    doc_offset = 0
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            remaining_seq_length -= doc_length
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                doc_idx_index += 1
                doc_offset = 0
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(
        start=0, stop=num_samples, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(
        start=num_samples, stop=total_size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


class GPT2Dataset(paddle.io.Dataset):
    def __init__(self,
                 file_path,
                 worker_index,
                 num_samples,
                 eod_id,
                 name="gpt2",
                 max_seq_len=1024,
                 mode="train",
                 seed=1234):
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.name = name
        process_datas = np.load(
            self.file_path, mmap_mode="r+", allow_pickle=True)
        self.sample_ids = process_datas["ids"]
        self.sample_lens = process_datas["lens"]
        document_ids = np.arange(0, self.sample_lens.shape[0])
        self.eod_id = eod_id
        self.doc_idx, self.sample_idx, self.shuffle_idx = \
            construct_samples_and_shuffle_data(self.name, self.file_path, document_ids,\
                self.sample_lens, num_samples, max_seq_len, seed, worker_index)
        self.start_pos = [0] + np.cumsum(self.sample_lens).tolist()

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape(
            (1, seq_length, seq_length))

        # the pad and eod tokens do not contribute the loss
        loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[np.where(np.array(tokens) == self.eod_id)] = 0.0
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def _get_single_sample_from_idx(self, doc_index_f, doc_index_l, offset_f,
                                    offset_l):
        if doc_index_f == doc_index_l:
            current_start_pos = self.start_pos[doc_index_f]
            return self.sample_ids[current_start_pos+offset_f:\
                       current_start_pos+offset_l+1].tolist()
        elif doc_index_f < doc_index_l:
            current_start_pos = self.start_pos[doc_index_f]
            next_start_pos = self.start_pos[doc_index_f + 1]
            tokens = self.sample_ids[current_start_pos + offset_f:
                                     next_start_pos].tolist()
            for i in range(doc_index_f + 1, doc_index_l):
                current_start_pos = self.start_pos[i]
                next_start_pos = self.start_pos[i + 1]
                tokens.extend(self.sample_ids[current_start_pos:next_start_pos]
                              .tolist())
            last_start_pos = self.start_pos[doc_index_l]
            tokens.extend(self.sample_ids[last_start_pos:last_start_pos +
                                          offset_l + 1].tolist())
        else:
            current_start_pos = self.start_pos[doc_index_f]
            next_start_pos = self.start_pos[-1]
            tokens = self.sample_ids[current_start_pos + offset_f:
                                     next_start_pos].tolist()
            for i in range(0, doc_index_l):
                current_start_pos = self.start_pos[i]
                next_start_pos = self.start_pos[i + 1]
                tokens.extend(self.sample_ids[current_start_pos:next_start_pos]
                              .tolist())
            last_start_pos = self.start_pos[doc_index_l]
            tokens.extend(self.sample_ids[last_start_pos:last_start_pos +
                                          offset_l + 1].tolist())
        return tokens

    def __getitem__(self, index):
        idx = self.shuffle_idx[index]
        # Start and end documents and offsets.
        doc_index_f_raw = self.sample_idx[idx][0]
        doc_index_l_raw = self.sample_idx[idx + 1][0]
        doc_index_f = self.doc_idx[self.sample_idx[idx][0]]
        doc_index_l = self.doc_idx[self.sample_idx[idx + 1][0]]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        tokens = self._get_single_sample_from_idx(doc_index_f, doc_index_l,
                                                  offset_f, offset_l)
        return self._construct_sample(tokens)

    def __len__(self):
        return self.sample_idx.shape[0] - 1
