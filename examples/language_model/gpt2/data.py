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
from paddle.io import DataLoader, Dataset
from paddlenlp.data import Stack, Tuple, Pad


def construct_samples_and_shuffle_data(name, data_prefix, documents, sizes,
                                       num_samples, seq_length, seed,
                                       worker_index):
    """
    documents: document index from 0 to len(docs)
    sizes: the length list of all docs.
    num_samples: total step*bs iterations of data.
    seq_length: the sequence length.


    sum(sizes) = tokens_per_epoch
    data_nums = num_samples *  batch_size
    num_epochs = (data_nums + 1) // sum(sizes)
    len(doc_idx) = num_epochs * sum(sizes) 
    
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=1234)

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
            # len(doc_idx) = num_epochs * len(doc) 
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)

            # sample-idx. pos of each seq_len of data.
            assert doc_idx.dtype == np.int32
            sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
                                           num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)

            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1

            # shuffle all seq len data. 
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


def _num_tokens(documents, lens):
    """Total number of tokens in the dataset."""
    return np.sum(lens[documents])


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
    """
    Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document.
    """
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        # documents repeat num_epochs times.
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    """
    num_samples + 1, pos of bs data
    the distance between two points for sample idx is bs tokens.
    """
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


def create_pretrained_dataset(
        args,
        input_path,
        worker_init,
        worker_index,
        worker_num,
        eod_id,
        max_seq_len=1024,
        places=None,
        data_holders=None,
        pipeline_mode=False, ):
    print("the distributed run, worker_num:{}".format(worker_num))

    process_datas = np.load(input_path, mmap_mode="r+", allow_pickle=True)
    # all documment ids, extend as 1-D array.
    sample_ids = process_datas["ids"]
    # len(sample_lens) num of docs
    # sum(sample_lens) should equal len(sample_ids)
    sample_lens = process_datas["lens"]

    splits = [0, 237125, 249618, 249868]
    assert len(sample_lens) >= splits[
        -1], "The document nums should larger than max of splits, but %s < %s" % (
            len(sample_lens), splits[-1])

    def build_dataset(index, name, num_samples):
        dataset = GPT2Dataset(
            file_path=input_path,
            worker_index=worker_index,
            name="gpt2" + name,
            max_seq_len=max_seq_len,
            num_samples=num_samples,
            documents=np.arange(splits[index], splits[index + 1]),
            sample_ids=sample_ids,
            sample_lens=sample_lens,
            eod_id=eod_id,
            seed=args.seed + worker_index)

        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

        if pipeline_mode:

            def data_gen():
                for data in dataset:
                    yield tuple(
                        [np.expand_dims(
                            np.array(x), axis=0) for x in data])

            data_loader = paddle.fluid.io.DataLoader.from_generator(
                feed_list=data_holders, capacity=70, iterable=False)
            data_loader.set_batch_generator(data_gen, places)
        else:
            data_loader = DataLoader(
                dataset=dataset,
                places=places,
                feed_list=data_holders,
                batch_sampler=batch_sampler,
                num_workers=0,
                worker_init_fn=worker_init,
                collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()),
                return_list=False)
        return data_loader

    train_data_loader = build_dataset(0, "train", args.batch_size *
                                      args.max_steps * worker_num)
    if pipeline_mode:
        valid_data_loader, test_data_loader = None, None
    else:
        valid_data_loader = build_dataset(
            1, "valid", args.batch_size *
            (args.max_steps // args.eval_steps + 1) * args.eval_iters)
        test_data_loader = build_dataset(2, "test",
                                         args.batch_size * args.test_iters)

    return train_data_loader, valid_data_loader, test_data_loader


class GPT2Dataset(paddle.io.Dataset):
    def __init__(self,
                 file_path,
                 worker_index,
                 num_samples,
                 eod_id,
                 sample_ids,
                 sample_lens,
                 documents=None,
                 name="gpt2",
                 max_seq_len=1024,
                 mode="train",
                 seed=1234):
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.name = name
        self.eod_id = eod_id
        self.sample_ids = sample_ids
        self.sample_lens = sample_lens

        if documents is None:
            document_ids = np.arange(0, self.sample_lens.shape[0])
        else:
            document_ids = documents

        self.doc_idx, self.sample_idx, self.shuffle_idx = \
            construct_samples_and_shuffle_data(self.name, self.file_path, document_ids,\
                self.sample_lens, num_samples, max_seq_len, seed, worker_index)
        # doc cumsum start pos
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
        # attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def _get_single_sample_from_idx(self, doc_index_f, doc_index_l, offset_f,
                                    offset_l):
        """
        doc_index_f: data from the first doc.
        doc_index_l: data from the last doc.
        offset_f: offset of the first doc.
        offset_l: offset of the last doc.
        """
        # data from the sample doc. just select the needed ids.
        if doc_index_f == doc_index_l:
            current_start_pos = self.start_pos[self.doc_idx[doc_index_f]]
            return self.sample_ids[current_start_pos+offset_f:\
                       current_start_pos+offset_l+1].tolist()

        # data from multi docs.
        else:
            current_start_pos = self.start_pos[self.doc_idx[doc_index_f]]
            next_start_pos = self.start_pos[self.doc_idx[doc_index_f] + 1]
            tokens = self.sample_ids[current_start_pos + offset_f:
                                     next_start_pos].tolist()
            for i in range(doc_index_f + 1, doc_index_l):
                current_start_pos = self.start_pos[self.doc_idx[i]]
                next_start_pos = self.start_pos[self.doc_idx[i] + 1]
                tokens.extend(self.sample_ids[current_start_pos:next_start_pos]
                              .tolist())
            last_start_pos = self.start_pos[self.doc_idx[doc_index_l]]
            tokens.extend(self.sample_ids[last_start_pos:last_start_pos +
                                          offset_l + 1].tolist())

        return tokens

    def __getitem__(self, index):
        idx = self.shuffle_idx[index]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        tokens = self._get_single_sample_from_idx(doc_index_f, doc_index_l,
                                                  offset_f, offset_l)
        return self._construct_sample(tokens)

    def __len__(self):
        return self.sample_idx.shape[0] - 1
