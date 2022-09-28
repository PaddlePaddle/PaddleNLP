# Copyright (c) 2020, NVIDIA CORPORATION.
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
import sys

import numpy as np
import paddle
from paddle.io import DataLoader, Dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger
from paddlenlp.utils.batch_sampler import DistributedBatchSampler

# Used to load data_tools path.
sys.path.insert(0, "../")


def construct_samples_and_shuffle_data(name, data_prefix, documents, sizes,
                                       num_samples, seq_length, seed,
                                       build_data_file):
    """
    documents: document index from 0 to len(docs)
    sizes: the length list of all docs.
    num_samples: total step*bs iterations of data.
    seq_length: the sequence length.


    sum(sizes) = tokens_per_epoch
    data_nums = num_samples *  micro_batch_size
    num_epochs = (data_nums + 1) // sum(sizes)
    len(doc_idx) = num_epochs * sum(sizes)

    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # Rng state
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
    if build_data_file:
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
                separate_last_epoch = (last_epoch_num_samples < int(
                    0.80 * num_samples_per_epoch))
            # Note. len(doc_idx) = num_epochs * len(doc)
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print(' > elasped time to build and save doc-idx mapping '
                  '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx. pos of each seq_len of data.
            start_time = time.time()
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32

            import data_tools.helpers as helpers

            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print(' > elasped time to build and save sample-idx mapping '
                  '(seconds): {:4f}'.format(time.time() - start_time))

            # shuffle-idx.
            start_time = time.time()

            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1

            # Shuffle all seq len data.
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print(' > elasped time to build and save shuffle-idx mapping'
                  ' (seconds): {:4f}'.format(time.time() - start_time))

    else:
        while True:
            if (not os.path.isfile(doc_idx_filename)) or \
               (not os.path.isfile(sample_idx_filename)) or \
               (not os.path.isfile(shuffle_idx_filename)):
                time.sleep(3)
            else:
                try:
                    np.load(shuffle_idx_filename,
                            allow_pickle=True,
                            mmap_mode='r')
                    break
                except Exception as e:
                    print(
                        "%s file is still writing or damaged, please wait a moment."
                        % shuffle_idx_filename)
                    time.sleep(3)

    if paddle.distributed.get_world_size() > 1:
        if paddle.in_dynamic_mode():
            paddle.distributed.barrier()

    # Load mappings.
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    shuffle_idx = np.load(shuffle_idx_filename,
                          allow_pickle=True,
                          mmap_mode='r')
    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, lens):
    """Total number of tokens in the dataset."""
    return np.sum(lens[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence length, calculate how many
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
        # The documents repeat num_epochs times.
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

    shuffle_idx_first = np.arange(start=0,
                                  stop=num_samples,
                                  step=1,
                                  dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples,
                                 stop=total_size,
                                 step=1,
                                 dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


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
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def create_pretrained_dataset(args,
                              input_path,
                              local_rank,
                              data_world_rank,
                              data_world_size,
                              eos_id,
                              worker_init=None,
                              max_seq_len=1024,
                              places=None,
                              data_holders=None,
                              pipeline_mode=False):

    if local_rank == 0:
        try:
            import data_tools.helpers as helpers
        except Exception as e:
            start_time = time.time()
            print('> compiling dataset index builder ...')
            from data_tools.dataset_utils import compile_helper
            compile_helper()
            print(
                '>>> done with dataset index builder. Compilation time: {:.3f} '
                'seconds'.format(time.time() - start_time),
                flush=True)

    device_world_size = paddle.distributed.get_world_size()
    device_world_rank = paddle.distributed.get_rank()

    if device_world_size > 1 and local_rank != 0:
        while True:
            try:
                import data_tools.helpers as helpers
                break
            except Exception as e:
                print("> wait for helpers to be compiled!")
                time.sleep(1)

    logger.info(
        "The distributed run, total device num:{}, distinct dataflow num:{}.".
        format(device_world_size, data_world_size))

    assert len(input_path) == 1, "GPT only support one dataset for now."

    input_prefix = input_path[0]

    if os.path.isfile(input_prefix + "_ids.npz"):
        logger.warning(
            "You are using compatible dataset, please make new dataset as the readme!"
        )
        process_data = np.load(input_prefix + "_ids.npz",
                               mmap_mode="r+",
                               allow_pickle=True)
        sample_ids = process_data["ids"]
        sample_lens = process_data["lens"].astype("int32")
    else:
        for suffix in ["_ids.npy", "_idx.npz"]:
            if not os.path.isfile(input_prefix + suffix):
                raise ValueError("File Not found, %s" % (path + suffix))

        sample_ids = np.load(input_prefix + "_ids.npy",
                             mmap_mode="r",
                             allow_pickle=True)
        # All documment ids, extend as 1-D array.

        process_data = np.load(input_prefix + "_idx.npz")
        # The len(sample_lens) num of docs
        # The sum(sample_lens) should equal len(sample_ids)
        sample_lens = process_data["lens"]

    splits = get_train_valid_test_split_(args.split, len(sample_lens))
    assert len(sample_lens) >= splits[
        -1], "The document nums should larger than max of splits, but %s < %s" % (
            len(sample_lens), splits[-1])

    def build_dataset(index, name, num_samples):
        dataset = GPTDataset(file_prefix=input_prefix,
                             build_data_file=local_rank == 0,
                             micro_batch_size=args.micro_batch_size,
                             name="gpt_" + name,
                             max_seq_len=max_seq_len,
                             num_samples=num_samples,
                             documents=np.arange(splits[index],
                                                 splits[index + 1]),
                             sample_ids=sample_ids,
                             sample_lens=sample_lens,
                             eos_id=eos_id,
                             seed=args.seed,
                             use_pure_fp16=args.use_amp
                             and args.amp_level == "O2",
                             data_holders=data_holders)
        batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=args.micro_batch_size,
            num_replicas=data_world_size,
            rank=data_world_rank,
            shuffle=False,
            drop_last=True)

        if pipeline_mode:

            def data_gen():
                for data in dataset:
                    yield tuple(
                        [np.expand_dims(np.array(x), axis=0) for x in data])

            data_loader = paddle.io.DataLoader.from_generator(
                feed_list=data_holders, capacity=70, iterable=False)
            data_loader.set_sample_generator(data_gen,
                                             batch_size=args.micro_batch_size,
                                             places=places)
        else:
            stacks = (Stack(), ) * len(data_holders)
            collate_fn = Tuple(*stacks)
            data_loader = DataLoader(dataset=dataset,
                                     places=places,
                                     feed_list=data_holders,
                                     batch_sampler=batch_sampler,
                                     num_workers=1,
                                     worker_init_fn=worker_init,
                                     collate_fn=collate_fn,
                                     return_list=False)
        return data_loader

    # Note, data should be broardcast to all devices.
    # for train, valid, test, the distinct data num is data_world_size
    train_data_loader = build_dataset(
        0, "train", args.micro_batch_size * args.max_steps * data_world_size)
    if pipeline_mode:
        valid_data_loader, test_data_loader = None, None
    else:
        valid_data_loader = build_dataset(
            1, "valid",
            args.micro_batch_size * (args.max_steps // args.eval_freq + 1) *
            args.eval_iters * data_world_size)
        test_data_loader = build_dataset(
            2, "test",
            args.micro_batch_size * args.test_iters * data_world_size)

    return train_data_loader, valid_data_loader, test_data_loader


class GPTDataset(paddle.io.Dataset):

    def __init__(self,
                 file_prefix,
                 micro_batch_size,
                 num_samples,
                 eos_id,
                 sample_ids,
                 sample_lens,
                 documents=None,
                 build_data_file=False,
                 name="gpt",
                 max_seq_len=1024,
                 seed=1234,
                 use_pure_fp16=False,
                 data_holders=None):
        self.file_prefix = file_prefix
        self.max_seq_len = max_seq_len
        self.name = name
        self.eos_id = eos_id
        self.sample_ids = sample_ids
        self.sample_lens = sample_lens
        self.micro_batch_size = micro_batch_size
        self.use_pure_fp16 = use_pure_fp16
        self.data_holders = data_holders

        if documents is None:
            document_ids = np.arange(0, self.sample_lens.shape[0])
        else:
            document_ids = documents

        self.doc_idx, self.sample_idx, self.shuffle_idx = \
            construct_samples_and_shuffle_data(self.name, self.file_prefix, document_ids,\
                self.sample_lens, num_samples, max_seq_len, seed, build_data_file)

        # The doc cumsum start pos
        self.start_pos = [0] + np.cumsum(self.sample_lens).tolist()

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)

        # The pad and eos tokens do not contribute the loss
        if self.use_pure_fp16:
            loss_mask = np.ones(seq_length, dtype="float16")
        else:
            loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[np.where(np.array(tokens) == self.eos_id)] = 0.0

        position_ids = np.arange(0, seq_length, dtype="int64")
        labels = np.array(labels, dtype="int64")
        if len(self.data_holders) == 4:
            return [tokens, loss_mask, position_ids, labels]
        elif len(self.data_holders) == 3:
            return [tokens, loss_mask, position_ids]
        else:
            assert len(self.data_holders) == 1, \
                "length of daat_holders should be 4, 3 or 1"
            return [tokens]

    def _get_single_sample_from_idx(self, doc_index_f, doc_index_l, offset_f,
                                    offset_l):
        """
        The input means:
            doc_index_f: data from the first doc.
            doc_index_l: data from the last doc.
            offset_f: offset of the first doc.
            offset_l: offset of the last doc.
        """
        # Data from the sample doc. just select the needed ids.
        if doc_index_f == doc_index_l:
            current_start_pos = self.start_pos[self.doc_idx[doc_index_f]]
            return self.sample_ids[current_start_pos+offset_f:\
                       current_start_pos+offset_l+1].tolist()

        # Data from multi docs.
        else:
            current_start_pos = self.start_pos[self.doc_idx[doc_index_f]]
            next_start_pos = self.start_pos[self.doc_idx[doc_index_f] + 1]
            tokens = self.sample_ids[current_start_pos +
                                     offset_f:next_start_pos].tolist()
            for i in range(doc_index_f + 1, doc_index_l):
                current_start_pos = self.start_pos[self.doc_idx[i]]
                next_start_pos = self.start_pos[self.doc_idx[i] + 1]
                tokens.extend(
                    self.sample_ids[current_start_pos:next_start_pos].tolist())
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
