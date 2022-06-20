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

import logging
import multiprocessing
import threading
from queue import Queue

import h5py
import numpy as np
import paddle

KEYS = ('input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
        'masked_lm_ids', 'next_sentence_labels')


def shuffle_dict(dic, len):
    idxs = np.arange(len)
    np.random.shuffle(idxs)
    for k, v in dic.items():
        dic[k] = v[idxs]


class PretrainingHDF5DataLoader:

    def __init__(self,
                 input_files,
                 max_seq_length=128,
                 max_mask_tokens=20,
                 batch_size=1,
                 dtype=np.int32,
                 shuffle=False,
                 pad_position_value=511,
                 num_workers=3):
        self.files = input_files
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_mask_tokens = max_mask_tokens
        self.dtype = dtype
        self.shuffle = shuffle
        self.pad_position_value = pad_position_value
        if shuffle:
            np.random.shuffle(self.files)

        self.counter = 0

        # get total number of samples
        pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), 32))
        num_samples = pool.map(self.samples_in_file, self.files)
        pool.close()
        pool.join()
        self.total_samples = sum(num_samples)
        self.len = self.total_samples // self.batch_size
        assert self.len > 1, f"Batch size {self.batch_size} larger than number of samples {self.total_samples}"

        # notify feed and fetch processes/thread to stop
        self.event_queue = multiprocessing.Manager().Queue(10)

        # buffer to store final data
        self.feed_buffer = Queue(20)

        # number of processes to do remask
        self.num_workers = num_workers
        # each feed_worker has one process_buffer to use
        self.process_buffers = [
            multiprocessing.Manager().Queue(10) for _ in range(num_workers)
        ]
        self.split_files = np.array_split(self.files, self.num_workers)
        # feed_worker will load data from h5py files, and do remask process
        self.feed_workers = [
            multiprocessing.Process(target=self.fill_buffer_loop,
                                    args=(self.split_files[idx],
                                          self.process_buffers[idx]))
            for idx in range(self.num_workers)
        ]
        for p in self.feed_workers:
            p.start()

        # index for which process_buffer is used each time
        self.post_fetch_idx = 0
        # load final data from process_buffers
        self.fetch_worker = threading.Thread(target=self.post_fetch)
        self.fetch_worker.start()

    def samples_in_file(self, filename):
        with h5py.File(filename, "r") as f:
            data_len = f[KEYS[0]].shape[0]
        return data_len

    def release(self):
        self.event_queue.put('END')
        while not self.feed_buffer.empty():
            self.feed_buffer.get()
        for process_buffer in self.process_buffers:
            while not process_buffer.empty():
                process_buffer.get()
        self.fetch_worker.join()
        for p in self.feed_workers:
            p.join()
        return

    def __len__(self):
        return self.len

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        result = self.feed_buffer.get()
        self.counter += 1
        return result

    def post_fetch(self):
        while True:
            if not self.event_queue.empty():
                return
            if not self.process_buffers[self.post_fetch_idx].empty():
                logging.debug(f"self.post_fetch_idx: {self.post_fetch_idx}")
                np_feed_list = self.process_buffers[self.post_fetch_idx].get()
                self.post_fetch_idx += 1
                if self.post_fetch_idx == self.num_workers:
                    self.post_fetch_idx = 0
                elif self.post_fetch_idx > self.num_workers:
                    raise Exception('post_fetch_idx must < num_workers')

                lod_feed_list = []
                for data in np_feed_list:
                    tensor = paddle.fluid.core.LoDTensor()
                    place = paddle.CPUPlace()
                    tensor.set(data, place)
                    lod_feed_list.append(tensor)
                self.feed_buffer.put(lod_feed_list)

    def fill_buffer_loop(self, files, process_buffer):
        data = None
        data_index = 0
        file_index = 0

        def multiprocess_fill_buffer(data, file_index, data_index):
            if data is None:
                data = self.load_one_file(files[file_index])
                file_index += 1
                data_index = 0

            curr_batch = []
            still_required = self.batch_size
            while still_required > 0:
                data_batch = {
                    k: data[k][data_index:data_index + still_required]
                    for k in KEYS
                }
                data_batch_len = len(data_batch[KEYS[0]])
                data_index += data_batch_len
                curr_batch.append(data_batch)
                curr_batch_len = sum(len(x[KEYS[0]]) for x in curr_batch)
                still_required = self.batch_size - curr_batch_len
                if still_required > 0:
                    if file_index >= len(files):
                        np.random.shuffle(files)
                        file_index = 0

                    data = self.load_one_file(files[file_index])
                    file_index += 1
                    data_index = 0
            if not curr_batch_len == self.batch_size:
                raise Exception("data length should equal to batch_size")

            result = {}
            for k in KEYS:
                result[k] = np.concatenate([item[k] for item in curr_batch],
                                           axis=0)
            process_buffer.put(self.do_remask(result))

            return data, file_index, data_index

        while True:
            if self.event_queue.empty():
                data, file_index, data_index = multiprocess_fill_buffer(
                    data, file_index, data_index)
            else:
                return

    def do_remask(self, samples):
        input_ids = samples['input_ids']
        segment_ids = samples['segment_ids']
        masked_lm_positions = samples['masked_lm_positions']
        masked_lm_ids = samples['masked_lm_ids']
        next_sentence_labels = samples['next_sentence_labels']
        masked_lm_weights = np.ones_like(masked_lm_ids, dtype=np.int32)
        masked_lm_weights[masked_lm_ids == 0] = 0

        # post process
        batch_size, seq_len = input_ids.shape
        formatted_pos = self.pad_position_value * np.ones_like(
            samples['input_ids'])
        formatted_input = np.zeros_like(input_ids)
        formatted_seg = np.zeros_like(segment_ids)
        formatted_mask_labels = np.zeros((batch_size, self.max_mask_tokens),
                                         dtype=masked_lm_ids.dtype)

        valid_seq_positions = []
        valid_mask_positions = masked_lm_weights == 1
        valid_mask_len = np.sum(valid_mask_positions, axis=1).reshape(-1, 1)
        for i, mask_pos in enumerate(masked_lm_positions):
            pos = [True] * seq_len
            for mask_index, m in enumerate(mask_pos):
                if mask_index < valid_mask_len[i]:
                    pos[m] = False
            valid_seq_positions.append(np.logical_and(pos, input_ids[i] != 0))
        valid_seq_len = np.minimum(
            np.sum(valid_seq_positions, axis=1) + self.max_mask_tokens,
            self.max_seq_length).reshape(-1, 1)
        unmasked_len = np.minimum(np.sum(valid_seq_positions, axis=1),
                                  self.max_seq_length - self.max_mask_tokens)
        for i in range(batch_size):
            target_mask_indices = np.arange(valid_mask_len[i])
            target_seq_indices = self.max_mask_tokens + np.arange(
                unmasked_len[i])
            source_mask_indices = masked_lm_positions[i][
                valid_mask_positions[i]]
            source_seq_indices = np.arange(seq_len)[
                valid_seq_positions[i]][:unmasked_len[i]]

            target_indices = np.hstack(
                [target_mask_indices, target_seq_indices])
            source_indices = np.hstack(
                [source_mask_indices, source_seq_indices])

            formatted_pos[i, target_indices] = source_indices
            formatted_input[i, target_indices] = input_ids[i, source_indices]
            formatted_seg[i, target_indices] = segment_ids[i, source_indices]
            formatted_mask_labels[i] = masked_lm_ids[i, :self.max_mask_tokens]

        return [
            formatted_input.astype(np.int32),
            formatted_seg.astype(np.int32),
            formatted_pos.astype(np.int32),
            valid_mask_len.astype(np.int32),
            valid_seq_len.astype(np.int32),
            formatted_mask_labels.astype(np.int32),
            next_sentence_labels.astype(np.int32)
        ]

    def load_one_file(self, file_path):
        data = self.load_hdf5(file_path)

        if self.shuffle:
            shuffle_dict(data, len(data[KEYS[0]]))

        return data

    def load_hdf5(self, filename):
        with h5py.File(filename, "r") as f:
            data = {key: np.asarray(f[key][:]) for key in KEYS}
        return data


if __name__ == "__main__":
    import glob
    base_dir = 'data_path/wikicorpus_en/'
    input_files = glob.glob(f"{base_dir}/*training*.hdf5")
    input_files.sort()
    # print(input_files)

    seed = 1984
    np.random.seed(seed)
    paddle.seed(seed)

    data_loader = PretrainingHDF5DataLoader(input_files,
                                            batch_size=65536,
                                            shuffle=True)

    for idx, batch in enumerate(data_loader):
        print(f"{idx}: {batch[0].shape()}")
