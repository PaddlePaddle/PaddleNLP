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

import multiprocessing
import random
import threading
from collections import deque
from queue import Queue

import numpy as np
import paddle

try:
    from torch_xla.utils.tf_record_reader import TfRecordReader
except ImportError:
    raise ImportError("""Torch-xla required for TFRecord dataset.
                      Please install torch 1.7.0 & torch-xla using
                     `pip install torch==1.7.0 torch-xla@https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.7-cp37-cp37m-linux_x86_64.whl`"""
                      )

KEYS = ('masked_lm_ids', 'masked_lm_weights', 'segment_ids', 'input_ids',
        'input_mask', 'next_sentence_labels', 'masked_lm_positions')


class PretrainingTfRecordDataLoader:
    def __init__(self,
                 input_files,
                 max_seq_length=128,
                 max_mask_tokens=20,
                 batch_size=1,
                 micro_batch_size=1,
                 dtype=np.int32,
                 shuffle=False,
                 pad_position_value=511,
                 prefetch=1,
                 drop_remainder=False,
                 enable_fp16=False,
                 enable_ipu=False,
                 enable_check_data=False,
                 ignore_index=-1):
        self.files = input_files
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.max_seq_length = max_seq_length
        self.max_mask_tokens = max_mask_tokens
        self.dtype = dtype
        self.file_index = 0
        self.data_index = 0
        self.shuffle = shuffle
        self.len = None
        self.pad_position_value = pad_position_value
        self.drop_remainder = drop_remainder
        self.enable_fp16 = enable_fp16
        self.enable_ipu = enable_ipu
        self.enable_check_data = enable_check_data
        self.ignore_index = ignore_index
        pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), 32))
        num_samples = pool.map(self.samples_in_file, self.files)
        pool.close()
        pool.join()
        self.total_samples = sum(num_samples)
        self.len = self.total_samples // (self.batch_size)
        self.num_prefetch_batches = prefetch
        self.prefetch_buffer = deque()
        self.process_buffer = multiprocessing.Manager().Queue(10)
        self.event_queue = multiprocessing.Manager().Queue(10)
        self.feed_buffer = Queue(20)
        if self.len < 1:
            raise ValueError(f"""Batch size {self.batch_size} larger than
                                number of samples in the TFRecord files {self.total_samples}."""
                             )

        if self.len < self.num_prefetch_batches:
            raise ValueError(
                f"""Not enough samples to prefetch: (length = {self.len},
                            num_to_prefech = {self.num_prefetch_batches}),
                            lower the number of prefetch batches.""")
        self.samples_per_file = {
            f: n
            for (f, n) in zip(self.files, num_samples)
        }
        self.data = None
        self.counter = 0

        # multi-process
        # workers = multiprocessing.Pool()
        self.thread_stop = False
        self.thread_process = threading.Thread(target=self.post_fetch)

        self.process_number = 3
        self.files_per_process = len(self.files) // self.process_number
        self.split_files = np.array_split(self.files, self.process_number)
        self.processor = [
            multiprocessing.Process(
                target=self.fill_buffer_loop, args=(i, self.split_files[i]))
            for i in range(self.process_number)
        ]
        for p in self.processor:
            p.start()
        self.thread_process.start()

    def post_fetch(self):
        while True:
            if not self.event_queue.empty():
                return
            if not self.process_buffer.empty():
                np_feed_list = self.process_buffer.get()
                lod_feed_list = []
                for data in np_feed_list:
                    tensor = paddle.fluid.core.LoDTensor()
                    place = paddle.CPUPlace()
                    tensor.set(data, place)
                    lod_feed_list.append(tensor)
                self.feed_buffer.put(lod_feed_list)

    def samples_in_file(self, filename):
        reader = TfRecordReader(
            filename,
            transforms={
                k: lambda x: x.numpy().astype(self.dtype)
                for k in KEYS
            })
        count = 0
        while reader.read_example():
            count += 1
        return count

    def release(self):
        self.event_queue.put('END')
        while not self.feed_buffer.empty():
            self.feed_buffer.get()
        while not self.process_buffer.empty():
            self.process_buffer.get()
        self.thread_process.join()
        for p in self.processor:
            p.join()
        return

    def __len__(self):
        return self.len

    def __iter__(self):
        self.file_index = 0
        self.data_index = 0
        self.counter = 0
        self.data = None
        # if self.shuffle:
        #     random.shuffle(self.files)
        # self.fill_buffer(self.num_prefetch_batches)
        return self

    def fill_buffer_loop(self, i, files):
        data = None
        data_index = 0
        file_index = 0

        def multiprocess_fill_buffer(num_batches, data, file_index, data_index):
            if data is None:
                data = self.return_load_data(files[file_index])
                file_index += 1
                data_index = 0
            for _ in range(num_batches):
                curr_batch = []
                still_required = self.batch_size
                while still_required > 0:
                    data_batch = data[data_index:data_index + still_required]
                    data_index += len(data_batch)
                    curr_batch += data_batch
                    still_required = self.batch_size - len(curr_batch)
                    if still_required > 0:
                        if file_index >= len(files):
                            random.shuffle(files)
                            file_index = 0

                        data = self.return_load_data(files[file_index])
                        file_index += 1
                        data_index = 0
                if len(curr_batch) == self.batch_size:
                    result = {}
                    for k in KEYS:
                        result[k] = np.vstack([item[k] for item in curr_batch])
                    self.process_buffer.put(self.post_process(result))

            return data, file_index, data_index

        while True:
            if self.event_queue.empty():
                data, file_index, data_index = multiprocess_fill_buffer(
                    1, data, file_index, data_index)
            else:
                return

    def post_process(self, samples):
        # process_start = time.time()
        batch_size, seq_len = samples['input_ids'].shape
        formatted_pos = self.pad_position_value * np.ones_like(samples[
            'input_ids'])
        formatted_input = np.zeros_like(samples['input_ids'])
        formatted_seg = np.zeros_like(samples['segment_ids'])
        formatted_mask_labels = np.zeros(
            (batch_size, self.max_mask_tokens),
            dtype=samples['masked_lm_ids'].dtype)

        valid_seq_positions = []
        valid_mask_positions = samples['masked_lm_weights'] == 1
        valid_mask_len = np.sum(valid_mask_positions, axis=1).reshape(-1, 1)
        for i, mask_pos in enumerate(samples['masked_lm_positions']):
            pos = [True] * seq_len
            for mask_index, m in enumerate(mask_pos):
                if mask_index < valid_mask_len[i]:
                    pos[m] = False
            valid_seq_positions.append(
                np.logical_and(pos, samples['input_ids'][i] != 0))
        valid_seq_len = np.minimum(
            np.sum(valid_seq_positions, axis=1) + self.max_mask_tokens,
            self.max_seq_length).reshape(-1, 1)
        unmasked_len = np.minimum(
            np.sum(valid_seq_positions, axis=1),
            self.max_seq_length - self.max_mask_tokens)
        for i in range(batch_size):
            target_mask_indices = np.arange(valid_mask_len[i])
            target_seq_indices = self.max_mask_tokens + np.arange(unmasked_len[
                i])
            source_mask_indices = samples['masked_lm_positions'][i][
                valid_mask_positions[i]]
            source_seq_indices = np.arange(seq_len)[valid_seq_positions[
                i]][:unmasked_len[i]]

            target_indices = np.hstack(
                [target_mask_indices, target_seq_indices])
            source_indices = np.hstack(
                [source_mask_indices, source_seq_indices])

            formatted_pos[i, target_indices] = source_indices
            formatted_input[i, target_indices] = samples['input_ids'][
                i, source_indices]
            formatted_seg[i, target_indices] = samples['segment_ids'][
                i, source_indices]
            formatted_mask_labels[i] = samples['masked_lm_ids'][
                i, :self.max_mask_tokens]

        # process_cost = time.time() - process_start
        # print("DEBUG: process cost: {}".format(process_cost))

        return [
            formatted_input.astype(np.int32), formatted_seg.astype(np.int32),
            formatted_pos.astype(np.int32), valid_mask_len.astype(np.int32),
            valid_seq_len.astype(np.int32),
            formatted_mask_labels.astype(np.int32),
            samples['next_sentence_labels'].astype(np.int32)
        ]

    def __next__(self):
        if self.drop_remainder:
            if self.counter == self.len:
                raise StopIteration

        result = self.feed_buffer.get()
        self.counter += 1
        return result

    def load_data(self):
        if self.file_index >= len(self.files):
            raise ValueError('No more files to load.')
        self.data = self.load_file(self.files[self.file_index])
        self.file_index += 1
        self.data_index = 0
        if self.shuffle:
            np.random.shuffle(self.data)

    def return_load_data(self, file_path):
        data = self.load_file(file_path)
        # self.file_index += 1
        # self.data_index = 0
        if self.shuffle:
            np.random.shuffle(data)
        return data

    def load_file(self, filename):
        reader = TfRecordReader(
            filename,
            transforms={
                k: lambda x: x.numpy().astype(self.dtype)
                for k in KEYS
            })
        data = []
        ex = reader.read_example()
        while ex:
            data.append(ex)
            ex = reader.read_example()
        return data
