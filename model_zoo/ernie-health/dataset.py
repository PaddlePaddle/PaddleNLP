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

import os

import numpy as np
import paddle


class MedicalCorpus(paddle.io.Dataset):

    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        # Add ids for suffixal chinese tokens in tokenized text, e.g. '##度' in '百度'.
        # It should coincide with the vocab dictionary in preprocess.py.
        orig_len = len(self.tokenizer)
        suffix_vocab = {}
        for idx, token in enumerate(range(0x4E00, 0x9FA6)):
            suffix_vocab[len(self.tokenizer) + idx] = '##' + chr(token)
        self.tokenizer.added_tokens_decoder.update(suffix_vocab)
        self._samples, self._global_index = self._read_data_files(data_path)

    def _get_data_files(self, data_path):
        # Get all prefix of .npy/.npz files in the current and next-level directories.
        files = [
            os.path.join(data_path, f) for f in os.listdir(data_path)
            if (os.path.isfile(os.path.join(data_path, f))
                and '_idx.npz' in str(f))
        ]
        files = [x.replace('_idx.npz', '') for x in files]
        return files

    def _read_data_files(self, data_path):
        data_files = self._get_data_files(data_path)
        samples = []
        indexes = []
        for file_id, file_name in enumerate(data_files):

            for suffix in ['_ids.npy', '_idx.npz']:
                if not os.path.isfile(file_name + suffix):
                    raise ValueError('File Not found, %s' %
                                     (file_name + suffix))

            token_ids = np.load(file_name + '_ids.npy',
                                mmap_mode='r',
                                allow_pickle=True)
            samples.append(token_ids)

            split_ids = np.load(file_name + '_idx.npz')
            end_ids = np.cumsum(split_ids['lens'], dtype=np.int64)
            file_ids = np.full(end_ids.shape, file_id)
            split_ids = np.stack([file_ids, end_ids], axis=-1)
            indexes.extend(split_ids)
        indexes = np.stack(indexes, axis=0)
        return samples, indexes

    def __len__(self):
        return len(self._global_index)

    def __getitem__(self, index):
        file_id, end_id = self._global_index[index]
        start_id = 0
        if index > 0:
            pre_file_id, pre_end_id = self._global_index[index - 1]
            if pre_file_id == file_id:
                start_id = pre_end_id
        word_token_ids = self._samples[file_id][start_id:end_id]
        token_ids = []
        is_suffix = np.zeros(word_token_ids.shape)
        for idx, token_id in enumerate(word_token_ids):
            token = self.tokenizer.convert_ids_to_tokens(int(token_id))
            if '##' in token:
                token_id = self.tokenizer.convert_tokens_to_ids(token[-1])
                is_suffix[idx] = 1
            token_ids.append(token_id)

        return token_ids, is_suffix.astype(np.int64)


class DataCollatorForErnieHealth(object):

    def __init__(self, tokenizer, mlm_prob, max_seq_length, return_dict=False):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.max_seq_len = max_seq_length
        self.return_dict = return_dict
        self._ids = {
            'cls':
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token),
            'sep':
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token),
            'pad':
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            'mask':
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        }

    def __call__(self, data):
        masked_input_ids_a, input_ids_a, labels_a = self.mask_tokens(data)
        masked_input_ids_b, input_ids_b, labels_b = self.mask_tokens(data)
        masked_input_ids = paddle.concat(
            [masked_input_ids_a, masked_input_ids_b], axis=0).astype('int64')
        input_ids = paddle.concat([input_ids_a, input_ids_b], axis=0)
        labels = paddle.concat([labels_a, labels_b], axis=0)
        if self.return_dict:
            return {
                "input_ids": masked_input_ids,
                "raw_input_ids": input_ids,
                "generator_labels": labels
            }

        else:
            return masked_input_ids, input_ids, labels

    def mask_tokens(self, batch_data):

        token_ids = [x[0] for x in batch_data]
        is_suffix = [x[1] for x in batch_data]

        # Create probability matrix where the probability of real tokens is
        # self.mlm_prob, while that of others is zero.
        data = self.add_special_tokens_and_set_maskprob(token_ids, is_suffix)
        token_ids, is_suffix, prob_matrix = data
        token_ids = paddle.to_tensor(token_ids,
                                     dtype='int64',
                                     stop_gradient=True)
        masked_token_ids = token_ids.clone()
        labels = token_ids.clone()

        # Create masks for words, where '百' must be masked if '度' is masked
        # for the word '百度'.
        prob_matrix = prob_matrix * (1 - is_suffix)
        word_mask_index = np.random.binomial(1, prob_matrix).astype('float')
        is_suffix_mask = (is_suffix == 1)
        word_mask_index_tmp = word_mask_index
        while word_mask_index_tmp.sum() > 0:
            word_mask_index_tmp = np.concatenate([
                np.zeros(
                    (word_mask_index.shape[0], 1)), word_mask_index_tmp[:, :-1]
            ],
                                                 axis=1)
            word_mask_index_tmp = word_mask_index_tmp * is_suffix_mask
            word_mask_index += word_mask_index_tmp
        word_mask_index = word_mask_index.astype('bool')
        labels[~word_mask_index] = -100

        # 80% replaced with [MASK].
        token_mask_index = paddle.bernoulli(paddle.full(
            labels.shape, 0.8)).astype('bool').numpy() & word_mask_index
        masked_token_ids[token_mask_index] = self._ids['mask']

        # 10% replaced with random token ids.
        token_random_index = paddle.to_tensor(
            paddle.bernoulli(paddle.full(labels.shape, 0.5)).astype(
                'bool').numpy() & word_mask_index & ~token_mask_index)
        random_tokens = paddle.randint(low=0,
                                       high=self.tokenizer.vocab_size,
                                       shape=labels.shape,
                                       dtype='int64')
        masked_token_ids = paddle.where(token_random_index, random_tokens,
                                        masked_token_ids)

        return masked_token_ids, token_ids, labels

    def add_special_tokens_and_set_maskprob(self, token_ids, is_suffix):
        batch_size = len(token_ids)
        batch_token_ids = np.full((batch_size, self.max_seq_len),
                                  self._ids['pad'])
        batch_token_ids[:, 0] = self._ids['cls']
        batch_is_suffix = np.full_like(batch_token_ids, -1)
        prob_matrix = np.zeros_like(batch_token_ids, dtype='float32')

        for idx in range(batch_size):
            if len(token_ids[idx]) > self.max_seq_len - 2:
                token_ids[idx] = token_ids[idx][:self.max_seq_len - 2]
                is_suffix[idx] = is_suffix[idx][:self.max_seq_len - 2]
            seq_len = len(token_ids[idx])
            batch_token_ids[idx, seq_len + 1] = self._ids['sep']
            batch_token_ids[idx, 1:seq_len + 1] = token_ids[idx]
            batch_is_suffix[idx, 1:seq_len + 1] = is_suffix[idx]
            prob_matrix[idx, 1:seq_len + 1] = self.mlm_prob

        return batch_token_ids, batch_is_suffix, prob_matrix


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      use_gpu=True,
                      data_collator=None):
    """
    Creats dataloader.
    Args:
        dataset(obj:`paddle.io.Dataset`):
            Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`):
            If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): 
            The sample number of a mini-batch.
        use_gpu(obj:`bool`, optional, defaults to obj:`True`):
            Whether to use gpu to run.
    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
        dataloader = paddle.io.DataLoader(dataset,
                                          batch_sampler=sampler,
                                          return_list=True,
                                          collate_fn=data_collator,
                                          num_workers=0)
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle)
        dataloader = paddle.io.DataLoader(dataset,
                                          batch_sampler=sampler,
                                          return_list=True,
                                          collate_fn=data_collator,
                                          num_workers=0)

    return dataloader
