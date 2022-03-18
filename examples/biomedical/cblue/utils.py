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

import numpy as np
import paddle

from paddlenlp.transformers import normalize_chars, tokenize_special_chars


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequences for sequence
    classification tasks by concatenating and adding special tokens. And
    creates a mask from the two sequences for sequence-pair classification
    tasks.

    The convention in Electra/EHealth is:

    - single sequence:
        input_ids:      ``[CLS] X [SEP]``
        token_type_ids: ``  0   0   0``
        position_ids:   ``  0   1   2``

    - a senquence pair:
        input_ids:      ``[CLS] X [SEP] Y [SEP]``
        token_type_ids: ``  0   0   0   1   1``
        position_ids:   ``  0   1   2   3   4``

    Args:
        example (obj:`dict`):
            A dictionary of input data, containing text and label if it has.
        tokenizer (obj:`PretrainedTokenizer`):
            A tokenizer inherits from :class:`paddlenlp.transformers.PretrainedTokenizer`.
            Users can refer to the superclass for more information.
        max_seq_length (obj:`int`):
            The maximum total input sequence length after tokenization.
            Sequences longer will be truncated, and the shorter will be padded.
        is_test (obj:`bool`, default to `False`):
            Whether the example contains label or not.

    Returns:
        input_ids (obj:`list[int]`):
            The list of token ids.
        token_type_ids (obj:`list[int]`):
            List of sequence pair mask.
        position_ids (obj:`list[int]`):
            List of position ids.
        label(obj:`numpy.array`, data type of int64, optional):
            The input label if not is_test.
    """
    text_a = example['text_a']
    text_b = example.get('text_b', None)

    text_a = tokenize_special_chars(normalize_chars(text_a))
    if text_b is not None:
        text_b = tokenize_special_chars(normalize_chars(text_b))

    encoded_inputs = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_seq_len=max_seq_length,
        return_position_ids=True)
    input_ids = encoded_inputs['input_ids']
    token_type_ids = encoded_inputs['token_type_ids']
    position_ids = encoded_inputs['position_ids']

    if is_test:
        return input_ids, token_type_ids, position_ids
    label = np.array([example['label']], dtype='int64')
    return input_ids, token_type_ids, position_ids, label


def convert_example_ner(example,
                        tokenizer,
                        max_seq_length=512,
                        pad_label_id=-100):
    text = example['text']
    text = tokenize_special_chars(normalize_chars(text))
    encoded_inputs = tokenizer(
        text=text,
        max_seq_len=max_seq_length,
        return_position_ids=True,
        return_attention_mask=True)
    input_len = len(encoded_inputs['input_ids'])

    if example.get('labels', None):
        labels = example['labels']
        if input_len - 2 < len(labels[0]):
            labels[0] = labels[0][:input_len - 2]
        if input_len - 2 < len(labels[1]):
            labels[1] = labels[1][:input_len - 2]
        encoded_inputs['label_oth'] = [pad_label_id[0]] + labels[
            0] + [pad_label_id[0]]
        encoded_inputs['label_sym'] = [pad_label_id[1]] + labels[
            1] + [pad_label_id[1]]

    return encoded_inputs


def convert_example_spo(example, tokenizer, max_seq_length=512, is_test=False):
    text = example['text']
    text = tokenize_special_chars(normalize_chars(text))
    encoded_inputs = tokenizer(
        text=text, max_seq_len=max_seq_length, return_position_ids=True)
    input_len = len(encoded_inputs['input_ids'])
    encoded_inputs['mask'] = np.ones(input_len)
    if not is_test:
        encoded_inputs['ent_label'] = example['ent_label']
        encoded_inputs['spo_label'] = example['spo_label']
    return encoded_inputs


def create_batch_label(ent_labels, spo_labels, num_classes, max_batch_len):
    batch_size = len(ent_labels)
    pad_ent_labels = np.zeros([batch_size, max_batch_len, 2], dtype=np.float32)
    pad_spo_labels = np.zeros(
        [batch_size, num_classes, max_batch_len, max_batch_len],
        dtype=np.float32)
    for idx, ent_idxs in enumerate(ent_labels):
        for x, y in ent_idxs:
            if x > 0 and x < max_batch_len and y < max_batch_len:
                pad_ent_labels[idx, x, 0] = 1
                pad_ent_labels[idx, y, 1] = 1
    for idx, spo_idxs in enumerate(spo_labels):
        for x, y, z in spo_idxs:
            if x > 0 and x < max_batch_len and y < max_batch_len:
                pad_spo_labels[idx, z, x, y] = 1
    pad_ent_labels = paddle.to_tensor(pad_ent_labels)
    pad_spo_labels = paddle.to_tensor(pad_spo_labels)
    return pad_ent_labels, pad_spo_labels


class SPOEvaluator(paddle.metric.Metric):
    def __init__(self, num_classes=None):
        super(SPOEvaluator, self).__init__()
        self.num_classes = num_classes
        self.num_infer_ent = 0
        self.num_infer_spo = 1e-10
        self.num_label_ent = 0
        self.num_label_spo = 1e-10
        self.num_correct_ent = 0
        self.num_correct_spo = 0

    def compute(self, lengths, ent_preds, spo_preds, ent_labels, spo_labels):
        ent_preds = ent_preds.numpy()
        spo_preds = spo_preds.numpy()
        ent_labels = self._unpadded_labels(ent_labels)
        spo_labels = self._unpadded_labels(spo_labels)

        ent_pred_list = []
        ent_idxs_list = []
        for idx, ent_pred in enumerate(ent_preds):
            seq_len = lengths[idx]
            start = np.where(ent_pred[:, 0] > 0.5)[0]
            end = np.where(ent_pred[:, 1] > 0.5)[0]
            ent_pred = []
            ent_idxs = {}
            for x in start:
                y = end[end >= x]
                if x == 0 or x > seq_len:
                    continue
                if len(y) > 0:
                    y = y[0]
                    if y > seq_len:
                        continue
                    ent_idxs[x] = (x - 1, y)
                    ent_pred.append((x - 1, y))
            ent_pred_list.append(ent_pred)
            ent_idxs_list.append(ent_idxs)

        spo_preds = spo_preds > 0
        spo_pred_list = [[] for _ in range(len(spo_preds))]
        idxs, preds, subs, objs = np.nonzero(spo_preds)
        for idx, p_id, s_id, o_id in zip(idxs, preds, subs, objs):
            obj = ent_idxs_list[idx].get(o_id, None)
            if obj is None:
                continue
            sub = ent_idxs_list[idx].get(s_id, None)
            if sub is None:
                continue
            spo_pred_list[idx].append((sub, p_id, obj))

        correct = {'ent': 0, 'spo': 0}
        infer = {'ent': 0, 'spo': 0}
        label = {'ent': 0, 'spo': 0}
        for ent_pred, ent_true in zip(ent_pred_list, ent_labels):
            infer['ent'] += len(set(ent_pred))
            label['ent'] += len(set(ent_true))
            correct['ent'] += len(set(ent_pred) & set(ent_true))

        for spo_pred, spo_true in zip(spo_pred_list, spo_labels):
            infer['spo'] += len(set(spo_pred))
            label['spo'] += len(set(spo_true))
            correct['spo'] += len(set(spo_pred) & set(spo_true))

        return infer, label, correct

    def update(self, corrects):
        assert len(corrects) == 3
        for item in corrects:
            assert isinstance(item, dict)
            for value in item.values():
                if not self._is_number_or_matrix(value):
                    raise ValueError(
                        'The numbers must be a number(int) or a numpy ndarray.')
        num_infer, num_label, num_correct = corrects
        self.num_infer_ent += num_infer['ent']
        self.num_infer_spo += num_infer['spo']
        self.num_label_ent += num_label['ent']
        self.num_label_spo += num_label['spo']
        self.num_correct_ent += num_correct['ent']
        self.num_correct_spo += num_correct['spo']

    def accumulate(self):
        spo_precision = self.num_correct_spo / self.num_infer_spo
        spo_recall = self.num_correct_spo / self.num_label_spo
        spo_f1 = 2 * self.num_correct_spo / (
            self.num_infer_spo + self.num_label_spo)
        ent_precision = self.num_correct_ent / self.num_infer_ent if self.num_infer_ent > 0 else 0.
        ent_recall = self.num_correct_ent / self.num_correct_ent if self.num_correct_ent > 0 else 0.
        ent_f1 = 2 * ent_precision * ent_recall / (
            ent_precision + ent_recall) if (ent_precision + ent_recall
                                            ) != 0 else 0.
        return {
            'entity': (ent_precision, ent_recall, ent_f1),
            'spo': (spo_precision, spo_recall, spo_f1)
        }

    def _unpadded_labels(self, labels):
        unpads = []
        for label in labels.numpy():
            unpad = []
            for x in label:
                if (x < 0).any():
                    break
                unpad.append(tuple(x.tolist()))
            unpads.append(unpad)
        return unpads

    def _is_number_or_matrix(self, var):
        def _is_number_(var):
            return isinstance(
                var, int) or isinstance(var, np.int64) or isinstance(
                    var, float) or (isinstance(var, np.ndarray) and
                                    var.shape == (1, ))

        return _is_number_(var) or isinstance(var, np.ndarray)

    def reset(self):
        self.num_infer_ent = 0
        self.num_infer_spo = 1e-10
        self.num_label_ent = 0
        self.num_label_spo = 1e-10
        self.num_correct_ent = 0
        self.num_correct_spo = 0

    def name(self):
        return {
            'entity': ('precision', 'recall', 'f1'),
            'spo': ('precision', 'recall', 'f1')
        }
