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

import sys
import math
import numpy as np
import paddle
from paddle.optimizer.lr import LambdaDecay

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


class LinearDecayWithWarmup(LambdaDecay):

    def __init__(self,
                 learning_rate,
                 total_steps,
                 warmup,
                 last_epoch=-1,
                 verbose=False):
        """
        Creates a learning rate scheduler, which increases learning rate linearly
        from 0 to given `learning_rate`, after this warmup period learning rate
        would be decreased linearly from the base learning rate to 0.

        Args:
            learning_rate (float):
                The base learning rate. It is a python float number.
            total_steps (int):
                The number of training steps.
            warmup (int or float):
                If int, it means the number of steps for warmup. If float, it means
                the proportion of warmup in total training steps.
            last_epoch (int, optional):
                The index of last epoch. It can be set to restart training. If
                None, it means initial learning rate. 
                Defaults to -1.
            verbose (bool, optional):
                If True, prints a message to stdout for each update.
                Defaults to False.
        """

        warmup_steps = warmup if isinstance(warmup, int) else int(
            math.floor(warmup * total_steps))

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, 1.0 - current_step / total_steps)

        super(LinearDecayWithWarmup, self).__init__(learning_rate, lr_lambda,
                                                    last_epoch, verbose)


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

    encoded_inputs = tokenizer(text=text_a,
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
                        pad_label_id=-100,
                        is_test=False):
    """
    Builds model inputs from a sequence and creates labels for named-
    entity recognition task CMeEE.

    For example, a sample should be:

    - input_ids:      ``[CLS]  x1   x2 [SEP] [PAD]``
    - token_type_ids: ``  0    0    0    0     0``
    - position_ids:   ``  0    1    2    3     0``
    - attention_mask: ``  1    1    1    1     0``
    - label_oth:      `` 32    3   32   32    32`` (optional, label ids of others)
    - label_sym:      ``  4    4    4    4     4`` (optional, label ids of symptom)

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
        encoded_output (obj: `dict[str, list|np.array]`):
            The sample dictionary including `input_ids`, `token_type_ids`,
            `position_ids`, `attention_mask`, `label_oth` (optional), 
            `label_sym` (optional)
    """

    encoded_inputs = {}
    text = example['text']
    if len(text) > max_seq_length - 2:
        text = text[:max_seq_length - 2]
    text = ['[CLS]'] + [x.lower() for x in text] + ['[SEP]']
    input_len = len(text)
    encoded_inputs['input_ids'] = tokenizer.convert_tokens_to_ids(text)
    encoded_inputs['token_type_ids'] = np.zeros(input_len)
    encoded_inputs['position_ids'] = list(range(input_len))
    encoded_inputs['attention_mask'] = np.ones(input_len)

    if not is_test:
        labels = example['labels']
        if input_len - 2 < len(labels[0]):
            labels[0] = labels[0][:input_len - 2]
        if input_len - 2 < len(labels[1]):
            labels[1] = labels[1][:input_len - 2]
        encoded_inputs['label_oth'] = [pad_label_id[0]
                                       ] + labels[0] + [pad_label_id[0]]
        encoded_inputs['label_sym'] = [pad_label_id[1]
                                       ] + labels[1] + [pad_label_id[1]]

    return encoded_inputs


def convert_example_spo(example,
                        tokenizer,
                        num_classes,
                        max_seq_length=512,
                        is_test=False):
    """
    Builds model inputs from a sequence and creates labels for SPO prediction
    task CMeIE.

    For example, a sample should be:
    
    - input_ids:      ``[CLS]  x1   x2 [SEP] [PAD]``
    - token_type_ids: ``  0    0    0    0     0``
    - position_ids:   ``  0    1    2    3     0``
    - attention_mask: ``  1    1    1    1     0``
    - ent_label:      ``[[0    1    0    0     0], # start ids are set as 1
                         [0    0    1    0     0]] # end ids are set as 1
    - spo_label: a tensor of shape [num_classes, max_batch_len, max_batch_len].
                 Set [predicate_id, subject_start_id, object_start_id] as 1
                 when (subject, predicate, object) exists.

    Args:
        example (obj:`dict`):
            A dictionary of input data, containing text and label if it has.
        tokenizer (obj:`PretrainedTokenizer`):
            A tokenizer inherits from :class:`paddlenlp.transformers.PretrainedTokenizer`.
            Users can refer to the superclass for more information.
        num_classes (obj:`int`):
            The number of predicates.
        max_seq_length (obj:`int`):
            The maximum total input sequence length after tokenization.
            Sequences longer will be truncated, and the shorter will be padded.
        is_test (obj:`bool`, default to `False`):
            Whether the example contains label or not.

    Returns:
        encoded_output (obj: `dict[str, list|np.array]`):
            The sample dictionary including `input_ids`, `token_type_ids`,
            `position_ids`, `attention_mask`, `ent_label` (optional),
            `spo_label` (optional)
    """
    encoded_inputs = {}
    text = example['text']
    if len(text) > max_seq_length - 2:
        text = text[:max_seq_length - 2]
    text = ['[CLS]'] + [x.lower() for x in text] + ['[SEP]']
    input_len = len(text)
    encoded_inputs['input_ids'] = tokenizer.convert_tokens_to_ids(text)
    encoded_inputs['token_type_ids'] = np.zeros(input_len)
    encoded_inputs['position_ids'] = list(range(input_len))
    encoded_inputs['attention_mask'] = np.ones(input_len)
    if not is_test:
        encoded_inputs['ent_label'] = example['ent_label']
        encoded_inputs['spo_label'] = example['spo_label']
    return encoded_inputs


class NERChunkEvaluator(paddle.metric.Metric):
    """
    NERChunkEvaluator computes the precision, recall and F1-score for chunk detection.
    It is often used in sequence tagging tasks, such as Named Entity Recognition (NER).

    Args:
        label_list (list):
            The label list.

    Note:
        Difference from `paddlenlp.metric.ChunkEvaluator`:

        - `paddlenlp.metric.ChunkEvaluator`
           All sequences with non-'O' labels are taken as chunks when computing num_infer.
        - `NERChunkEvaluator`
           Only complete sequences are taken as chunks, namely `B- I- E-` or `S-`. 
    """

    def __init__(self, label_list):
        super(NERChunkEvaluator, self).__init__()
        self.id2label = [dict(enumerate(x)) for x in label_list]
        self.num_classes = [len(x) for x in label_list]
        self.num_infer = 0
        self.num_label = 0
        self.num_correct = 0

    def compute(self, lengths, predictions, labels):
        """
        Computes the prediction, recall and F1-score for chunk detection.

        Args:
            lengths (Tensor):
                The valid length of every sequence, a tensor with shape `[batch_size]`.
            predictions (Tensor):
                The predictions index, a tensor with shape `[batch_size, sequence_length]`.
            labels (Tensor):
                The labels index, a tensor with shape `[batch_size, sequence_length]`.

        Returns:
            tuple: Returns tuple (`num_infer_chunks, num_label_chunks, num_correct_chunks`).

            With the fields:

            - `num_infer_chunks` (Tensor): The number of the inference chunks.
            - `num_label_chunks` (Tensor): The number of the label chunks.
            - `num_correct_chunks` (Tensor): The number of the correct chunks.
        """
        assert len(predictions) == len(labels)
        assert len(predictions) == len(self.id2label)
        preds = [x.numpy() for x in predictions]
        labels = [x.numpy() for x in labels]

        preds_chunk = set()
        label_chunk = set()
        for idx, (pred, label) in enumerate(zip(preds, labels)):
            for i, case in enumerate(pred):
                case = [self.id2label[idx][x] for x in case[:lengths[i]]]
                preds_chunk |= self.extract_chunk(case, i)
            for i, case in enumerate(label):
                case = [self.id2label[idx][x] for x in case[:lengths[i]]]
                label_chunk |= self.extract_chunk(case, i)

        num_infer = len(preds_chunk)
        num_label = len(label_chunk)
        num_correct = len(preds_chunk & label_chunk)
        return num_infer, num_label, num_correct

    def update(self, correct):
        num_infer, num_label, num_correct = correct
        self.num_infer += num_infer
        self.num_label += num_label
        self.num_correct += num_correct

    def accumulate(self):
        precision = self.num_correct / (self.num_infer + 1e-6)
        recall = self.num_correct / (self.num_label + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return precision, recall, f1

    def reset(self):
        self.num_infer = 0
        self.num_label = 0
        self.num_correct = 0

    def name(self):
        return 'precision', 'recall', 'f1'

    def extract_chunk(self, sequence, cid=0):
        chunks = set()

        start_idx, cur_idx = 0, 0
        while cur_idx < len(sequence):
            if sequence[cur_idx][0] == 'B':
                start_idx = cur_idx
                cur_idx += 1
                while cur_idx < len(sequence) and sequence[cur_idx][0] == 'I':
                    if sequence[cur_idx][2:] == sequence[start_idx][2:]:
                        cur_idx += 1
                    else:
                        break
                if cur_idx < len(sequence) and sequence[cur_idx][0] == 'E':
                    if sequence[cur_idx][2:] == sequence[start_idx][2:]:
                        chunks.add(
                            (cid, sequence[cur_idx][2:], start_idx, cur_idx))
                        cur_idx += 1
            elif sequence[cur_idx][0] == 'S':
                chunks.add((cid, sequence[cur_idx][2:], cur_idx, cur_idx))
                cur_idx += 1
            else:
                cur_idx += 1

        return chunks


class SPOChunkEvaluator(paddle.metric.Metric):
    """
    SPOChunkEvaluator computes the precision, recall and F1-score for multiple
    chunk detections, including Named Entity Recognition (NER) and SPO Prediction.

    Args:
        num_classes (int):
            The number of predicates.
    """

    def __init__(self, num_classes=None):
        super(SPOChunkEvaluator, self).__init__()
        self.num_classes = num_classes
        self.num_infer_ent = 0
        self.num_infer_spo = 1e-10
        self.num_label_ent = 0
        self.num_label_spo = 1e-10
        self.num_correct_ent = 0
        self.num_correct_spo = 0

    def compute(self, lengths, ent_preds, spo_preds, ent_labels, spo_labels):
        """
        Computes the prediction, recall and F1-score for NER and SPO prediction.

        Args:
            lengths (Tensor):
                The valid length of every sequence, a tensor with shape `[batch_size]`.
            ent_preds (Tensor):
                The predictions of entities.
                A tensor with shape `[batch_size, sequence_length, 2]`.
                `ent_preds[:, :, 0]` denotes the start indexes of entities.
                `ent_preds[:, :, 1]` denotes the end indexes of entities.
            spo_preds (Tensor):
                The predictions of predicates between all possible entities.
                A tensor with shape `[batch_size, num_classes, sequence_length, sequence_length]`.
            ent_labels (list[list|tuple]):
                The entity labels' indexes. A list of pair `[start_index, end_index]`.
            spo_labels (list[list|tuple]):
                The SPO labels' indexes. A list of triple `[[subject_start_index, subject_end_index], 
                predicate_id, [object_start_index, object_end_index]]`.

        Returns:
            tuple:
                Returns tuple (`num_infer_chunks, num_label_chunks, num_correct_chunks`).
                The `ent` denotes results of NER and the `spo` denotes results of SPO prediction.

            With the fields:

            - `num_infer_chunks` (dict): The number of the inference chunks.
            - `num_label_chunks` (dict): The number of the label chunks.
            - `num_correct_chunks` (dict): The number of the correct chunks.
        """
        ent_preds = ent_preds.numpy()
        spo_preds = spo_preds.numpy()

        ent_pred_list = []
        ent_idxs_list = []
        for idx, ent_pred in enumerate(ent_preds):
            seq_len = lengths[idx] - 2
            start = np.where(ent_pred[:, 0] > 0.5)[0]
            end = np.where(ent_pred[:, 1] > 0.5)[0]
            ent_pred = []
            ent_idxs = {}
            for x in start:
                y = end[end >= x]
                if (x == 0) or (x > seq_len):
                    continue
                if len(y) > 0:
                    y = y[0]
                    if y > seq_len:
                        continue
                    ent_idxs[x] = (x - 1, y - 1)
                    ent_pred.append((x - 1, y - 1))
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
            ent_true = [tuple(x) for x in ent_true]
            infer['ent'] += len(set(ent_pred))
            label['ent'] += len(set(ent_true))
            correct['ent'] += len(set(ent_pred) & set(ent_true))

        for spo_pred, spo_true in zip(spo_pred_list, spo_labels):
            spo_true = [(tuple(s), p, tuple(o)) for s, p, o in spo_true]
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
        spo_f1 = 2 * self.num_correct_spo / (self.num_infer_spo +
                                             self.num_label_spo)
        ent_precision = self.num_correct_ent / self.num_infer_ent if self.num_infer_ent > 0 else 0.
        ent_recall = self.num_correct_ent / self.num_label_ent if self.num_label_ent > 0 else 0.
        ent_f1 = 2 * ent_precision * ent_recall / (
            ent_precision + ent_recall) if (ent_precision +
                                            ent_recall) != 0 else 0.
        return {
            'entity': (ent_precision, ent_recall, ent_f1),
            'spo': (spo_precision, spo_recall, spo_f1)
        }

    def _is_number_or_matrix(self, var):

        def _is_number_(var):
            return isinstance(var,
                              int) or isinstance(var, np.int64) or isinstance(
                                  var, float) or (isinstance(var, np.ndarray)
                                                  and var.shape == (1, ))

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
