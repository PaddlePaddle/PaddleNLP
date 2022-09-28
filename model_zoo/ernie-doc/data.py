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

import itertools
import numpy as np
from collections import namedtuple

import paddle
from paddle.io import IterableDataset
from paddle.utils import try_import

from paddlenlp.utils.log import logger
from paddlenlp.transformers import tokenize_chinese_chars

__all__ = [
    'ClassifierIterator', 'MRCIterator', 'MCQIterator', 'ImdbTextPreProcessor',
    'HYPTextPreProcessor'
]


def get_related_pos(insts, seq_len, memory_len=128):
    """generate relative postion ids"""
    beg = seq_len + seq_len + memory_len
    r_position = [list(range(beg - 1, seq_len - 1, -1)) + \
                  list(range(0, seq_len)) for i in range(len(insts))]
    return np.array(r_position).astype('int64').reshape([len(insts), beg, 1])


def pad_batch_data(insts,
                   insts_data_type="int64",
                   pad_idx=0,
                   final_cls=False,
                   pad_max_len=None,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    if pad_max_len:
        max_len = pad_max_len
    else:
        max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    # Input id
    if final_cls:
        inst_data = np.array([
            inst[:-1] + list([pad_idx] * (max_len - len(inst))) + [inst[-1]]
            for inst in insts
        ])
    else:
        inst_data = np.array(
            [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype(insts_data_type).reshape([-1, max_len, 1])]

    # Position id
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        if final_cls:
            input_mask_data = np.array([[1] * len(inst[:-1]) + [0] *
                                        (max_len - len(inst)) + [1]
                                        for inst in insts])
        else:
            input_mask_data = np.array([[1] * len(inst) + [0] *
                                        (max_len - len(inst))
                                        for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens_type = [-1]
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape(seq_lens_type)]

    return return_list if len(return_list) > 1 else return_list[0]


class TextPreprocessor(object):

    def __call__(self, text):
        raise NotImplementedError("TextPreprocessor object can't be called")


class ImdbTextPreprocessor(TextPreprocessor):

    def __call__(self, text):
        text = text.strip().replace('<br /><br />', ' ')
        text = text.replace('\t', '')
        return text


class HYPTextPreprocessor(TextPreprocessor):

    def __init__(self):
        self.bs4 = try_import('bs4')

    def __call__(self, text):
        text = self.bs4.BeautifulSoup(text, "html.parser").get_text()
        text = text.strip().replace('\n', '').replace('\t', '')
        return text


class ClassifierIterator(object):

    def __init__(self,
                 dataset,
                 batch_size,
                 tokenizer,
                 trainer_num,
                 trainer_id,
                 max_seq_length=512,
                 memory_len=128,
                 repeat_input=False,
                 in_tokens=False,
                 mode="train",
                 random_seed=None,
                 preprocess_text_fn=None):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.trainer_num = trainer_num
        self.trainer_id = trainer_id
        self.max_seq_length = max_seq_length
        self.memory_len = memory_len
        self.repeat_input = repeat_input
        self.in_tokens = in_tokens
        self.dataset = [data for data in dataset]
        self.num_examples = None
        self.mode = mode
        self.shuffle = True if mode == "train" else False
        if random_seed is None:
            random_seed = 12345
        self.random_seed = random_seed
        self.preprocess_text_fn = preprocess_text_fn

    def shuffle_sample(self):
        if self.shuffle:
            self.global_rng = np.random.RandomState(self.random_seed)
            self.global_rng.shuffle(self.dataset)

    def _cnt_list(self, inp):
        """Cnt_list"""
        cnt = 0
        for lit in inp:
            if lit:
                cnt += 1
        return cnt

    def _convert_to_features(self, example, qid):
        """
        Convert example to features fed into model
        """
        if "text" in example:  # imdb
            text = example["text"]
        elif "sentence" in example:  # iflytek
            text = example["sentence"]

        if self.preprocess_text_fn:
            text = self.preprocess_text_fn(text)
        label = example["label"]
        doc_spans = []
        _DocSpan = namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        max_tokens_for_doc = self.max_seq_length - 2
        tokens_a = self.tokenizer.tokenize(text)
        while start_offset < len(tokens_a):
            length = len(tokens_a) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(tokens_a):
                break
            start_offset += min(length, self.memory_len)

        features = []
        Feature = namedtuple("Feature",
                             ["src_ids", "label_id", "qid", "cal_loss"])
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = tokens_a[doc_span.start:doc_span.start +
                              doc_span.length] + ["[SEP]"] + ["[CLS]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            features.append(
                Feature(src_ids=token_ids, label_id=label, qid=qid, cal_loss=1))

        if self.repeat_input:
            features_repeat = features
            features = list(map(lambda x: x._replace(cal_loss=0), features))
            features = features + features_repeat
        return features

    def _get_samples(self, pre_batch_list, is_last=False):
        if is_last:
            # Pad batch
            len_doc = [len(doc) for doc in pre_batch_list]
            max_len_idx = len_doc.index(max(len_doc))
            dirty_sample = pre_batch_list[max_len_idx][-1]._replace(cal_loss=0)
            for sample_list in pre_batch_list:
                sample_list.extend([dirty_sample] *
                                   (max(len_doc) - len(sample_list)))

        samples = []
        min_len = min([len(doc) for doc in pre_batch_list])
        for cnt in range(min_len):
            for batch_idx in range(self.batch_size * self.trainer_num):
                sample = pre_batch_list[batch_idx][cnt]
                samples.append(sample)

        for idx in range(len(pre_batch_list)):
            pre_batch_list[idx] = pre_batch_list[idx][min_len:]
        return samples

    def _pad_batch_records(self, batch_records, gather_idx=[]):
        batch_token_ids = [record.src_ids for record in batch_records]
        if batch_records[0].label_id is not None:
            batch_labels = [record.label_id for record in batch_records]
            batch_labels = np.array(batch_labels).astype("int64").reshape(
                [-1, 1])
        else:
            batch_labels = np.array([]).astype("int64").reshape([-1, 1])
        # Qid
        if batch_records[-1].qid is not None:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])
        else:
            batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        if gather_idx:
            batch_gather_idx = np.array(gather_idx).astype("int64").reshape(
                [-1, 1])
            need_cal_loss = np.array([1]).astype("int64")
        else:
            batch_gather_idx = np.array(list(range(
                len(batch_records)))).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([0]).astype("int64")

        # Padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids, pad_idx=self.tokenizer.pad_token_id, pad_max_len=self.max_seq_length, \
            final_cls=True, return_input_mask=True)
        padded_task_ids = np.zeros_like(padded_token_ids, dtype="int64")
        padded_position_ids = get_related_pos(padded_token_ids, \
            self.max_seq_length, self.memory_len)

        return_list = [
            padded_token_ids, padded_position_ids, padded_task_ids, input_mask,
            batch_labels, batch_qids, batch_gather_idx, need_cal_loss
        ]
        return return_list

    def _prepare_batch_data(self, examples):
        batch_records, max_len, gather_idx = [], 0, []
        for index, example in enumerate(examples):
            max_len = max(max_len, len(example.src_ids))
            if self.in_tokens:
                to_append = (len(batch_records) +
                             1) * max_len <= self.batch_size
            else:
                to_append = len(batch_records) < self.batch_size
            if to_append:
                batch_records.append(example)
                if example.cal_loss == 1:
                    gather_idx.append(index % self.batch_size)
            else:
                yield self._pad_batch_records(batch_records, gather_idx)
                batch_records, max_len = [example], len(example.src_ids)
                gather_idx = [index %
                              self.batch_size] if example.cal_loss == 1 else []
        yield self._pad_batch_records(batch_records, gather_idx)

    def _create_instances(self):
        examples = self.dataset
        pre_batch_list = []
        insert_idx = []
        for qid, example in enumerate(examples):
            features = self._convert_to_features(example, qid)
            if self._cnt_list(
                    pre_batch_list) < self.batch_size * self.trainer_num:
                if insert_idx:
                    pre_batch_list[insert_idx[0]] = features
                    insert_idx.pop(0)
                else:
                    pre_batch_list.append(features)
            if self._cnt_list(
                    pre_batch_list) == self.batch_size * self.trainer_num:
                assert self._cnt_list(pre_batch_list) == len(
                    pre_batch_list), "the two value must be equal"
                assert not insert_idx, "the insert_idx must be null"
                sample_batch = self._get_samples(pre_batch_list)

                for idx, lit in enumerate(pre_batch_list):
                    if not lit:
                        insert_idx.append(idx)
                for batch_records in self._prepare_batch_data(sample_batch):
                    yield batch_records

        if self.mode != "train":
            if self._cnt_list(pre_batch_list):
                pre_batch_list += [
                    [] for _ in range(self.batch_size * self.trainer_num -
                                      self._cnt_list(pre_batch_list))
                ]
                sample_batch = self._get_samples(pre_batch_list, is_last=True)
                for batch_records in self._prepare_batch_data(sample_batch):
                    yield batch_records

    def __call__(self):
        curr_id = 0
        for batch_records in self._create_instances():
            if curr_id == self.trainer_id or self.mode != "train":
                yield batch_records
            curr_id = (curr_id + 1) % self.trainer_num

    def get_num_examples(self):
        if self.num_examples is None:
            self.num_examples = 0
            for qid, example in enumerate(self.dataset):
                self.num_examples += len(self._convert_to_features(
                    example, qid))
        return self.num_examples


class MRCIterator(ClassifierIterator):
    """
    Machine Reading Comprehension iterator. Only for answer extraction.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 tokenizer,
                 trainer_num,
                 trainer_id,
                 max_seq_length=512,
                 memory_len=128,
                 repeat_input=False,
                 in_tokens=False,
                 mode="train",
                 random_seed=None,
                 doc_stride=128,
                 max_query_length=64):
        super(MRCIterator, self).__init__(dataset,
                                          batch_size,
                                          tokenizer,
                                          trainer_num,
                                          trainer_id,
                                          max_seq_length,
                                          memory_len,
                                          repeat_input,
                                          in_tokens,
                                          mode,
                                          random_seed,
                                          preprocess_text_fn=None)
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.examples = []
        self.features = []
        self.features_all = []
        self._preprocess_data()

    def shuffle_sample(self):
        if self.shuffle:
            self.global_rng = np.random.RandomState(self.random_seed)
            self.global_rng.shuffle(self.features_all)

    def _convert_qa_to_examples(self):
        Example = namedtuple('Example', [
            'qas_id', 'question_text', 'doc_tokens', 'orig_answer_text',
            'start_position', 'end_position'
        ])
        examples = []
        for qa in self.dataset:
            qas_id = qa["id"]
            question_text = qa["question"]
            context = qa["context"]
            start_pos = None
            end_pos = None
            orig_answer_text = None
            if self.mode == 'train':
                if len(qa["answers"]) != 1:
                    raise ValueError(
                        "For training, each question should have exactly 1 answer."
                    )
                orig_answer_text = qa["answers"][0]
                answer_offset = qa["answer_starts"][0]
                answer_length = len(orig_answer_text)
                doc_tokens = [
                    context[:answer_offset],
                    context[answer_offset:answer_offset + answer_length],
                    context[answer_offset + answer_length:]
                ]

                start_pos = 1
                end_pos = 1

                actual_text = " ".join(doc_tokens[start_pos:(end_pos + 1)])
                if orig_answer_text.islower():
                    actual_text = actual_text.lower()
                if actual_text.find(orig_answer_text) == -1:
                    logger.info("Could not find answer: '%s' vs. '%s'" %
                                (actual_text, orig_answer_text))
                    continue

            else:
                doc_tokens = tokenize_chinese_chars(context)

            example = Example(qas_id=qas_id,
                              question_text=question_text,
                              doc_tokens=doc_tokens,
                              orig_answer_text=orig_answer_text,
                              start_position=start_pos,
                              end_position=end_pos)
            examples.append(example)
        return examples

    def _convert_example_to_feature(self, examples):
        Feature = namedtuple("Feature", [
            "qid", "example_index", "doc_span_index", "tokens",
            "token_to_orig_map", "token_is_max_context", "src_ids",
            "start_position", "end_position", "cal_loss"
        ])
        features = []
        self.features_all = []
        unique_id = 1000
        is_training = self.mode == "train"
        print("total {} examples".format(len(examples)), flush=True)
        for (example_index, example) in enumerate(examples):
            query_tokens = self.tokenizer.tokenize(example.question_text)
            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if is_training:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position +
                                                         1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position,
                 tok_end_position) = self._improve_answer_span(
                     all_doc_tokens, tok_start_position, tok_end_position,
                     example.orig_answer_text)

            max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3
            _DocSpan = namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            features_each = []
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                tokens.append("[CLS]")
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[i +
                                      1] = tok_to_orig_index[split_token_index]
                    is_max_context = self._check_is_max_context(
                        doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[i + 1] = is_max_context
                tokens += all_doc_tokens[doc_span.start:doc_span.start +
                                         doc_span.length]
                tokens.append("[SEP]")

                for token in query_tokens:
                    tokens.append(token)
                tokens.append("[SEP]")

                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                start_position = None
                end_position = None
                if is_training:
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start
                            and tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = 1  #len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                feature = Feature(qid=unique_id,
                                  example_index=example_index,
                                  doc_span_index=doc_span_index,
                                  tokens=tokens,
                                  token_to_orig_map=token_to_orig_map,
                                  token_is_max_context=token_is_max_context,
                                  src_ids=token_ids,
                                  start_position=start_position,
                                  end_position=end_position,
                                  cal_loss=1)
                features.append(feature)
                features_each.append(feature)
                if example_index % 1000 == 0:
                    print("processing {} examples".format(example_index),
                          flush=True)

                unique_id += 1
            # Repeat
            if self.repeat_input:
                features_each_repeat = features_each
                features_each = list(
                    map(lambda x: x._replace(cla_loss=0), features_each))
                features_each += features_each_repeat

            self.features_all.append(features_each)

        return features

    def _preprocess_data(self):
        # Construct examples
        self.examples = self._convert_qa_to_examples()
        # Construct features
        self.features = self._convert_example_to_feature(self.examples)

    def get_num_examples(self):
        if not self.features_all:
            self._preprocess_data()
        return len(sum(self.features_all, []))

    def _improve_answer_span(self, doc_tokens, input_start, input_end,
                             orig_answer_text):
        """Improve answer span"""
        tok_answer_text = " ".join(self.tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check is max context"""
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                break
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context,
                        num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index
                if best_span_index > cur_span_index:
                    return False

        return cur_span_index == best_span_index

    def _pad_batch_records(self, batch_records, gather_idx=[]):
        """Pad batch data"""
        batch_token_ids = [record.src_ids for record in batch_records]

        if self.mode == "train":
            batch_start_position = [
                record.start_position for record in batch_records
            ]
            batch_end_position = [
                record.end_position for record in batch_records
            ]
            batch_start_position = np.array(batch_start_position).astype(
                "int64").reshape([-1, 1])
            batch_end_position = np.array(batch_end_position).astype(
                "int64").reshape([-1, 1])
        else:
            batch_size = len(batch_token_ids)
            batch_start_position = np.zeros(shape=[batch_size, 1],
                                            dtype="int64")
            batch_end_position = np.zeros(shape=[batch_size, 1], dtype="int64")

        batch_qids = [record.qid for record in batch_records]
        batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])

        if gather_idx:
            batch_gather_idx = np.array(gather_idx).astype("int64").reshape(
                [-1, 1])
            need_cal_loss = np.array([1]).astype("int64")
        else:
            batch_gather_idx = np.array(list(range(
                len(batch_records)))).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([0]).astype("int64")

        # padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids,
            pad_idx=self.tokenizer.pad_token_id,
            pad_max_len=self.max_seq_length,
            return_input_mask=True)
        padded_task_ids = np.zeros_like(padded_token_ids, dtype="int64")
        padded_position_ids = get_related_pos(padded_task_ids,
                                              self.max_seq_length,
                                              self.memory_len)

        return_list = [
            padded_token_ids, padded_position_ids, padded_task_ids, input_mask,
            batch_start_position, batch_end_position, batch_qids,
            batch_gather_idx, need_cal_loss
        ]

        return return_list

    def _create_instances(self):
        """Generate batch records"""
        pre_batch_list = []
        insert_idx = []
        for qid, features in enumerate(self.features_all):
            if self._cnt_list(
                    pre_batch_list) < self.batch_size * self.trainer_num:
                if insert_idx:
                    pre_batch_list[insert_idx[0]] = features
                    insert_idx.pop(0)
                else:
                    pre_batch_list.append(features)
            if self._cnt_list(
                    pre_batch_list) == self.batch_size * self.trainer_num:
                assert self._cnt_list(pre_batch_list) == len(
                    pre_batch_list), "the two value must be equal"
                assert not insert_idx, "the insert_idx must be null"
                sample_batch = self._get_samples(pre_batch_list)

                for idx, lit in enumerate(pre_batch_list):
                    if not lit:
                        insert_idx.append(idx)
                for batch_records in self._prepare_batch_data(sample_batch):
                    yield batch_records

        if self.mode != "train":
            if self._cnt_list(pre_batch_list):
                pre_batch_list += [
                    [] for _ in range(self.batch_size * self.trainer_num -
                                      self._cnt_list(pre_batch_list))
                ]
                sample_batch = self._get_samples(pre_batch_list, is_last=True)
                for batch_records in self._prepare_batch_data(sample_batch):
                    yield batch_records


class MCQIterator(MRCIterator):
    """
    Multiple choice question iterator. 
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 tokenizer,
                 trainer_num,
                 trainer_id,
                 max_seq_length=512,
                 memory_len=128,
                 repeat_input=False,
                 in_tokens=False,
                 mode="train",
                 random_seed=None,
                 doc_stride=128,
                 max_query_length=64,
                 choice_num=4):
        self.choice_num = choice_num
        super(MCQIterator,
              self).__init__(dataset, batch_size, tokenizer, trainer_num,
                             trainer_id, max_seq_length, memory_len,
                             repeat_input, in_tokens, mode, random_seed)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        tokens_a = list(tokens_a)
        tokens_b = list(tokens_b)
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def _convert_qa_to_examples(self):
        Example = namedtuple(
            'Example', ['qas_id', 'context', 'question', 'choice', 'label'])
        examples = []
        for qas_id, qa in enumerate(self.dataset):
            context = '\n'.join(qa['context']).lower()
            question = qa['question'].lower()
            choice = [c.lower() for c in qa['choice']]
            # pad empty choice
            for k in range(len(choice), self.choice_num):
                choice.append('')
            label = qa['label']

            example = Example(qas_id=qas_id,
                              context=context,
                              question=question,
                              choice=choice,
                              label=label)
            examples.append(example)
        return examples

    def _convert_example_to_feature(self, examples):
        Feature = namedtuple(
            'Feature', ['qid', 'src_ids', 'segment_ids', 'label', 'cal_loss'])
        features = []
        self.features_all = []
        pad_token_id = self.tokenizer.pad_token_id
        for (ex_index, example) in enumerate(examples):
            context_tokens = self.tokenizer.tokenize(example.context)
            question_tokens = self.tokenizer.tokenize(example.question)
            choice_tokens_lst = [
                self.tokenizer.tokenize(choice) for choice in example.choice
            ]
            # nums = 4
            question_choice_pairs = \
                [self._truncate_seq_pair(question_tokens, choice_tokens, self.max_query_length - 2)
                for choice_tokens in choice_tokens_lst]
            total_qc_num = sum([(len(q) + len(c))
                                for q, c in question_choice_pairs])
            max_tokens_for_doc = self.max_seq_length - total_qc_num - 4
            _DocSpan = namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0

            while start_offset < len(context_tokens):
                length = len(context_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(context_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            features_each = []
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                qa_features = []
                for q_tokens, c_tokens in question_choice_pairs:
                    segment_tokens = ['[CLS]']
                    token_type_ids = [0]

                    segment_tokens += context_tokens[doc_span.
                                                     start:doc_span.start +
                                                     doc_span.length]
                    token_type_ids += [0] * doc_span.length

                    segment_tokens += ['[SEP]']
                    token_type_ids += [0]

                    segment_tokens += q_tokens
                    token_type_ids += [1] * len(q_tokens)

                    segment_tokens += ['[SEP]']
                    token_type_ids += [1]

                    segment_tokens += c_tokens
                    token_type_ids += [1] * len(c_tokens)

                    segment_tokens += ['[SEP]']
                    token_type_ids += [1]

                    input_ids = self.tokenizer.convert_tokens_to_ids(
                        segment_tokens)
                    feature = Feature(qid=example.qas_id,
                                      label=example.label,
                                      src_ids=input_ids,
                                      segment_ids=token_type_ids,
                                      cal_loss=1)
                    qa_features.append(feature)

                features.append(qa_features)
                features_each.append(qa_features)

            # Repeat
            if self.repeat_input:
                features_each_repeat = features_each
                features_each = list(
                    map(lambda x: x._replace(cla_loss=0), features_each))
                features_each += features_each_repeat

            self.features_all.append(features_each)

        return features

    def _pad_batch_records(self, batch_records, gather_idx=[]):
        batch_token_ids = [[record.src_ids for record in records]
                           for records in batch_records]
        if batch_records[0][0].label is not None:
            batch_labels = [[record.label for record in records]
                            for records in batch_records]
            batch_labels = np.array(batch_labels).astype("int64").reshape(
                [-1, 1])
        else:
            batch_labels = np.array([]).astype("int64").reshape([-1, 1])
        # Qid
        batch_qids = [[record.qid for record in records]
                      for records in batch_records]
        batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])

        if gather_idx:
            batch_gather_idx = np.array(gather_idx).astype("int64").reshape(
                [-1, 1])
            need_cal_loss = np.array([1]).astype("int64")
        else:
            batch_gather_idx = np.array(list(range(
                len(batch_records)))).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([0]).astype("int64")

        batch_task_ids = [[record.segment_ids for record in records]
                          for records in batch_records]

        # Padding
        batch_padded_token_ids = []
        batch_input_mask = []
        batch_padded_task_ids = []
        batch_padded_position_ids = []
        batch_size = len(batch_token_ids)
        for i in range(batch_size):
            padded_token_ids, input_mask = pad_batch_data(
                batch_token_ids[i],
                pad_idx=self.tokenizer.pad_token_id,
                pad_max_len=self.max_seq_length,
                return_input_mask=True)
            padded_task_ids = pad_batch_data(
                batch_task_ids[i],
                pad_idx=self.tokenizer.pad_token_id,
                pad_max_len=self.max_seq_length)

            padded_position_ids = get_related_pos(padded_task_ids,
                                                  self.max_seq_length,
                                                  self.memory_len)

            batch_padded_token_ids.append(padded_token_ids)
            batch_input_mask.append(input_mask)
            batch_padded_task_ids.append(padded_task_ids)
            batch_padded_position_ids.append(padded_position_ids)

        batch_padded_token_ids = np.array(batch_padded_token_ids).astype(
            "int64").reshape([batch_size * self.choice_num, -1, 1])
        batch_padded_position_ids = np.array(batch_padded_position_ids).astype(
            "int64").reshape([batch_size * self.choice_num, -1, 1])
        batch_padded_task_ids = np.array(batch_padded_task_ids).astype(
            "int64").reshape([batch_size * self.choice_num, -1, 1])
        batch_input_mask = np.array(batch_input_mask).astype("float32").reshape(
            [batch_size * self.choice_num, -1, 1])

        return_list = [
            batch_padded_token_ids, batch_padded_position_ids,
            batch_padded_task_ids, batch_input_mask, batch_labels, batch_qids,
            batch_gather_idx, need_cal_loss
        ]
        return return_list

    def _prepare_batch_data(self, examples_list):
        batch_records, max_len, gather_idx = [], 0, []
        real_batch_size = self.batch_size * self.choice_num
        index = 0
        for examples in examples_list:
            records = []
            gather_idx_candidate = []
            for example in examples:
                if example.cal_loss == 1:
                    gather_idx_candidate.append(index % real_batch_size)
                max_len = max(max_len, len(example.src_ids))
                records.append(example)
                index += 1

            if self.in_tokens:
                to_append = (len(batch_records) +
                             1) * self.choice_num * max_len <= self.batch_size
            else:
                to_append = len(batch_records) < self.batch_size
            if to_append:
                batch_records.append(records)
                gather_idx += gather_idx_candidate
            else:
                yield self._pad_batch_records(batch_records, gather_idx)
                batch_records, max_len = [records], max(
                    len(record.src_ids) for record in records)
                start_index = index - len(records) + 1
                gather_idx = gather_idx_candidate
        if len(batch_records) > 0:
            yield self._pad_batch_records(batch_records, gather_idx)

    def _get_samples(self, pre_batch_list, is_last=False):
        if is_last:
            # Pad batch
            len_doc = [[len(doc) for doc in doc_list]
                       for doc_list in pre_batch_list]
            len_doc = list(itertools.chain(*len_doc))
            max_len_idx = len_doc.index(max(len_doc))
            doc_idx = max_len_idx % self.choice_num
            doc_list_idx = max_len_idx // self.choice_num
            dirty_sample = pre_batch_list[doc_list_idx][doc_idx][-1]._replace(
                cal_loss=0)
            for sample_list in pre_batch_list:
                for samples in sample_list:
                    samples.extend([dirty_sample] *
                                   (max(len_doc) - len(samples)))
        samples = []
        min_len = min([len(doc) for doc in pre_batch_list])
        for cnt in range(min_len):
            for batch_idx in range(self.batch_size * self.trainer_num):
                sample = pre_batch_list[batch_idx][cnt]
                samples.append(sample)

        for idx in range(len(pre_batch_list)):
            pre_batch_list[idx] = pre_batch_list[idx][min_len:]
        return samples


class SemanticMatchingIterator(MRCIterator):

    def _convert_qa_to_examples(self):
        Example = namedtuple('Example',
                             ['qid', 'text_a', 'text_b', 'text_c', 'label'])
        examples = []
        for qid, qa in enumerate(self.dataset):
            text_a, text_b, text_c = list(
                map(lambda x: x.replace('\n', '').strip(),
                    [qa["text_a"], qa["text_b"], qa["text_c"]]))

            example = Example(qid=qid,
                              text_a=text_a,
                              text_b=text_b,
                              text_c=text_c,
                              label=qa["label"])
            examples += [example]
        return examples

    def _create_tokens_and_type_id(self, text_a_tokens, text_b_tokens, start,
                                   length):
        tokens = ['[CLS]'] + text_a_tokens[start:start + length] + [
            '[SEP]'
        ] + text_b_tokens[start:start + length] + ['[SEP]']
        token_type_ids = [0] + [0] * (length + 1) + [1] * (length + 1)
        return tokens, token_type_ids

    def _convert_example_to_feature(self, examples):
        Feature = namedtuple('Feature', [
            'qid', 'src_ids', 'segment_ids', 'pair_src_ids', 'pair_segment_ids',
            'label', 'cal_loss'
        ])
        features = []
        self.features_all = []
        pad_token_id = self.tokenizer.pad_token_id
        for (ex_index, example) in enumerate(examples):
            text_a_tokens = self.tokenizer.tokenize(example.text_a)
            text_b_tokens = self.tokenizer.tokenize(example.text_b)
            text_c_tokens = self.tokenizer.tokenize(example.text_c)
            a_len, b_len, c_len = list(
                map(lambda x: len(x),
                    [text_a_tokens, text_b_tokens, text_c_tokens]))

            # Align 3 text
            min_text_len = min([a_len, b_len, c_len])
            text_a_tokens = text_a_tokens[:min_text_len]
            text_b_tokens = text_b_tokens[:min_text_len]
            text_c_tokens = text_c_tokens[:min_text_len]

            _DocSpan = namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0

            max_tokens_for_doc = (self.max_seq_length - 3) // 2

            while start_offset < len(text_a_tokens):
                length = len(text_a_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(text_a_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            features_each = []
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens1, token_type_ids1 = self._create_tokens_and_type_id(
                    text_a_tokens, text_b_tokens, doc_span.start,
                    doc_span.length)
                tokens2, token_type_ids2 = self._create_tokens_and_type_id(
                    text_a_tokens, text_c_tokens, doc_span.start,
                    doc_span.length)

                input_ids1 = self.tokenizer.convert_tokens_to_ids(tokens1)
                input_ids2 = self.tokenizer.convert_tokens_to_ids(tokens2)
                feature = Feature(qid=example.qid,
                                  label=example.label,
                                  src_ids=input_ids1,
                                  segment_ids=token_type_ids1,
                                  pair_src_ids=input_ids2,
                                  pair_segment_ids=token_type_ids2,
                                  cal_loss=1)

                features.append(feature)
                features_each.append(feature)

            # Repeat
            if self.repeat_input:
                features_each_repeat = features_each
                features_each = list(
                    map(lambda x: x._replace(cla_loss=0), features_each))
                features_each += features_each_repeat

            self.features_all.append(features_each)

        return features

    def _create_pad_ids(self, batch_records, prefix=""):
        src_ids = prefix + "src_ids"
        segment_ids = prefix + "segment_ids"
        batch_token_ids = [getattr(record, src_ids) for record in batch_records]
        batch_task_ids = [
            getattr(record, segment_ids) for record in batch_records
        ]

        # Padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids,
            pad_idx=self.tokenizer.pad_token_id,
            pad_max_len=self.max_seq_length,
            return_input_mask=True)
        padded_task_ids = pad_batch_data(batch_task_ids,
                                         pad_idx=self.tokenizer.pad_token_id,
                                         pad_max_len=self.max_seq_length)

        padded_position_ids = get_related_pos(padded_task_ids,
                                              self.max_seq_length,
                                              self.memory_len)

        return [
            padded_token_ids, padded_position_ids, padded_task_ids, input_mask
        ]

    def _pad_batch_records(self, batch_records, gather_idx=[]):
        if batch_records[0].label is not None:
            batch_labels = [record.label for record in batch_records]
            batch_labels = np.array(batch_labels).astype("int64").reshape(
                [-1, 1])
        else:
            batch_labels = np.array([]).astype("int64").reshape([-1, 1])
        # Qid
        batch_qids = [record.qid for record in batch_records]
        batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])

        if gather_idx:
            batch_gather_idx = np.array(gather_idx).astype("int64").reshape(
                [-1, 1])
            need_cal_loss = np.array([1]).astype("int64")
        else:
            batch_gather_idx = np.array(list(range(
                len(batch_records)))).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([0]).astype("int64")

        return_list = self._create_pad_ids(batch_records) \
                    + self._create_pad_ids(batch_records, "pair_") \
                    + [batch_labels, batch_qids, batch_gather_idx, need_cal_loss]
        return return_list


class SequenceLabelingIterator(ClassifierIterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 tokenizer,
                 trainer_num,
                 trainer_id,
                 max_seq_length=512,
                 memory_len=128,
                 repeat_input=False,
                 in_tokens=False,
                 mode="train",
                 random_seed=None,
                 no_entity_id=-1):
        super(SequenceLabelingIterator, self).__init__(dataset,
                                                       batch_size,
                                                       tokenizer,
                                                       trainer_num,
                                                       trainer_id,
                                                       max_seq_length,
                                                       memory_len,
                                                       repeat_input,
                                                       in_tokens,
                                                       mode,
                                                       random_seed,
                                                       preprocess_text_fn=None)
        self.no_entity_id = no_entity_id

    def _convert_to_features(self, example, qid):
        """
        Convert example to features fed into model
        """
        tokens = example['tokens']
        label = example["labels"]
        doc_spans = []
        _DocSpan = namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        max_tokens_for_doc = self.max_seq_length - 2
        while start_offset < len(tokens):
            length = len(tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(tokens):
                break
            start_offset += min(length, self.memory_len)

        features = []
        Feature = namedtuple("Feature",
                             ["src_ids", "label_ids", "qid", "cal_loss"])
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            curr_tokens = ["[CLS]"] + tokens[doc_span.start:doc_span.start +
                                             doc_span.length] + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(curr_tokens)
            label = [self.no_entity_id
                     ] + label[doc_span.start:doc_span.start +
                               doc_span.length] + [self.no_entity_id]

            features.append(
                Feature(src_ids=token_ids, label_ids=label, qid=qid,
                        cal_loss=1))

        if self.repeat_input:
            features_repeat = features
            features = list(map(lambda x: x._replace(cal_loss=0), features))
            features = features + features_repeat
        return features

    def _pad_batch_records(self, batch_records, gather_idx=[]):
        batch_token_ids = [record.src_ids for record in batch_records]
        batch_length = [len(record.src_ids) for record in batch_records]
        batch_length = np.array(batch_length).astype("int64").reshape([-1, 1])

        if batch_records[0].label_ids is not None:
            batch_labels = [record.label_ids for record in batch_records]
        else:
            batch_labels = np.array([]).astype("int64").reshape([-1, 1])
        # Qid
        if batch_records[-1].qid is not None:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])
        else:
            batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        if gather_idx:
            batch_gather_idx = np.array(gather_idx).astype("int64").reshape(
                [-1, 1])
            need_cal_loss = np.array([1]).astype("int64")
        else:
            batch_gather_idx = np.array(list(range(
                len(batch_records)))).astype("int64").reshape([-1, 1])
            need_cal_loss = np.array([0]).astype("int64")
        # Padding
        padded_token_ids, input_mask = pad_batch_data(
            batch_token_ids,
            pad_idx=self.tokenizer.pad_token_id,
            pad_max_len=self.max_seq_length,
            return_input_mask=True)
        if batch_records[0].label_ids is not None:
            padded_batch_labels = pad_batch_data(
                batch_labels,
                pad_idx=self.no_entity_id,
                pad_max_len=self.max_seq_length)
        padded_task_ids = np.zeros_like(padded_token_ids, dtype="int64")
        padded_position_ids = get_related_pos(padded_token_ids, \
            self.max_seq_length, self.memory_len)

        return_list = [
            padded_token_ids, padded_position_ids, padded_task_ids, input_mask,
            padded_batch_labels, batch_length, batch_qids, batch_gather_idx,
            need_cal_loss
        ]
        return return_list
