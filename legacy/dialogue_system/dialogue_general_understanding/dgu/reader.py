# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""data reader"""
import os
import io
import csv
import sys
import types
import numpy as np

from dgu import tokenization
from dgu.batching import prepare_batch_data

if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding('utf-8')


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self,
                 data_dir,
                 vocab_path,
                 max_seq_len,
                 do_lower_case,
                 in_tokens,
                 task_name,
                 random_seed=None):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.in_tokens = in_tokens

        np.random.seed(random_seed)

        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        self.task_name = task_name

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    @staticmethod
    def get_labels():
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_example(self, index, example, labels, max_seq_len, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        feature = convert_single_example(index, example, labels, max_seq_len,
                                         tokenizer, self.task_name)
        return feature

    def generate_instance(self, feature):
        """
        generate instance with given feature

        Args:
            feature: InputFeatures(object). A single set of features of data.
        """
        input_pos = list(range(len(feature.input_ids)))
        return [
            feature.input_ids, feature.segment_ids, input_pos, feature.label_id
        ]

    def generate_batch_data(self,
                            batch_data,
                            max_len,
                            total_token_num,
                            voc_size=-1,
                            mask_id=-1,
                            return_input_mask=True,
                            return_max_len=False,
                            return_num_token=False):
        """generate batch data"""
        return prepare_batch_data(
            self.task_name,
            batch_data,
            max_len,
            total_token_num,
            voc_size=-1,
            pad_id=self.vocab["[PAD]"],
            cls_id=self.vocab["[CLS]"],
            sep_id=self.vocab["[SEP]"],
            mask_id=-1,
            return_input_mask=True,
            return_max_len=False,
            return_num_token=False)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        f = io.open(input_file, "r", encoding="utf8")
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

    def get_num_examples(self, phase):
        """Get number of examples for train, dev or test."""
        if phase not in ['train', 'dev', 'test']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test'].")
        return self.num_examples[phase]

    def data_generator(self, batch_size, phase='train', shuffle=False):
        """
        Generate data for train, dev or test.
    
        Args:
          batch_size: int. The batch size of generated data.
          phase: string. The phase for which to generate data.
          shuffle: bool. Whether to shuffle examples.
        """
        if phase == 'train':
            examples = self.get_train_examples(self.data_dir)
            self.num_examples['train'] = len(examples)
        elif phase == 'dev':
            examples = self.get_dev_examples(self.data_dir)
            self.num_examples['dev'] = len(examples)
        elif phase == 'test':
            examples = self.get_test_examples(self.data_dir)
            self.num_examples['test'] = len(examples)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test'].")

        def instance_reader():
            """generate instance data"""
            if shuffle:
                np.random.shuffle(examples)
            for (index, example) in enumerate(examples):
                feature = self.convert_example(index, example,
                                               self.get_labels(),
                                               self.max_seq_len, self.tokenizer)
                instance = self.generate_instance(feature)
                yield instance

        def batch_reader(reader, batch_size, in_tokens):
            """read batch data"""
            batch, total_token_num, max_len = [], 0, 0
            for instance in reader():
                token_ids, sent_ids, pos_ids, label = instance[:4]
                max_len = max(max_len, len(token_ids))
                if in_tokens:
                    to_append = (len(batch) + 1) * max_len <= batch_size
                else:
                    to_append = len(batch) < batch_size
                if to_append:
                    batch.append(instance)
                    total_token_num += len(token_ids)
                else:
                    yield batch, total_token_num
                    batch, total_token_num, max_len = [instance], len(
                        token_ids), len(token_ids)

            if len(batch) > 0:
                yield batch, total_token_num

        def wrapper():
            """yield batch data to network"""
            for batch_data, total_token_num in batch_reader(
                    instance_reader, batch_size, self.in_tokens):
                if self.in_tokens:
                    max_seq = -1
                else:
                    max_seq = self.max_seq_len
                batch_data = self.generate_batch_data(
                    batch_data,
                    max_seq,
                    total_token_num,
                    voc_size=-1,
                    mask_id=-1,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)
                yield batch_data

        return wrapper


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class UDCProcessor(DataProcessor):
    """Processor for the UDC data set."""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        print(
            "UDC dataset is too big, loading data spent a long time, please wait patiently.................."
        )
        for (i, line) in enumerate(lines):
            if len(line) < 3:
                print("data format error: %s" % "\t".join(line))
                print(
                    "data row contains at least three parts: label\tconv1\t.....\tresponse"
                )
                continue
            guid = "%s-%d" % (set_type, i)
            text_a = "\t".join(line[1:-1])
            text_a = tokenization.convert_to_unicode(text_a)
            text_a = text_a.split('\t')
            text_b = line[-1]
            text_b = tokenization.convert_to_unicode(text_b)
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "train.txt"))
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "dev.txt"))
        examples = self._create_examples(lines, "dev")
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "test.txt"))
        examples = self._create_examples(lines, "test")
        return examples

    @staticmethod
    def get_labels():
        """See base class."""
        return ["0", "1"]


class SWDAProcessor(DataProcessor):
    """Processor for the SWDA data set."""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = create_multi_turn_examples(lines, set_type)
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "train.txt"))
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "dev.txt"))
        examples = self._create_examples(lines, "dev")
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "test.txt"))
        examples = self._create_examples(lines, "test")
        return examples

    @staticmethod
    def get_labels():
        """See base class."""
        labels = range(42)
        labels = [str(label) for label in labels]
        return labels


class MRDAProcessor(DataProcessor):
    """Processor for the MRDA data set."""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = create_multi_turn_examples(lines, set_type)
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "train.txt"))
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "dev.txt"))
        examples = self._create_examples(lines, "dev")
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "test.txt"))
        examples = self._create_examples(lines, "test")
        return examples

    @staticmethod
    def get_labels():
        """See base class."""
        labels = range(42)
        labels = [str(label) for label in labels]
        return labels


class ATISSlotProcessor(DataProcessor):
    """Processor for the ATIS Slot data set."""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 2:
                print("data format error: %s" % "\t".join(line))
                print(
                    "data row contains two parts: conversation_content \t label1 label2 label3"
                )
                continue
            guid = "%s-%d" % (set_type, i)
            text_a = line[0]
            label = line[1]
            text_a = tokenization.convert_to_unicode(text_a)
            label_list = label.split()
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, label=label_list))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "train.txt"))
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "dev.txt"))
        examples = self._create_examples(lines, "dev")
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "test.txt"))
        examples = self._create_examples(lines, "test")
        return examples

    @staticmethod
    def get_labels():
        """See base class."""
        labels = range(130)
        labels = [str(label) for label in labels]
        return labels


class ATISIntentProcessor(DataProcessor):
    """Processor for the ATIS intent data set."""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 2:
                print("data format error: %s" % "\t".join(line))
                print(
                    "data row contains two parts: label \t conversation_content")
                continue
            guid = "%s-%d" % (set_type, i)
            text_a = line[1]
            text_a = tokenization.convert_to_unicode(text_a)
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "train.txt"))
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "dev.txt"))
        examples = self._create_examples(lines, "dev")
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "test.txt"))
        examples = self._create_examples(lines, "test")
        return examples

    @staticmethod
    def get_labels():
        """See base class."""
        labels = range(26)
        labels = [str(label) for label in labels]
        return labels


class DSTC2Processor(DataProcessor):
    """Processor for the DSTC2 data set."""

    def _create_turns(self, conv_example):
        """create multi turn dataset"""
        samples = []
        max_turns = 20
        for i in range(len(conv_example)):
            conv_turns = conv_example[max(i - max_turns, 0):i + 1]
            conv_info = "\1".join([sample[0] for sample in conv_turns])
            samples.append((conv_info.split('\1'), conv_example[i][1]))
        return samples

    def _create_examples(self, lines, set_type):
        """Creates examples for multi-turn dialogue sets."""
        examples = []
        conv_id = -1
        index = 0
        conv_example = []
        for (i, line) in enumerate(lines):
            if len(line) != 3:
                print("data format error: %s" % "\t".join(line))
                print(
                    "data row contains three parts: conversation_content \t question \1 answer \t state1 state2 state3......"
                )
                continue
            conv_no = line[0]
            text_a = line[1]
            label_list = line[2].split()
            if conv_no != conv_id and i != 0:
                samples = self._create_turns(conv_example)
                for sample in samples:
                    guid = "%s-%s" % (set_type, index)
                    index += 1
                    history = sample[0]
                    dst_label = sample[1]
                    examples.append(
                        InputExample(
                            guid=guid, text_a=history, label=dst_label))
                conv_example = []
                conv_id = conv_no
            if i == 0:
                conv_id = conv_no
            conv_example.append((text_a, label_list))
        if conv_example:
            samples = self._create_turns(conv_example)
            for sample in samples:
                guid = "%s-%s" % (set_type, index)
                index += 1
                history = sample[0]
                dst_label = sample[1]
                examples.append(
                    InputExample(
                        guid=guid, text_a=history, label=dst_label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "train.txt"))
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "dev.txt"))
        examples = self._create_examples(lines, "dev")
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "test.txt"))
        examples = self._create_examples(lines, "test")
        return examples

    @staticmethod
    def get_labels():
        """See base class."""
        labels = range(217)
        labels = [str(label) for label in labels]
        return labels


class MULTIWOZProcessor(DataProcessor):
    """Processor for the MULTIWOZ data set."""

    def _create_turns(self, conv_example):
        """create multi turn dataset"""
        samples = []
        max_turns = 2
        for i in range(len(conv_example)):
            prefix_turns = conv_example[max(i - max_turns, 0):i]
            conv_info = "\1".join([turn[0] for turn in prefix_turns])
            current_turns = conv_example[i][0]
            samples.append((conv_info.split('\1'), current_turns.split('\1'),
                            conv_example[i][1]))
        return samples

    def _create_examples(self, lines, set_type):
        """Creates examples for multi-turn dialogue sets."""
        examples = []
        conv_id = -1
        index = 0
        conv_example = []
        for (i, line) in enumerate(lines):
            conv_no = line[0]
            text_a = line[2]
            label_list = line[1].split()
            if conv_no != conv_id and i != 0:
                samples = self._create_turns(conv_example)
                for sample in samples:
                    guid = "%s-%s" % (set_type, index)
                    index += 1
                    history = sample[0]
                    current = sample[1]
                    dst_label = sample[2]
                    examples.append(
                        InputExample(
                            guid=guid,
                            text_a=history,
                            text_b=current,
                            label=dst_label))
                conv_example = []
                conv_id = conv_no
            if i == 0:
                conv_id = conv_no
            conv_example.append((text_a, label_list))
        if conv_example:
            samples = self._create_turns(conv_example)
            for sample in samples:
                guid = "%s-%s" % (set_type, index)
                index += 1
                history = sample[0]
                current = sample[1]
                dst_label = sample[2]
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=history,
                        text_b=current,
                        label=dst_label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "train.txt"))
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "dev.txt"))
        examples = self._create_examples(lines, "dev")
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = []
        lines = self._read_tsv(os.path.join(data_dir, "test.txt"))
        examples = self._create_examples(lines, "test")
        return examples

    @staticmethod
    def get_labels():
        """See base class."""
        labels = range(722)
        labels = [str(label) for label in labels]
        return labels


def create_dialogue_examples(conv):
    """Creates dialogue sample"""
    samples = []
    for i in range(len(conv)):
        cur_txt = "%s : %s" % (conv[i][2], conv[i][3])
        pre_txt = ["%s : %s" % (c[2], c[3]) for c in conv[max(0, i - 5):i]]
        suf_txt = [
            "%s : %s" % (c[2], c[3]) for c in conv[i + 1:min(len(conv), i + 3)]
        ]
        sample = [conv[i][1], pre_txt, cur_txt, suf_txt]
        samples.append(sample)
    return samples


def create_multi_turn_examples(lines, set_type):
    """Creates examples for multi-turn dialogue sets."""
    conv_id = -1
    examples = []
    conv_example = []
    index = 0
    for (i, line) in enumerate(lines):
        if len(line) != 4:
            print("data format error: %s" % "\t".join(line))
            print(
                "data row contains four parts: conversation_id \t label \t caller \t conversation_content"
            )
            continue
        tokens = line
        conv_no = tokens[0]
        if conv_no != conv_id and i != 0:
            samples = create_dialogue_examples(conv_example)
            for sample in samples:
                guid = "%s-%s" % (set_type, index)
                index += 1
                label = sample[0]
                text_a = sample[1]
                text_b = sample[2]
                text_c = sample[3]
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=text_a,
                        text_b=text_b,
                        text_c=text_c,
                        label=label))
            conv_example = []
            conv_id = conv_no
        if i == 0:
            conv_id = conv_no
        conv_example.append(tokens)
    if conv_example:
        samples = create_dialogue_examples(conv_example)
        for sample in samples:
            guid = "%s-%s" % (set_type, index)
            index += 1
            label = sample[0]
            text_a = sample[1]
            text_b = sample[2]
            text_c = sample[3]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    text_c=text_c,
                    label=label))
    return examples


def convert_tokens(tokens, sep_id, tokenizer):
    """Converts tokens to ids"""
    tokens_ids = []
    if not tokens:
        return tokens_ids
    if isinstance(tokens, list):
        for text in tokens:
            tok_text = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tok_text)
            tokens_ids.extend(ids)
            tokens_ids.append(sep_id)
        tokens_ids = tokens_ids[:-1]
    else:
        tok_text = tokenizer.tokenize(tokens)
        tokens_ids = tokenizer.convert_tokens_to_ids(tok_text)
    return tokens_ids


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, task_name):
    """Converts a single DA `InputExample` into a single `InputFeatures`."""
    label_map = {}
    SEP = 102
    CLS = 101

    if task_name == 'udc':
        INNER_SEP = 1
        limit_length = 60
    elif task_name == 'swda':
        INNER_SEP = 1
        limit_length = 50
    elif task_name == 'mrda':
        INNER_SEP = 1
        limit_length = 50
    elif task_name == 'atis_intent':
        INNER_SEP = -1
        limit_length = -1
    elif task_name == 'atis_slot':
        INNER_SEP = -1
        limit_length = -1
    elif task_name == 'dstc2':
        INNER_SEP = 1
        limit_length = -1
    elif task_name == 'dstc2_asr':
        INNER_SEP = 1
        limit_length = -1
    elif task_name == 'multi-woz':
        INNER_SEP = 1
        limit_length = 200
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = example.text_a
    tokens_b = example.text_b
    tokens_c = example.text_c

    tokens_a_ids = convert_tokens(tokens_a, INNER_SEP, tokenizer)
    tokens_b_ids = convert_tokens(tokens_b, INNER_SEP, tokenizer)
    tokens_c_ids = convert_tokens(tokens_c, INNER_SEP, tokenizer)

    if tokens_b_ids:
        tokens_b_ids = tokens_b_ids[:min(limit_length, len(tokens_b_ids))]
    else:
        if len(tokens_a_ids) > max_seq_length - 2:
            tokens_a_ids = tokens_a_ids[len(tokens_a_ids) - max_seq_length + 2:]
    if not tokens_c_ids:
        if len(tokens_a_ids) > max_seq_length - len(tokens_b_ids) - 3:
            tokens_a_ids = tokens_a_ids[len(tokens_a_ids) - max_seq_length +
                                        len(tokens_b_ids) + 3:]
    else:
        if len(tokens_a_ids) + len(tokens_b_ids) + len(
                tokens_c_ids) > max_seq_length - 4:
            left_num = max_seq_length - len(tokens_b_ids) - 4
            if len(tokens_a_ids) > len(tokens_c_ids):
                suffix_num = int(left_num / 2)
                tokens_c_ids = tokens_c_ids[:min(len(tokens_c_ids), suffix_num)]
                prefix_num = left_num - len(tokens_c_ids)
                tokens_a_ids = tokens_a_ids[max(
                    0, len(tokens_a_ids) - prefix_num):]
            else:
                if not tokens_a_ids:
                    tokens_c_ids = tokens_c_ids[max(
                        0, len(tokens_c_ids) - left_num):]
                else:
                    prefix_num = int(left_num / 2)
                    tokens_a_ids = tokens_a_ids[max(
                        0, len(tokens_a_ids) - prefix_num):]
                    suffix_num = left_num - len(tokens_a_ids)
                    tokens_c_ids = tokens_c_ids[:min(
                        len(tokens_c_ids), suffix_num)]

    input_ids = []
    segment_ids = []
    input_ids.append(CLS)
    segment_ids.append(0)
    input_ids.extend(tokens_a_ids)
    segment_ids.extend([0] * len(tokens_a_ids))
    input_ids.append(SEP)
    segment_ids.append(0)
    if tokens_b_ids:
        input_ids.extend(tokens_b_ids)
        segment_ids.extend([1] * len(tokens_b_ids))
        input_ids.append(SEP)
        segment_ids.append(1)
    if tokens_c_ids:
        input_ids.extend(tokens_c_ids)
        segment_ids.extend([0] * len(tokens_c_ids))
        input_ids.append(SEP)
        segment_ids.append(0)

    input_mask = [1] * len(input_ids)
    if task_name == 'atis_slot':
        label_id = [0] + [label_map[l] for l in example.label] + [0]
    elif task_name in ['dstc2', 'dstc2_asr', 'multi-woz']:
        label_id_enty = [label_map[l] for l in example.label]
        label_id = []
        for i in range(len(label_map)):
            if i in label_id_enty:
                label_id.append(1)
            else:
                label_id.append(0)
    else:
        label_id = label_map[example.label]

    if ex_index < 5:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label: %s (id = %s)" % (example.label, label_id))
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)

    return feature
