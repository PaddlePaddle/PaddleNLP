from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import logging
import os
import pickle
import random
import time
import numpy as np
from tqdm import tqdm

n_class = 4
reverse_order = False
sa_step = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class c3Processor(DataProcessor):
    def __init__(self, data_dir):
        self.D = [[], [], []]
        self.data_dir = data_dir

        for sid in range(3):
            data = []
            if(sid<2):
                for subtask in ["d", "m"]:
                    with open(self.data_dir + "/" + subtask + "-" + ["train.json", "dev.json", "test.json"][sid],
                            "r", encoding="utf8") as f:
                        data += json.load(f)
            else:

                with open(self.data_dir + "/" + "test1.0.json",
                                "r", encoding="utf8") as f:
                    data += json.load(f)
            if sid == 0:
                random.shuffle(data)

            if(sid<2):
                for i in range(len(data)):
                    for j in range(len(data[i][1])):
                        d = [
                            '\n'.join(data[i][0]).lower(),
                            data[i][1][j]["question"].lower()
                        ]
                        for k in range(len(data[i][1][j]["choice"])):
                            d += [data[i][1][j]["choice"][k].lower()]
                        for k in range(len(data[i][1][j]["choice"]), 4):
                            d += ['无效答案']  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                        d += [data[i][1][j]["answer"].lower()]
                        self.D[sid] += [d]
            else:
                for i in range(len(data)):
                    for j in range(len(data[i][1])):
                        d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                        for k in range(len(data[i][1][j]["choice"])):
                            d += [data[i][1][j]["choice"][k].lower()]
                        for k in range(len(data[i][1][j]["choice"]), 4):
                            d += ['无效答案']  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                        d += [data[i][1][j]["choice"][0].lower()]
                        self.D[sid] += [d]

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self.D[0], "train")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self.D[2], "test")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        cache_dir = os.path.join(self.data_dir, set_type + '_examples.pkl')
        if os.path.exists(cache_dir):
            examples = pickle.load(open(cache_dir, 'rb'))
        else:
            examples = []
            for (i, d) in enumerate(data):
                answer = -1
                # 这里data[i]有6个元素，0是context，1是问题，2~5是choice，6是答案
                for k in range(4):
                    if data[i][2 + k] == data[i][6]:
                        answer = str(k)

                # label = tokenization.convert_to_unicode(answer)
                label = answer

                for k in range(4):
                    guid = "%s-%s-%s" % (set_type, i, k)
                    # text_a = tokenization.convert_to_unicode(data[i][0])
                    # text_b = tokenization.convert_to_unicode(data[i][k + 2])
                    # text_c = tokenization.convert_to_unicode(data[i][1])
                    text_a = data[i][0]
                    text_b = data[i][k + 2]
                    text_c = data[i][1]
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, text_c=text_c))

            with open(cache_dir, 'wb') as w:
                pickle.dump(examples, w)

        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = [[]]
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)

        tokens_c = tokenizer.tokenize(example.text_c)

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_c + ["[SEP]"] + tokens_b

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            # logger.info("tokens: %s" % " ".join(
            #     [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
        if len(features[-1]) == n_class:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features


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


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)