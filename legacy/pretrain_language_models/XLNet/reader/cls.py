"""this file is adapted from https://github.com/zihangdai/xlnet"""

import io
import os
import types
import csv
import numpy as np
import sentencepiece as spm

from classifier_utils import PaddingInputExample
from classifier_utils import convert_single_example
from prepro_utils import preprocess_text, encode_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.max_seq_length = args.max_seq_length
        self.uncased = args.uncased
        np.random.seed(args.random_seed)

        sp = spm.SentencePieceProcessor()
        sp.Load(args.spiece_model_file)

        def tokenize_fn(text):
            text = preprocess_text(text, lower=self.uncased)
            return encode_ids(sp, text)

        self.tokenize_fn = tokenize_fn

        self.current_train_example = -1
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        self.current_train_epoch = -1

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_example(self, index, example, labels, max_seq_length,
                        tokenize_fn):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        feature = convert_single_example(index, example, labels, max_seq_length,
                                         tokenize_fn)
        return feature

    def generate_instance(self, feature):
        """
        generate instance with given feature

        Args:
            feature: InputFeatures(object). A single set of features of data.
        """
        return [
            feature.input_ids, feature.segment_ids, input_pos, feature.label_id
        ]

    def prepare_batch_data(self, batch_data, is_regression):
        """Generate numpy tensors"""
        input_ids = np.expand_dims(
            np.array([inst[0] for inst in batch_data]).astype('int64'), axis=-1)
        input_mask = np.array(
            [inst[1] for inst in batch_data]).astype('float32')
        segment_ids = np.array([inst[2] for inst in batch_data]).astype('int64')
        labels = np.expand_dims(
            np.array([inst[3] for inst in batch_data]).astype(
                'int64' if not is_regression else 'float32'),
            axis=-1)
        is_real_example = np.array(
            [inst[4] for inst in batch_data]).astype('int64')

        return [input_ids, input_mask, segment_ids, labels, is_real_example]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="utf8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) == 0: continue
                lines.append(line)
            return lines

    def get_num_examples(self, phase):
        """Get number of examples for train, dev or test."""
        if phase not in ['train', 'dev', 'test']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test'].")
        return self.num_examples[phase]

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_train_example, self.current_train_epoch

    def data_generator(self,
                       batch_size,
                       is_regression,
                       phase='train',
                       epoch=1,
                       dev_count=1,
                       shuffle=True):
        """
        Generate data for train, dev or test.
    
        Args:
          batch_size: int. The batch size of generated data.
          phase: string. The phase for which to generate data.
          epoch: int. Total epoches to generate data.
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
            label_list = self.get_labels() if not is_regression else None
            for epoch_index in range(epoch):
                if shuffle:
                    np.random.shuffle(examples)
                if phase == 'train':
                    self.current_train_epoch = epoch_index
                for (index, example) in enumerate(examples):
                    if phase == 'train':
                        self.current_train_example = index + 1
                    feature = convert_single_example(index, example, label_list,
                                                     self.max_seq_length,
                                                     self.tokenize_fn)
                    instance = [
                        feature.input_ids, feature.input_mask,
                        feature.segment_ids, feature.label_id,
                        feature.is_real_example
                    ]
                    yield instance

        def batch_reader(reader, batch_size):
            batch = []
            for instance in reader():
                if len(batch) < batch_size:
                    batch.append(instance)
                else:
                    yield batch
                    batch = [instance]

            if len(batch) > 0:
                yield batch

        def wrapper():
            all_dev_batches = []
            for batch_data in batch_reader(instance_reader, batch_size):
                batch_data = self.prepare_batch_data(batch_data, is_regression)
                if len(all_dev_batches) < dev_count:
                    all_dev_batches.append(batch_data)

                if len(all_dev_batches) == dev_count:
                    for batch in all_dev_batches:
                        yield batch
                    all_dev_batches = []

        return wrapper


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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


class GLUEProcessor(DataProcessor):
    def __init__(self, args):
        super(GLUEProcessor, self).__init__(args)
        self.train_file = "train.tsv"
        self.dev_file = "dev.tsv"
        self.test_file = "test.tsv"
        self.label_column = None
        self.text_a_column = None
        self.text_b_column = None
        self.contains_header = True
        self.test_text_a_column = None
        self.test_text_b_column = None
        self.test_contains_header = True

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.train_file)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.dev_file)), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        if self.test_text_a_column is None:
            self.test_text_a_column = self.text_a_column
        if self.test_text_b_column is None:
            self.test_text_b_column = self.text_b_column

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and self.contains_header and set_type != "test":
                continue
            if i == 0 and self.test_contains_header and set_type == "test":
                continue
            guid = "%s-%s" % (set_type, i)

            a_column = (self.text_a_column
                        if set_type != "test" else self.test_text_a_column)
            b_column = (self.text_b_column
                        if set_type != "test" else self.test_text_b_column)

            # there are some incomplete lines in QNLI
            if len(line) <= a_column:
                tf.logging.warning('Incomplete line, ignored.')
                continue
            text_a = line[a_column]

            if b_column is not None:
                if len(line) <= b_column:
                    tf.logging.warning('Incomplete line, ignored.')
                    continue
                text_b = line[b_column]
            else:
                text_b = None

            if set_type == "test":
                label = self.get_labels()[0]
            else:
                if len(line) <= self.label_column:
                    tf.logging.warning('Incomplete line, ignored.')
                    continue
                label = line[self.label_column]
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Yelp5Processor(DataProcessor):
    def __init__(self, args):
        super(Yelp5Processor, self).__init__(args)

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.csv"))

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5"]

    def _create_examples(self, input_file):
        """Creates examples for the training and dev sets."""
        examples = []
        with tf.gfile.Open(input_file) as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):

                label = line[0]
                text_a = line[1].replace('""', '"').replace('\\"', '"')
                examples.append(
                    InputExample(
                        guid=str(i), text_a=text_a, text_b=None, label=label))
        return examples


class ImdbProcessor(DataProcessor):
    def __init__(self, args):
        super(ImdbProcessor, self).__init__(args)

    def get_labels(self):
        return ["neg", "pos"]

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test"))

    def _create_examples(self, data_dir):
        examples = []
        for label in ["neg", "pos"]:
            cur_dir = os.path.join(data_dir, label)
            for filename in os.listdir(cur_dir):
                if not filename.endswith("txt"): continue

                path = os.path.join(cur_dir, filename)
                with io.open(path, 'r', encoding='utf8') as f:
                    text = f.read().strip().replace("<br />", " ")
                examples.append(
                    InputExample(
                        guid="unused_id", text_a=text, text_b=None,
                        label=label))
        return examples


class MnliMatchedProcessor(GLUEProcessor):
    def __init__(self, args):
        super(MnliMatchedProcessor, self).__init__(args)
        self.dev_file = "dev_matched.tsv"
        self.test_file = "test_matched.tsv"
        self.label_column = -1
        self.text_a_column = 8
        self.text_b_column = 9

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]


class MnliMismatchedProcessor(MnliMatchedProcessor):
    def __init__(self, args):
        super(MnliMismatchedProcessor, self).__init__(args)
        self.dev_file = "dev_mismatched.tsv"
        self.test_file = "test_mismatched.tsv"


class StsbProcessor(GLUEProcessor):
    def __init__(self, args):
        super(StsbProcessor, self).__init__(args)
        self.label_column = 9
        self.text_a_column = 7
        self.text_b_column = 8

    def get_labels(self):
        return [0.0]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and self.contains_header and set_type != "test":
                continue
            if i == 0 and self.test_contains_header and set_type == "test":
                continue
            guid = "%s-%s" % (set_type, i)

            a_column = (self.text_a_column
                        if set_type != "test" else self.test_text_a_column)
            b_column = (self.text_b_column
                        if set_type != "test" else self.test_text_b_column)

            # there are some incomplete lines in QNLI
            if len(line) <= a_column:
                tf.logging.warning('Incomplete line, ignored.')
                continue
            text_a = line[a_column]

            if b_column is not None:
                if len(line) <= b_column:
                    tf.logging.warning('Incomplete line, ignored.')
                    continue
                text_b = line[b_column]
            else:
                text_b = None

            if set_type == "test":
                label = self.get_labels()[0]
            else:
                if len(line) <= self.label_column:
                    tf.logging.warning('Incomplete line, ignored.')
                    continue
                label = float(line[self.label_column])
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


if __name__ == '__main__':
    pass
