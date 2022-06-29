import os
import numpy as np
from typing import List

from paddle.io import Dataset

# The input data bigin with '[CLS]', using '[SEP]' split conversation content(
# Previous part, current part, following part, etc.). If there are multiple
# conversation in split part, using 'INNER_SEP' to further split.
INNER_SEP = '[unused0]'


def get_label_map(label_list):
    """ Create label maps """
    label_map = {}
    for (i, l) in enumerate(label_list):
        label_map[l] = i
    return label_map


class UDCv1(Dataset):
    """
    The UDCv1 dataset is using in task Dialogue Response Selection.
    The source dataset is UDCv1(Ubuntu Dialogue Corpus v1.0). See detail at
    http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/
    """
    MAX_LEN_OF_RESPONSE = 60
    LABEL_MAP = get_label_map(['0', '1'])

    def __init__(self, data_dir, mode='train'):
        super(UDCv1, self).__init__()
        self._data_dir = data_dir
        self._mode = mode
        self.read_data()

    def read_data(self):
        if self._mode == 'train':
            data_path = os.path.join(self._data_dir, 'train.txt')
        elif self._mode == 'dev':
            data_path = os.path.join(self._data_dir, 'dev.txt-small')
        elif self._mode == 'test':
            data_path = os.path.join(self._data_dir, 'test.txt')
        self.data = []
        with open(data_path, 'r', encoding='utf8') as fin:
            for line in fin:
                if not line:
                    continue
                arr = line.rstrip('\n').split('\t')
                if len(arr) < 3:
                    print('Data format error: %s' % '\t'.join(arr))
                    print(
                        'Data row contains at least three parts: label\tconversation1\t.....\tresponse.'
                    )
                    continue
                label = arr[0]
                text_a = arr[1:-1]
                text_b = arr[-1]
                self.data.append([label, text_a, text_b])

    @classmethod
    def get_label(cls, label):
        return cls.LABEL_MAP[label]

    @classmethod
    def num_classes(cls):
        return len(cls.LABEL_MAP)

    @classmethod
    def convert_example(cls, example, tokenizer, max_seq_length=512):
        """ Convert a glue example into necessary features. """

        def _truncate_and_concat(text_a: List[str], text_b: str, tokenizer,
                                 max_seq_length):
            tokens_b = tokenizer.tokenize(text_b)
            tokens_b = tokens_b[:min(cls.MAX_LEN_OF_RESPONSE, len(tokens_b))]
            tokens_a = []
            for text in text_a:
                tokens_a.extend(tokenizer.tokenize(text))
                tokens_a.append(INNER_SEP)
            tokens_a = tokens_a[:-1]
            if len(tokens_a) > max_seq_length - len(tokens_b) - 3:
                tokens_a = tokens_a[len(tokens_a) - max_seq_length +
                                    len(tokens_b) + 3:]
            tokens, segment_ids = [], []
            tokens.extend([tokenizer.cls_token] + tokens_a +
                          [tokenizer.sep_token])
            segment_ids.extend([0] * len(tokens))
            tokens.extend(tokens_b + [tokenizer.sep_token])
            segment_ids.extend([1] * (len(tokens_b) + 1))
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            return input_ids, segment_ids

        label, text_a, text_b = example
        label = np.array([cls.get_label(label)], dtype='int64')
        input_ids, segment_ids = _truncate_and_concat(text_a, text_b, tokenizer,
                                                      max_seq_length)
        return input_ids, segment_ids, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DSTC2(Dataset):
    """
    The dataset DSTC2 is using in task Dialogue State Tracking.
    The source dataset is DSTC2(Dialog State Tracking Challenges 2). See detail at
    https://github.com/matthen/dstc
    """
    LABEL_MAP = get_label_map([str(i) for i in range(217)])

    def __init__(self, data_dir, mode='train'):
        super(DSTC2, self).__init__()
        self._data_dir = data_dir
        self._mode = mode
        self.read_data()

    def read_data(self):

        def _concat_dialogues(examples):
            """concat multi turns dialogues"""
            new_examples = []
            max_turns = 20
            for i in range(len(examples)):
                multi_turns = examples[max(i - max_turns, 0):i + 1]
                new_qa = '\1'.join([example[0] for example in multi_turns])
                new_examples.append((new_qa.split('\1'), examples[i][1]))
            return new_examples

        if self._mode == 'train':
            data_path = os.path.join(self._data_dir, 'train.txt')
        elif self._mode == 'dev':
            data_path = os.path.join(self._data_dir, 'dev.txt')
        elif self._mode == 'test':
            data_path = os.path.join(self._data_dir, 'test.txt')
        self.data = []
        with open(data_path, 'r', encoding='utf8') as fin:
            pre_idx = -1
            examples = []
            for line in fin:
                if not line:
                    continue
                arr = line.rstrip('\n').split('\t')
                if len(arr) != 3:
                    print('Data format error: %s' % '\t'.join(arr))
                    print(
                        'Data row should contains three parts: id\tquestion\1answer\tlabel1 label2 ...'
                    )
                    continue
                idx = arr[0]
                qa = arr[1]
                label_list = arr[2].split()
                if idx != pre_idx:
                    if idx != 0:
                        examples = _concat_dialogues(examples)
                        self.data.extend(examples)
                        examples = []
                    pre_idx = idx
                examples.append((qa, label_list))
            if examples:
                examples = _concat_dialogues(examples)
                self.data.extend(examples)

    @classmethod
    def get_label(cls, label):
        return cls.LABEL_MAP[label]

    @classmethod
    def num_classes(cls):
        return len(cls.LABEL_MAP)

    @classmethod
    def convert_example(cls, example, tokenizer, max_seq_length=512):
        """ Convert a glue example into necessary features. """

        def _truncate_and_concat(texts: List[str], tokenizer, max_seq_length):
            tokens = []
            for text in texts:
                tokens.extend(tokenizer.tokenize(text))
                tokens.append(INNER_SEP)
            tokens = tokens[:-1]
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[len(tokens) - max_seq_length + 2:]
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            return input_ids, segment_ids

        texts, labels = example
        input_ids, segment_ids = _truncate_and_concat(texts, tokenizer,
                                                      max_seq_length)
        labels = [cls.get_label(l) for l in labels]
        label = np.zeros(cls.num_classes(), dtype='int64')
        for l in labels:
            label[l] = 1
        return input_ids, segment_ids, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ATIS_DSF(Dataset):
    """
    The dataset ATIS_DSF is using in task Dialogue Slot Filling.
    The source dataset is ATIS(Airline Travel Information System). See detail at
    https://www.kaggle.com/siddhadev/ms-cntk-atis
    """
    LABEL_MAP = get_label_map([str(i) for i in range(130)])

    def __init__(self, data_dir, mode='train'):
        super(ATIS_DSF, self).__init__()
        self._data_dir = data_dir
        self._mode = mode
        self.read_data()

    def read_data(self):
        if self._mode == 'train':
            data_path = os.path.join(self._data_dir, 'train.txt')
        elif self._mode == 'dev':
            data_path = os.path.join(self._data_dir, 'dev.txt')
        elif self._mode == 'test':
            data_path = os.path.join(self._data_dir, 'test.txt')
        self.data = []
        with open(data_path, 'r', encoding='utf8') as fin:
            for line in fin:
                if not line:
                    continue
                arr = line.rstrip('\n').split('\t')
                if len(arr) != 2:
                    print('Data format error: %s' % '\t'.join(arr))
                    print(
                        'Data row should contains two parts: conversation_content\tlabel1 label2 label3.'
                    )
                    continue
                text = arr[0]
                label_list = arr[1].split()
                self.data.append([text, label_list])

    @classmethod
    def get_label(cls, label):
        return cls.LABEL_MAP[label]

    @classmethod
    def num_classes(cls):
        return len(cls.LABEL_MAP)

    @classmethod
    def convert_example(cls, example, tokenizer, max_seq_length=512):
        """ Convert a glue example into necessary features. """
        text, labels = example
        tokens, label_list = [], []
        words = text.split()
        assert len(words) == len(labels)
        for word, label in zip(words, labels):
            piece_words = tokenizer.tokenize(word)
            tokens.extend(piece_words)
            label = cls.get_label(label)
            label_list.extend([label] * len(piece_words))
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[len(tokens) - max_seq_length + 2:]
            label_list = label_list[len(tokens) - max_seq_length + 2:]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        label_list = [0] + label_list + [0]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label = np.array(label_list, dtype='int64')
        return input_ids, segment_ids, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ATIS_DID(Dataset):
    """
    The dataset ATIS_ID is using in task Dialogue Intent Detection.
    The source dataset is ATIS(Airline Travel Information System). See detail at
    https://www.kaggle.com/siddhadev/ms-cntk-atis
    """
    LABEL_MAP = get_label_map([str(i) for i in range(26)])

    def __init__(self, data_dir, mode='train'):
        super(ATIS_DID, self).__init__()
        self._data_dir = data_dir
        self._mode = mode
        self.read_data()

    def read_data(self):
        if self._mode == 'train':
            data_path = os.path.join(self._data_dir, 'train.txt')
        elif self._mode == 'dev':
            data_path = os.path.join(self._data_dir, 'dev.txt')
        elif self._mode == 'test':
            data_path = os.path.join(self._data_dir, 'test.txt')
        self.data = []
        with open(data_path, 'r', encoding='utf8') as fin:
            for line in fin:
                if not line:
                    continue
                arr = line.rstrip('\n').split('\t')
                if len(arr) != 2:
                    print('Data format error: %s' % '\t'.join(arr))
                    print(
                        'Data row should contains two parts: label\tconversation_content.'
                    )
                    continue
                label = arr[0]
                text = arr[1]
                self.data.append([label, text])

    @classmethod
    def get_label(cls, label):
        return cls.LABEL_MAP[label]

    @classmethod
    def num_classes(cls):
        return len(cls.LABEL_MAP)

    @classmethod
    def convert_example(cls, example, tokenizer, max_seq_length=512):
        """ Convert a glue example into necessary features. """
        label, text = example
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[len(tokens) - max_seq_length + 2:]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label = np.array([cls.get_label(label)], dtype='int64')
        return input_ids, segment_ids, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def read_da_data(data_dir, mode):

    def _concat_dialogues(examples):
        """concat multi turns dialogues"""
        new_examples = []
        for i in range(len(examples)):
            label, caller, text = examples[i]
            cur_txt = "%s : %s" % (caller, text)
            pre_txt = [
                "%s : %s" % (item[1], item[2])
                for item in examples[max(0, i - 5):i]
            ]
            suf_txt = [
                "%s : %s" % (item[1], item[2])
                for item in examples[i + 1:min(len(examples), i + 3)]
            ]
            sample = [label, pre_txt, cur_txt, suf_txt]
            new_examples.append(sample)
        return new_examples

    if mode == 'train':
        data_path = os.path.join(data_dir, 'train.txt')
    elif mode == 'dev':
        data_path = os.path.join(data_dir, 'dev.txt')
    elif mode == 'test':
        data_path = os.path.join(data_dir, 'test.txt')
    data = []
    with open(data_path, 'r', encoding='utf8') as fin:
        pre_idx = -1
        examples = []
        for line in fin:
            if not line:
                continue
            arr = line.rstrip('\n').split('\t')
            if len(arr) != 4:
                print('Data format error: %s' % '\t'.join(arr))
                print(
                    'Data row should contains four parts: id\tlabel\tcaller\tconversation_content.'
                )
                continue
            idx, label, caller, text = arr
            if idx != pre_idx:
                if idx != 0:
                    examples = _concat_dialogues(examples)
                    data.extend(examples)
                    examples = []
                pre_idx = idx
            examples.append((label, caller, text))
        if examples:
            examples = _concat_dialogues(examples)
            data.extend(examples)
    return data


def truncate_and_concat(pre_txt: List[str], cur_txt: str, suf_txt: List[str],
                        tokenizer, max_seq_length, max_len_of_cur_text):
    cur_tokens = tokenizer.tokenize(cur_txt)
    cur_tokens = cur_tokens[:min(max_len_of_cur_text, len(cur_tokens))]
    pre_tokens = []
    for text in pre_txt:
        pre_tokens.extend(tokenizer.tokenize(text))
        pre_tokens.append(INNER_SEP)
    pre_tokens = pre_tokens[:-1]
    suf_tokens = []
    for text in suf_txt:
        suf_tokens.extend(tokenizer.tokenize(text))
        suf_tokens.append(INNER_SEP)
    suf_tokens = suf_tokens[:-1]
    if len(cur_tokens) + len(pre_tokens) + len(suf_tokens) > max_seq_length - 4:
        left_num = max_seq_length - 4 - len(cur_tokens)
        if len(pre_tokens) > len(suf_tokens):
            suf_num = int(left_num / 2)
            suf_tokens = suf_tokens[:suf_num]
            pre_num = left_num - len(suf_tokens)
            pre_tokens = pre_tokens[max(0, len(pre_tokens) - pre_num):]
        else:
            pre_num = int(left_num / 2)
            pre_tokens = pre_tokens[max(0, len(pre_tokens) - pre_num):]
            suf_num = left_num - len(pre_tokens)
            suf_tokens = suf_tokens[:suf_num]
    tokens, segment_ids = [], []
    tokens.extend([tokenizer.cls_token] + pre_tokens + [tokenizer.sep_token])
    segment_ids.extend([0] * len(tokens))
    tokens.extend(cur_tokens + [tokenizer.sep_token])
    segment_ids.extend([1] * (len(cur_tokens) + 1))
    if suf_tokens:
        tokens.extend(suf_tokens + [tokenizer.sep_token])
        segment_ids.extend([0] * (len(suf_tokens) + 1))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return input_ids, segment_ids


class MRDA(Dataset):
    """
    The dataset MRDA is using in task Dialogue Act.
    The source dataset is MRDA(Meeting Recorder Dialogue Act). See detail at
    https://www.aclweb.org/anthology/W04-2319.pdf
    """
    MAX_LEN_OF_CUR_TEXT = 50
    LABEL_MAP = get_label_map([str(i) for i in range(5)])

    def __init__(self, data_dir, mode='train'):
        super(MRDA, self).__init__()
        self.data = read_da_data(data_dir, mode)

    @classmethod
    def get_label(cls, label):
        return cls.LABEL_MAP[label]

    @classmethod
    def num_classes(cls):
        return len(cls.LABEL_MAP)

    @classmethod
    def convert_example(cls, example, tokenizer, max_seq_length=512):
        """ Convert a glue example into necessary features. """
        label, pre_txt, cur_txt, suf_txt = example
        label = np.array([cls.get_label(label)], dtype='int64')
        input_ids, segment_ids = truncate_and_concat(pre_txt, cur_txt, suf_txt,
                                                     tokenizer, max_seq_length,
                                                     cls.MAX_LEN_OF_CUR_TEXT)
        return input_ids, segment_ids, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SwDA(Dataset):
    """
    The dataset SwDA is using in task Dialogue Act.
    The source dataset is SwDA(Switchboard Dialog Act). See detail at
    http://compprag.christopherpotts.net/swda.html
    """
    MAX_LEN_OF_CUR_TEXT = 50
    LABEL_MAP = get_label_map([str(i) for i in range(42)])

    def __init__(self, data_dir, mode='train'):
        super(SwDA, self).__init__()
        self.data = read_da_data(data_dir, mode)

    @classmethod
    def get_label(cls, label):
        return cls.LABEL_MAP[label]

    @classmethod
    def num_classes(cls):
        return len(cls.LABEL_MAP)

    @classmethod
    def convert_example(cls, example, tokenizer, max_seq_length=512):
        """ Convert a glue example into necessary features. """
        label, pre_txt, cur_txt, suf_txt = example
        label = np.array([cls.get_label(label)], dtype='int64')
        input_ids, segment_ids = truncate_and_concat(pre_txt, cur_txt, suf_txt,
                                                     tokenizer, max_seq_length,
                                                     cls.MAX_LEN_OF_CUR_TEXT)
        return input_ids, segment_ids, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
