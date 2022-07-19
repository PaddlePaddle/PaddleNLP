import glob
import random
import numpy as np
from copy import deepcopy
from typing import List

import paddle
from paddle.io import IterableDataset
import paddle.distributed as dist


class Vocabulary(object):
    """
    A token vocabulary. Holds a map from token to ids and provides a method for 
    encoding text to a sequence of ids.

    Parameters:
        filename (str): The vocabulary file. It is a flat text file with 
            one (normalized) token per line.
    """

    def __init__(self, filename):
        self._word_to_id = {}
        for word in ['UNK', '<S>', '</S>', '<PAD>']:
            self._word_to_id[word] = len(self._word_to_id)
        with open(filename, 'r') as fin:
            for line in fin:
                word = line.strip()
                if word in self._word_to_id:
                    raise ValueError(
                        "There has repeated token in the vocabulary file: %s" %
                        word)
                self._word_to_id[word] = len(self._word_to_id)

    @property
    def bos(self):
        return self._word_to_id['<S>']

    @property
    def eos(self):
        return self._word_to_id['</S>']

    @property
    def unk(self):
        return self._word_to_id['UNK']

    @property
    def pad(self):
        return self._word_to_id['<PAD>']

    @property
    def size(self):
        return len(self._word_to_id)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def encode(self, sentence, split=True):
        """
        Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.
        """
        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        word_ids = [self.bos] + word_ids + [self.eos]
        word_ids_reverse = deepcopy(word_ids)
        word_ids_reverse.reverse()
        return np.array(word_ids, dtype=np.int64), np.array(word_ids_reverse,
                                                            dtype=np.int64)


class UnicodeCharsVocabulary(Vocabulary):
    """
    Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and a character id 
    that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8').
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence, 
    begin word, end word and char padding.

    Parameters:
        filename (str): The vocabulary file. It is a flat text file with 
            one (normalized) token per line.
        max_word_length (int): The maximum characters number of token in sequence.
    """

    def __init__(self, filename, max_word_length, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_word_length = max_word_length

        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260  # <char padding>

        num_words = len(self._word_to_id)

        self._word_char_ids = {}

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int64)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r

        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        for word in self._word_to_id:
            self._word_char_ids[word] = self._convert_word_to_char_ids(word)

        self._word_char_ids['<S>'] = self.bos_chars
        self._word_char_ids['</S>'] = self.eos_chars

    @property
    def char_size(self):
        # char ids 0-255 come from utf-8 encoding bytes.
        # assign 256-300 to special chars.
        # all +1, the id=0 is for token padding and mask.
        return 262

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int64)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8',
                                   'ignore')[:(self.max_word_length - 2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[word]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, split=True):
        """
        Encode the sentence as a white space delimited string of tokens.
        """
        if split:
            chars_ids = [
                self.word_to_char_ids(cur_word)
                for cur_word in sentence.split()
            ]
        else:
            chars_ids = [
                self.word_to_char_ids(cur_word) for cur_word in sentence
            ]

        chars_ids = [self.bos_chars] + chars_ids + [self.eos_chars]
        chars_ids_reverse = deepcopy(chars_ids)
        chars_ids_reverse.reverse()

        # +1 for token padding and mask
        chars_ids = np.vstack(chars_ids) + 1
        chars_ids_reverse = np.vstack(chars_ids_reverse) + 1
        return chars_ids, chars_ids_reverse


class CharsVocabulary(object):

    def __init__(self, max_word_length):
        self._max_word_length = max_word_length

        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260  # <char padding>

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int64)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r

        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

    @property
    def char_size(self):
        # char ids 0-255 come from utf-8 encoding bytes.
        # assign 256-300 to special chars.
        # all +1, the id=0 is for token padding and mask.
        return 262

    @property
    def max_word_length(self):
        return self._max_word_length

    def convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int64)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8',
                                   'ignore')[:(self.max_word_length - 2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = self.eow_char

        return code

    def encode_chars(self, sentence, split=True):
        """
        Encode the sentence as a white space delimited string of tokens.
        """
        if split:
            chars_ids = [
                self.convert_word_to_char_ids(cur_word)
                for cur_word in sentence.split()
            ]
        else:
            chars_ids = [
                self.convert_word_to_char_ids(cur_word) for cur_word in sentence
            ]

        chars_ids = [self.bos_chars] + chars_ids + [self.eos_chars]
        chars_ids_reverse = deepcopy(chars_ids)
        chars_ids_reverse.reverse()

        # +1 for token padding and mask
        chars_ids = np.vstack(chars_ids) + 1
        chars_ids_reverse = np.vstack(chars_ids_reverse) + 1
        return chars_ids, chars_ids_reverse


def load_vocab(vocab_file=None, max_word_length=50):
    if vocab_file is None:
        return CharsVocabulary(max_word_length)
    elif max_word_length:
        return UnicodeCharsVocabulary(vocab_file, max_word_length)
    else:
        return Vocabulary(vocab_file)


class OneBillionWordDataset(IterableDataset):
    """
    Hold the one billion word dataset, consisting of 1B Words which is used for 
    benchmarking of Language Modeling. The training/held-out data was produced 
    from the WMT 2011 News Crawl data.
    
    The dataset is a list of tokenized files. Each file contains one sentence 
    per line. Each sentence is pre-tokenized and white space joined.

    Parameters:
        filepattern (str): A glob string that specifies the list of files.
        vocab (Vocabulary): An instance of Vocabulary or UnicodeCharsVocabulary.
        batch_size (int): The batch_size.
        num_steps (int): The sentence length after re-cutting in dataset.
        n_procs (int): The number of GPUs.
        mode (str, optional): The dataset mode. It can be "train" and "test". 
            When "test", the dataset iterate through all data once then stop. 
            When "train", it will iterate forever. Default: "test".
        shuffle (bool, optional): Whether shuffle the data. Default: False.
        seed (int, optional): The random seed. Default: 0.
    """

    def __init__(self,
                 filepattern,
                 vocab,
                 batch_size,
                 num_steps,
                 n_procs=1,
                 rank=0,
                 mode='test',
                 shuffle=False,
                 seed=0):
        super(OneBillionWordDataset, self).__init__()

        self._all_files = glob.glob(filepattern)
        print('\nFound %d files at %s\n' % (len(self._all_files), filepattern))
        self._vocab = vocab
        self._max_word_length = vocab.max_word_length
        self._use_char_inputs = hasattr(vocab, 'encode_chars')
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._n_procs = n_procs
        self._rank = rank
        self._mode = mode
        self._shuffle = shuffle
        self._seed = abs(seed)
        self._file_seed = self._get_file_random_seed()

    def _get_file_random_seed(self):
        file_seed = {}
        np.random.seed(self._seed)
        seed_list = list(np.random.random(len(self._all_files)))
        for file_path, seed in zip(list(self._all_files), seed_list):
            file_seed[file_path] = seed
        return file_seed

    def _load_file(self, file_path):
        print('\nLoading data from: %s\n' % file_path)
        with open(file_path) as f:
            sentences_raw = f.readlines()
        sentences = sentences_raw

        if self._shuffle:
            if self._n_procs > 1:
                seed = self._file_seed[file_path] * self._seed
                random.seed(seed)
            random.shuffle(sentences)

        for sentence in sentences:
            ids, ids_reverse = self._vocab.encode(sentence)
            if self._use_char_inputs:
                char_ids, char_ids_reverse = self._vocab.encode_chars(sentence)
            else:
                char_ids, char_ids_reverse = None, None
            yield (ids, char_ids, ids_reverse, char_ids_reverse)

    def get_sentence(self):
        while True:
            self._seed += 1
            all_files = list(self._all_files)
            if self._shuffle:
                if self._n_procs > 1:
                    random.seed(self._seed)
                random.shuffle(all_files)
            for file_path in all_files:
                for ret in self._load_file(file_path):
                    yield ret
            if self._mode == 'test':
                break

    @property
    def number_of_tokens(self):
        # number of tokens in training data (1B Word Benchmark)
        return 768648884

    def __iter__(self):
        sentence_generator = self.get_sentence()
        n_batch_size = self._batch_size * self._n_procs
        cur_stream = [None] * n_batch_size

        while True:
            inputs = np.zeros([n_batch_size, self._num_steps], np.int64)
            inputs_reverse = np.zeros([n_batch_size, self._num_steps], np.int64)
            if self._max_word_length is not None:
                char_inputs = np.zeros(
                    [n_batch_size, self._num_steps, self._max_word_length],
                    np.int64)
                char_inputs_reverse = np.zeros(
                    [n_batch_size, self._num_steps, self._max_word_length],
                    np.int64)
            else:
                char_inputs = None
                char_inputs_reverse = None
            targets = np.zeros([n_batch_size, self._num_steps], np.int64)
            targets_reverse = np.zeros([n_batch_size, self._num_steps],
                                       np.int64)

            for i in range(n_batch_size):
                cur_pos = 0
                while cur_pos < self._num_steps:
                    if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                        try:
                            cur_stream[i] = list(next(sentence_generator))
                        except StopIteration:
                            return

                    how_many = min(
                        len(cur_stream[i][0]) - 1, self._num_steps - cur_pos)
                    next_pos = cur_pos + how_many

                    inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                    inputs_reverse[
                        i, cur_pos:next_pos] = cur_stream[i][2][:how_many]
                    if self._max_word_length is not None:
                        char_inputs[
                            i, cur_pos:next_pos] = cur_stream[i][1][:how_many]
                        char_inputs_reverse[
                            i, cur_pos:next_pos] = cur_stream[i][3][:how_many]
                    targets[i,
                            cur_pos:next_pos] = cur_stream[i][0][1:how_many + 1]
                    targets_reverse[
                        i, cur_pos:next_pos] = cur_stream[i][2][1:how_many + 1]

                    cur_pos = next_pos

                    cur_stream[i][0] = cur_stream[i][0][how_many:]
                    cur_stream[i][2] = cur_stream[i][2][how_many:]
                    if self._max_word_length is not None:
                        cur_stream[i][1] = cur_stream[i][1][how_many:]
                        cur_stream[i][3] = cur_stream[i][3][how_many:]

            # token_ids: (n_batch_size, self._num_steps)
            # char_inputs: character ids (n_batch_size, self._num_steps, 50)
            # targets: word ID of next word (n_batch_size, self._num_steps)
            batch_data = {
                'token_ids': inputs,
                'tokens_characters': char_inputs,
                'next_token_ids': targets,
                'token_ids_reverse': inputs_reverse,
                'tokens_characters_reverse': char_inputs_reverse,
                'next_token_ids_reverse': targets_reverse
            }
            if self._n_procs > 1:
                start = self._rank * self._batch_size
                end = start + self._batch_size
                for key in batch_data:
                    batch_data[key] = batch_data[key][start:end]

            yield (batch_data['tokens_characters'],
                   batch_data['next_token_ids'],
                   batch_data['tokens_characters_reverse'],
                   batch_data['next_token_ids_reverse'])


def create_one_batch(sentences, vocab, max_seq_len):
    # Add <S>, </S> for every sentence
    max_len = max([len(sentence) for sentence in sentences]) + 2
    max_len = min(max_len, max_seq_len)
    batch_ids = np.zeros([len(sentences), max_len, vocab.max_word_length],
                         dtype=np.int64)
    batch_ids_reverse = np.zeros(
        [len(sentences), max_len, vocab.max_word_length], dtype=np.int64)
    batch_lens = []
    for i, sentence in enumerate(sentences):
        sentence = sentence[:max_len - 2]
        seq_len = len(sentence) + 2
        ids, ids_reverse = vocab.encode_chars(sentence, split=False)
        batch_ids[i, :seq_len, :] = ids
        batch_ids_reverse[i, :seq_len, :] = ids_reverse
        batch_lens.append(seq_len)
    return batch_ids, batch_ids_reverse, batch_lens


def create_batches(sentences: List[List[str]], batch_size, vocab, max_seq_len):
    """
    Batch the sentences as character ids
    Each sentence is a list of tokens without <s> or </s>, e.g.
    [['The', 'first', 'sentence', '.'], ['Second', '.']]
    """
    n_batch = (len(sentences) - 1) // batch_size + 1
    for i in range(n_batch):
        start, end = i * batch_size, (i + 1) * batch_size
        ids, ids_reverse, seq_lens = create_one_batch(sentences[start:end],
                                                      vocab, max_seq_len)
        ids = paddle.to_tensor(ids)
        ids_reverse = paddle.to_tensor(ids_reverse)
        yield ids, ids_reverse, seq_lens
