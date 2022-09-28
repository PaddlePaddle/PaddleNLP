# This file is adapted from https://github.com/abisee/pointer-generator/blob/master/data.py

import glob
import struct
import csv
import queue
import time
import re
import json
import io

from threading import Thread
import numpy as np

import random
from random import shuffle

random.seed(123)

import config
import data

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Example(object):

    def __init__(self, article, abstract_sentences, vocab):
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # Process the article
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(
            article_words
        )  # store the length after truncation but before padding
        self.enc_input = [
            vocab.word2id(w) for w in article_words
        ]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        abstract = ' '.join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings
        abs_ids = [
            vocab.word2id(w) for w in abstract_words
        ]  # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(
            abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(
                article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab,
                                                     self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab,
                                                        config.max_dec_steps,
                                                        start_decoding,
                                                        stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):

    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(
            data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(
            example_list)  # initialize the input to the encoder
        self.init_decoder_seq(
            example_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len),
                                  dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len),
                                         dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max(
                [len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros(
                (self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps),
                                  dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps),
                                     dtype=np.int32)
        self.dec_padding_mask = np.zeros(
            (self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article
                                  for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract
                                   for ex in example_list]  # list of lists
        self.original_abstracts_sents = [
            ex.original_abstract_sents for ex in example_list
        ]  # list of list of lists


class Batcher(object):
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, mode, batch_size, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX *
                                          self.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 1  # 16 # num threads to fill example queue
            self._num_batch_q_threads = 1  # 4  # num threads to fill batch queue
            self._bucketing_cache_size = 1  # 100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(
                Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            print(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i'
                % (self._batch_queue.qsize(), self._example_queue.qsize()))
            if self._single_pass and self._finished_reading:
                print("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        input_gen = self.text_generator(
            data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                (article, abstract) = next(
                    input_gen
                )  # read the next example from file. article and abstract are both strings.
            except StopIteration:  # if there are no more examples:
                print(
                    "The example generator for this example queue filling thread has exhausted data."
                )
                if self._single_pass:
                    print(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping."
                    )
                    self._finished_reading = True
                    break
                else:
                    raise Exception(
                        "single_pass mode is off but the example generator is out of data; error."
                    )

            abstract_sentences = [
                sent.strip() for sent in data.abstract2sents(abstract)
            ]  # Use the <s> and </s> tags in abstract to get a list of sentences.
            example = Example(article, abstract_sentences,
                              self._vocab)  # Process into an Example.
            self._example_queue.put(
                example)  # place the Example in the example queue.

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # Beam search decode mode where a single example is repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(
                    inputs, key=lambda inp: inp.enc_len,
                    reverse=True)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._vocab,
                                                self.batch_size))

    def watch_threads(self):
        while True:
            print('Bucket queue size: %i, Input queue size: %i' %
                  (self._batch_queue.qsize(), self._example_queue.qsize()))

            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    print('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    print('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator):
        while True:
            e = next(example_generator)
            try:
                article_text = e['article']
                abstract_text = e['abstract']
            except ValueError:
                print('Failed to get article or abstract from example')
                continue
            if len(
                    article_text
            ) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                print('Found an example with empty article text. Skipping it.')
                continue
            else:
                yield (article_text, abstract_text)


class Vocab(object):

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf8') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print(
                        'Warning: incorrectly formatted line in vocabulary file: %s\n'
                        % line)
                    continue
                w = pieces[0]
                if w in [
                        SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN,
                        START_DECODING, STOP_DECODING
                ]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is'
                        % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' %
                                    w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print(
                        "max_size of vocab was specified as %i; we now have %i words. Stopping reading."
                        % (max_size, self._count))
                    break

        print(
            "Finished constructing vocabulary of %i total words. Last word added: %s"
            % (self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, fpath):
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass):
    while True:
        filelist = glob.glob(data_path)  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path
                          )  # check filelist isn't empty
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader = io.open(f, 'r', encoding='utf8')
            while True:
                reader_str = reader.readline()
                if reader_str == '': break
                reader_json = json.loads(reader_str)
                yield reader_json
        if single_pass:
            print(
                "example_generator completed reading all datafiles. No more data."
            )
            break


def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(
                w
            )  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(
                vocab.size() + oov_num
            )  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(
                    w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs'
                    % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w
             for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str
