import random
import numpy as np
import gzip
from glob import glob
from contextlib import contextmanager
from collections import defaultdict
import paddle.distributed as dist
from paddle.io import IterableDataset


@contextmanager
def open_file(filename):
    """Open file."""
    if filename.endswith(".gz"):
        fp = gzip.open(filename, "rt")
    else:
        fp = open(filename)
    yield fp
    fp.close()


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class Vocabulary(object):
    """
    A token vocabulary. Holds a map from token to ids and provides a method for 
    encoding text to a sequence of ids.

    Parameters:
        filename (str): The vocabulary file. It is a flat text file with 
            one (normalized) token per line.
    """

    def __init__(self, filename):
        self.word_to_id, self.id_to_word = self.load_vocab(filename)

    def load_vocab(self, filename):
        """Loads a vocabulary file into a dictionary."""
        word_to_id = {}
        id_to_word = {}
        with open(filename) as fin:
            for num, line in enumerate(fin):
                items = convert_to_unicode(line.rstrip()).split("\t")
                if len(items) > 2:
                    break
                token = items[0]
                index = items[1] if len(items) == 2 else num
                token = token.strip()
                word_to_id[token] = int(index)
                id_to_word[int(index)] = token
        return word_to_id, id_to_word

    @property
    def bos_id(self):
        return self.word_to_id['[CLS]']

    @property
    def eos_id(self):
        return self.word_to_id['[SEP]']

    @property
    def unk_id(self):
        return self.word_to_id['[UNK]']

    @property
    def pad_id(self):
        return self.word_to_id['[PAD]']

    @property
    def mask_id(self):
        return self.word_to_id['[MASK]']

    @property
    def size(self):
        return len(self.id_to_word)

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of [tokens|ids] using the vocab."""
        tokens = []
        for id in ids:
            tokens.append(self.id_to_word[id])
        return tokens


class DialogueDataset(IterableDataset):
    def __init__(self,
                 filepattern,
                 vocab,
                 batch_size,
                 sort_pool_size=2**16,
                 seed=1,
                 n_gpus=None,
                 rank=None,
                 mode='test'):
        super(DialogueDataset, self).__init__()

        self.file_list = glob(filepattern)
        self.sort_pool_size = 0 if mode == 'test' else sort_pool_size
        self.n_gpus = n_gpus if n_gpus else dist.get_world_size()
        self.rank = rank if rank else dist.get_rank()
        self.batch_size = batch_size * self.n_gpus
        self.shuffle = True if mode == 'train' else False
        self.mode = mode
        self.pad_id = vocab.pad_id
        self.bos_id = vocab.bos_id
        self.global_rng = np.random.RandomState(seed)

        assert len(self.file_list) > 0, 'There is no files in %s.' % filepattern

    def load_file(self, file_path):
        with open_file(file_path) as fin:
            for i, line in enumerate(fin):
                cols = convert_to_unicode(line).strip().split(";")
                cols = list(map(lambda x: list(map(int, x.split(" "))), cols))
                if len(cols) > 3:
                    cols = cols[:3]
                token_ids, type_ids, pos_ids = cols
                if self.mode == 'test':
                    tgt_start_idx = len(cols[0])
                else:
                    tgt_start_idx = token_ids.index(self.bos_id, 1)
                data_id = i
                sample = [token_ids, type_ids, pos_ids, tgt_start_idx]
                yield sample

    def get_sorted_batch(self, pool):
        """Generate sorted batches from pool."""
        pool = sorted(pool, key=lambda sample: len(sample[0]))
        batches = []
        batch, max_len = [], 0
        for sample in pool:
            max_len = max(max_len, len(sample[0]))
            if self.mode == 'test':
                to_append = len(batch) < self.batch_size
            else:
                to_append = (len(batch) + 1) * max_len <= self.batch_size
            if to_append:
                batch.append(sample)
            else:
                batches.append(batch)
                batch, max_len = [sample], len(sample[0])
        if len(batch) > 0:
            batches.append(batch)
        if self.shuffle:
            self.global_rng.shuffle(batches)
        for batch in batches:
            yield batch

    @property
    def get_batch(self):
        all_files = list(self.file_list)
        if self.shuffle:
            self.global_rng.shuffle(all_files)
        if self.sort_pool_size > 0:
            pool = []
            for file_path in all_files:
                for sample in self.load_file(file_path):
                    pool.append(sample)
                    if len(pool) == self.sort_pool_size:
                        for batch in self.get_sorted_batch(pool):
                            yield batch
                        pool = []
                if len(pool) > 0:
                    for batch in self.get_sorted_batch(pool):
                        yield batch
        else:
            batch, max_len = [], 0
            for file_path in all_files:
                for sample in self.load_file(file_path):
                    max_len = max(max_len, len(sample[0]))
                    if self.mode == 'test':
                        to_append = len(batch) < self.batch_size
                    else:
                        to_append = (len(batch) + 1
                                     ) * max_len <= self.batch_size
                    if to_append:
                        batch.append(sample)
                    else:
                        yield batch
                        batch, max_len = [sample], len(sample[0])
            if len(batch) > 0:
                yield batch

    def pad_batch_data(self, batch):
        """Pad the instances to the max sequence length in batch. """
        max_len = max(map(len, batch))
        batch_data = np.array(
            [
                list(data) + [self.pad_id] * (max_len - len(data))
                for data in batch
            ],
            dtype='int64')
        return batch_data

    def gen_tgt_label_and_pos(self, batch_token_ids, batch_tgt_start_idx):
        batch_token_ids = np.copy(batch_token_ids)
        max_len = max(map(len, batch_token_ids))
        tgt_label = []
        tgt_pos = []

        for sent_index, sent in enumerate(batch_token_ids):
            sent_b_index = batch_tgt_start_idx[sent_index]
            need_cal = True
            tgt_label.extend(sent[sent_b_index + 1:])
            tgt_pos.extend([
                sent_index * max_len + i for i in range(sent_b_index,
                                                        len(sent) - 1)
            ])
        tgt_label = np.array(tgt_label).astype("int64")
        tgt_pos = np.array(tgt_pos).astype("int64")

        return tgt_label, tgt_pos

    def gen_self_attn_mask(self, batch_token_ids, batch_tgt_start_idx):
        max_len = max(map(len, batch_token_ids))
        input_mask_data = np.zeros((len(batch_token_ids), max_len, max_len))
        for index, mask_data in enumerate(input_mask_data):
            start = batch_tgt_start_idx[index]
            end = len(batch_token_ids[index])
            mask_data[:end, :start] = 1.0
            # Generate the lower triangular matrix using the slice of matrix
            b = np.tril(np.ones([end - start, end - start]), 0)
            mask_data[start:end, start:end] = b
        return input_mask_data.astype("float32")

    def __iter__(self):
        for batch_data in self.get_batch:
            # sample [token_ids, type_ids, pos_ids, tgt_start_idx]
            # raw_batch [sample0, sample1, ...]
            if self.n_gpus > 1:
                batch_data = batch_data[self.rank::self.n_gpus]
            batch_data = zip(*batch_data)
            token_ids, type_ids, pos_ids, tgt_start_idx = batch_data

            pad_token_ids = self.pad_batch_data(token_ids)
            pad_type_ids = self.pad_batch_data(type_ids)
            pad_pos_ids = self.pad_batch_data(pos_ids)

            generation_mask = self.gen_self_attn_mask(token_ids, tgt_start_idx)

            if self.mode == 'test':
                # [batch_size, 1]
                tgt_ids = np.array(
                    [[self.bos_id]] * len(token_ids), dtype="int64")
                tgt_type = np.ones((len(token_ids), 1), dtype="int64")
                tgt_pos = np.array(tgt_start_idx, dtype="int64").reshape(-1, 1)
                tgt_generation_mask = generation_mask[:, 0:1, :].astype(
                    "float32")

                pad_token_ids = np.concatenate((pad_token_ids, tgt_ids), axis=1)
                pad_type_ids = np.concatenate((pad_type_ids, tgt_type), axis=1)
                pad_pos_ids = np.concatenate((pad_pos_ids, tgt_pos), axis=1)
                generation_mask = np.concatenate(
                    (generation_mask, tgt_generation_mask), axis=1)

                append_mask = np.zeros(
                    (generation_mask.shape[0], generation_mask.shape[1], 1),
                    dtype="float32")
                append_mask[:, -1, :] = 1.0
                generation_mask = np.concatenate(
                    (generation_mask, append_mask), axis=2)
                yield (pad_token_ids, pad_type_ids, pad_pos_ids,
                       generation_mask)
            else:
                tgt_label, tgt_pos = self.gen_tgt_label_and_pos(token_ids,
                                                                tgt_start_idx)
                yield (pad_token_ids, pad_type_ids, pad_pos_ids,
                       generation_mask, tgt_label, tgt_pos)


def merge_subword(tokens):
    """Merge subword."""
    response = []
    for token in tokens:
        if token.startswith(u"▁"):
            response.append(token[1:])
        else:
            if len(response):
                response[-1] += token
            else:
                response.append(token)

    response = [token for token in response if token]
    return response


def post_process_response(token_ids, vocab):
    """
    Post-process the decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
    """
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == vocab.eos_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = vocab.convert_ids_to_tokens(token_ids)
    response = merge_subword(tokens)
    return token_ids, response


def get_in_turn_repetition(pred, is_cn=False):
    """Get in-turn repetition."""
    if len(pred) == 0:
        return 1.0
    if isinstance(pred[0], str):
        pred = [tok.lower() for tok in pred]
        if is_cn:
            pred = "".join(pred)
    tri_grams = set()
    for i in range(len(pred) - 2):
        tri_gram = tuple(pred[i:i + 3])
        if tri_gram in tri_grams:
            return True
        tri_grams.add(tri_gram)
    return False


def select_response(ids, scores, vocab, max_dec_len=None, num_samples=1):
    ids = ids.numpy()
    scores = scores.numpy()

    if len(ids) != len(scores) or (len(ids) % num_samples) != 0:
        raise ValueError("the length of output is error: {}, {}".format(
            len(ids), len(scores)))

    group = []
    tmp = []
    for pred, score in zip(ids, scores):
        pred_token_ids, pred_tokens = post_process_response(pred, vocab)
        num_token = len(pred_token_ids)
        response = " ".join(pred_tokens)

        in_turn_repetition = get_in_turn_repetition(
            pred_tokens, True) or get_in_turn_repetition(pred_token_ids)
        # not ending
        if max_dec_len is not None and num_token >= max_dec_len:
            score -= 1e3
        elif in_turn_repetition:
            score -= 1e3

        tmp.append([response, score])
        if len(tmp) == num_samples:
            group.append(tmp)
            tmp = []

    results = []
    for preds in group:
        preds = sorted(preds, key=lambda x: -x[1])
        results.append(preds[0][0])
    return results
