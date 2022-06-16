# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, and NVIDIA, and PaddlePaddle Authors.
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

# Most of the code here has been copied from:
#   https://github.com/google-research/albert/blob/master/create_pretraining_data.py
# with some modifications.

import math
import os
import re
import time
import collections

import numpy as np
import paddle


def get_local_rank():
    return int(os.getenv("PADDLE_RANK_IN_NODE", 0))


print_rank_0 = print

COMPILED = False
DSET_TYPE_BERT = 'standard_bert'
DSET_TYPE_T5 = 't5'
DSET_TYPE_ERNIE = 'ernie'

DSET_TYPES = [DSET_TYPE_BERT, DSET_TYPE_T5, DSET_TYPE_ERNIE]


def compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess
    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(['make', '-C', path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys
        sys.exit(1)


class BlendableDataset(paddle.io.Dataset):
    """
    The BlendableDataset is a wrapper which used to mix different dataset.
    
    The input is a list of dataset and corresponding weights for each dataset.
    """

    def __init__(self, datasets, weights):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indecies.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        local_rank = 0 if fleet.local_rank() is None else int(
            fleet.local_rank())

        while True:
            try:
                import data_tools.helpers as helpers
                break
            except Exception as e:
                if local_rank == 0:
                    compile_helper()
                print_rank_0('> wait for hepers to be compiled!')
                time.sleep(1)

        import data_tools.helpers as helpers
        helpers.build_blending_indices(self.dataset_index,
                                       self.dataset_sample_index, weights,
                                       num_datasets, self.size, local_rank == 0)
        print_rank_0('> elapsed time for building blendable dataset indices: '
                     '{:.2f} (sec)'.format(time.time() - start_time))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return self.datasets[dataset_idx][sample_idx]


def get_datasets_weights_and_num_samples(data_prefix,
                                         train_valid_test_num_samples):

    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0] * num_datasets
    prefixes = [0] * num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2 * i])
        prefixes[i] = (data_prefix[2 * i + 1]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    # Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    datasets_train_valid_test_num_samples = []
    for weight in weights:
        datasets_train_valid_test_num_samples.append([
            int(math.ceil(val * weight * 1.005))
            for val in train_valid_test_num_samples
        ])

    return prefixes, weights, datasets_train_valid_test_num_samples


class MMapIndexedDataset(paddle.io.Dataset):

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = path

        # All documment ids, extend as 1-D array.

        for suffix in ["_ids.npy", "_idx.npz"]:
            if not os.path.isfile(path + suffix):
                raise ValueError("File Not found, %s" % (path + suffix))

        self._token_ids = np.load(path + "_ids.npy",
                                  mmap_mode="r",
                                  allow_pickle=True)
        process_data = np.load(path + "_idx.npz")
        self._sizes = process_data["lens"]
        self._pointers = np.empty(len(self._sizes) + 1, dtype=np.int64)
        self._pointers[0] = 0
        np.cumsum(self._sizes, out=self._pointers[1:])
        self._doc_idx = process_data["docs"]

    def __getstate__(self):
        return self._path

    def __len__(self):
        return len(self._sizes)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            size = self._sizes[idx]
            ptr = self._pointers[idx]
            np_array = self._token_ids[ptr:ptr + size]
            return np_array

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError(
                    "Slices into indexed_dataset must be contiguous")
            ptr = self._pointers[start]
            sizes = self._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = self._token_ids[ptr:ptr + total_size]
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        size = self._sizes[idx]
        ptr = self._pointers[idx]

        if length is None:
            length = size - offset
        ptr += offset
        np_array = self._token_ids[ptr:prt + length]
        return np_array

    @property
    def sizes(self):
        return self._sizes

    @property
    def doc_idx(self):
        return self._doc_idx

    def get_doc_idx(self):
        return self._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._doc_idx = doc_idx_


def make_indexed_dataset(data_prefix, data_impl=None, skip_warmup=False):
    return MMapIndexedDataset(data_prefix)


def get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, 'make sure each sample has at least two sentences.'

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in numpy is exclusive.
        a_end = np_rng.randint(1, n_sentences)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random


def truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, np_rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    #print(len_a, len_b, max_num_tokens)
    assert len_a > 0
    if len_a + len_b <= max_num_tokens:
        return False
    while len_a + len_b > max_num_tokens:
        if len_a > len_b:
            len_a -= 1
            tokens = tokens_a
        else:
            len_b -= 1
            tokens = tokens_b
        if np_rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()
    return True


def create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id):
    """Merge segments A and B, add [CLS] and [SEP] and build tokentypes."""

    tokens = []
    tokentypes = []
    # [CLS].
    tokens.append(cls_id)
    tokentypes.append(0)
    # Segment A.
    for token in tokens_a:
        tokens.append(token)
        tokentypes.append(0)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(0)
    # Segment B.
    for token in tokens_b:
        tokens.append(token)
        tokentypes.append(1)
    if tokens_b:
        # [SEP].
        tokens.append(sep_id)
        tokentypes.append(1)

    return tokens, tokentypes


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


def create_masked_lm_predictions(tokens,
                                 vocab_id_list,
                                 vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id,
                                 sep_id,
                                 mask_id,
                                 max_predictions_per_seq,
                                 np_rng,
                                 max_ngrams=3,
                                 vocab_token_to_id_dict=None,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False,
                                 geometric_dist=False,
                                 to_chinese_char=False,
                                 inplace_random_mask=False,
                                 masking_style="bert"):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        vocab_id = vocab_id_to_token_dict[token]
        if (do_whole_word_mask and len(cand_indexes) >= 1
                and not is_start_piece(vocab_id)):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

    if to_chinese_char:
        char_tokens = []
        assert vocab_token_to_id_dict is not None
        for i, b in enumerate(token_boundary):
            if b == 0:
                vocab_id = vocab_id_to_token_dict[tokens[i]]
                new_vocab_id = vocab_id[2:] if len(
                    re.findall('##[\u4E00-\u9FA5]', vocab_id)) > 0 else vocab_id
                char_tokens.append(
                    vocab_token_to_id_dict[new_vocab_id] if new_vocab_id in
                    vocab_token_to_id_dict else token)
            else:
                char_tokens.append(tokens[i])
        output_tokens = list(char_tokens)
    else:
        output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions, masked_lm_labels,
                token_boundary)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1. / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    backup_output_tokens = list(output_tokens)
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(ngrams[:len(cand_index_set)],
                              p=pvals[:len(cand_index_set)] /
                              pvals[:len(cand_index_set)].sum(keepdims=True))
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
            n = min(np_rng.geometric(0.2), max_ngrams)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = output_tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        if inplace_random_mask:
                            masked_token = backup_output_tokens[np_rng.randint(
                                0, len(output_tokens))]
                        else:
                            masked_token = vocab_id_list[np_rng.randint(
                                0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(
                MaskedLmInstance(index=index,
                                 label=backup_output_tokens[index]))

        masked_spans.append(
            MaskedLmInstance(
                index=index_set,
                label=[backup_output_tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                 pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(
                MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels,
            token_boundary, masked_spans)


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


def build_train_valid_test_datasets(data_prefix,
                                    args,
                                    tokenizer,
                                    splits_string,
                                    train_valid_test_num_samples,
                                    max_seq_length,
                                    masked_lm_prob,
                                    short_seq_prob,
                                    seed,
                                    skip_warmup,
                                    binary_head=False,
                                    max_seq_length_dec=None,
                                    dataset_type='standard_bert'):

    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(data_prefix[0],
                                                args,
                                                tokenizer,
                                                splits_string,
                                                train_valid_test_num_samples,
                                                max_seq_length,
                                                masked_lm_prob,
                                                short_seq_prob,
                                                seed,
                                                skip_warmup,
                                                binary_head,
                                                max_seq_length_dec,
                                                dataset_type=dataset_type)

    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix,
                                                  train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i],
            args,
            tokenizer,
            splits_string,
            datasets_train_valid_test_num_samples[i],
            max_seq_length,
            masked_lm_prob,
            short_seq_prob,
            seed,
            skip_warmup,
            binary_head,
            max_seq_length_dec,
            dataset_type=dataset_type)
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

        # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)


def _build_train_valid_test_datasets(data_prefix,
                                     args,
                                     tokenizer,
                                     splits_string,
                                     train_valid_test_num_samples,
                                     max_seq_length,
                                     masked_lm_prob,
                                     short_seq_prob,
                                     seed,
                                     skip_warmup,
                                     binary_head,
                                     max_seq_length_dec,
                                     dataset_type='standard_bert'):

    if dataset_type not in DSET_TYPES:
        raise ValueError("Invalid dataset_type: ", dataset_type)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, None, skip_warmup)

    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
        start_index = indexed_dataset.doc_idx[splits[index]]
        end_index = indexed_dataset.doc_idx[splits[index + 1]]
        print_rank_0('     sentence indices in [{}, {}) total of {} '
                     'sentences'.format(start_index, end_index,
                                        end_index - start_index))

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        # from megatron.data.bert_dataset import BertDataset
        # from megatron.data.t5_dataset import T5Dataset
        from .ernie_dataset import ErnieDataset
        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_doc_idx()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
            # Build the dataset accordingly.
            kwargs = dict(
                name=name,
                data_prefix=data_prefix,
                num_epochs=None,
                max_num_samples=train_valid_test_num_samples[index],
                max_seq_length=max_seq_length,
                seed=seed,
                share_folder=args.share_folder,
            )
            if dataset_type == DSET_TYPE_T5:
                dataset = T5Dataset(indexed_dataset=indexed_dataset,
                                    tokenizer=tokenizer,
                                    masked_lm_prob=masked_lm_prob,
                                    max_seq_length_dec=max_seq_length_dec,
                                    short_seq_prob=short_seq_prob,
                                    **kwargs)
            elif dataset_type == DSET_TYPE_BERT:
                dataset = BertDataset(indexed_dataset=indexed_dataset,
                                      tokenizer=tokenizer,
                                      masked_lm_prob=masked_lm_prob,
                                      short_seq_prob=short_seq_prob,
                                      binary_head=binary_head,
                                      **kwargs)
            elif dataset_type == DSET_TYPE_ERNIE:
                dataset = ErnieDataset(indexed_dataset=indexed_dataset,
                                       tokenizer=tokenizer,
                                       masked_lm_prob=masked_lm_prob,
                                       short_seq_prob=short_seq_prob,
                                       binary_head=binary_head,
                                       **kwargs)
            else:
                raise NotImplementedError("Dataset type not fully implemented.")

            # Set the original pointer so dataset remains the main dataset.
            indexed_dataset.set_doc_idx(doc_idx_ptr)
            # Checks.
            assert indexed_dataset.doc_idx[0] == 0
            assert indexed_dataset.doc_idx.shape[0] == \
                (total_num_of_documents + 1)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):

    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    assert indexed_dataset.sizes.shape[0] == indexed_dataset.doc_idx[-1]
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))

    print_rank_0(' > indexed dataset stats:')
    print_rank_0(
        '    number of documents: {}'.format(indexed_dataset.doc_idx.shape[0] -
                                             1))
    print_rank_0('    number of sentences: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def get_samples_mapping(indexed_dataset, data_prefix, num_epochs,
                        max_num_samples, max_seq_length, short_seq_prob, seed,
                        name, binary_head, share_folder):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""

    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples "
                             "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    local_rank = get_local_rank()
    if share_folder:
        local_rank = paddle.distributed.get_rank()
    # Build the indexed mapping if not exist.

    if local_rank == 0 and \
       not os.path.isfile(indexmap_filename):
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))

        # Make sure the types match the helpers input types.
        assert indexed_dataset.doc_idx.dtype == np.int64
        print(indexed_dataset.sizes.dtype)
        assert indexed_dataset.sizes.dtype == np.int32

        # Build samples mapping
        verbose = local_rank == 0
        start_time = time.time()
        print_rank_0(
            ' > building sapmles index mapping for {} ...'.format(name))
        # First compile and then import.
        if local_rank == 0:
            compile_helper()
        import data_tools.helpers as helpers
        samples_mapping = helpers.build_mapping(indexed_dataset.doc_idx,
                                                indexed_dataset.sizes,
                                                num_epochs, max_num_samples,
                                                max_seq_length, short_seq_prob,
                                                seed, verbose,
                                                2 if binary_head else 1)
        print_rank_0(' > done building sapmles index maping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0(
            ' > saved the index mapping in {}'.format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(' > elasped time to build and save samples mapping '
                     '(seconds): {:4f}'.format(time.time() - start_time))

    else:
        while True:
            if (not os.path.isfile(indexmap_filename)):
                time.sleep(3)
            else:
                try:
                    np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
                    break
                except Exception as e:
                    print(
                        "%s file is still writing or damaged, please wait a moment."
                        % indexmap_filename)
                    time.sleep(3)

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    if paddle.distributed.get_world_size() > 1:
        if paddle.in_dynamic_mode():
            paddle.distributed.barrier()

    # Load indexed dataset.
    print_rank_0(' > loading indexed mapping from {}'.format(indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename,
                              allow_pickle=True,
                              mmap_mode='r')
    print_rank_0(
        '    loaded indexed file in {:3.3f} seconds'.format(time.time() -
                                                            start_time))
    print_rank_0('    total number of samples: {}'.format(
        samples_mapping.shape[0]))

    return samples_mapping
