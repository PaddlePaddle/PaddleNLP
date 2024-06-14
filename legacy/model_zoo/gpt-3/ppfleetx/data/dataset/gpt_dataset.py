# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import json
import math
import os
import re
import time

import numpy as np
import paddle
from ppfleetx.data.tokenizers import GPTTokenizer
from ppfleetx.distributed.apis import env
from ppfleetx.utils.log import logger

# TODO(haohongxiang): to solve the problem of cross-reference
import paddlenlp  # noqa: F401
from paddlenlp.transformers.gpt.tokenizer import GPTChineseTokenizer

mode_to_index = {"Train": 0, "Eval": 1, "Test": 2}

MODEL_CLASSES = {
    "GPT": (GPTTokenizer, "gpt2"),
    "GPT-cn": (GPTChineseTokenizer, "gpt-cpm-large-cn"),
}


class GPTDataset(paddle.io.Dataset):
    def __init__(self, input_dir, split, max_seq_len, num_samples, mode, model_type="GPT", seed=1234):

        files = get_train_data_file(input_dir)
        files.sort()
        input_dir = [files[0]]

        local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

        if local_rank == 0:
            try:
                import ppfleetx.data.data_tools.cpp.fast_index_map_helpers
            except Exception:
                start_time = time.time()
                print("> compiling dataset index builder ...")
                from ppfleetx.data.data_tools.cpp.compile import compile_helper

                compile_helper()
                print(
                    ">>> done with dataset index builder. Compilation time: {:.3f} "
                    "seconds".format(time.time() - start_time),
                    flush=True,
                )

        device_world_size = paddle.distributed.get_world_size()

        if device_world_size > 1 and local_rank != 0:
            while True:
                try:
                    import ppfleetx.data.data_tools.cpp.fast_index_map_helpers  # noqa: F401, F811

                    break
                except Exception:
                    print("> wait for helpers to be compiled!")
                    time.sleep(1)

        try:
            data_world_size = env.get_data_world_size()

            logger.info(
                "The distributed run, total device num:{}, distinct dataflow num:{}.".format(
                    device_world_size, data_world_size
                )
            )
        except AttributeError:
            pass

        assert len(input_dir) == 1, "GPT only support one dataset for now."

        input_prefix = input_dir[0]

        if os.path.isfile(input_prefix + "_ids.npz"):
            logger.warning("You are using compatible dataset, please make new dataset as the readme!")
            process_data = np.load(input_prefix + "_ids.npz", mmap_mode="r+", allow_pickle=True)
            sample_ids = process_data["ids"]
            sample_lens = process_data["lens"].astype("int32")
        else:
            for suffix in ["_ids.npy", "_idx.npz"]:
                if not os.path.isfile(input_prefix + suffix):
                    raise ValueError("File Not found, %s" % (input_prefix + suffix))

            sample_ids = np.load(input_prefix + "_ids.npy", mmap_mode="r", allow_pickle=True)
            # All documment ids, extend as 1-D array.

            process_data = np.load(input_prefix + "_idx.npz")
            # The len(sample_lens) num of docs
            # The sum(sample_lens) should equal len(sample_ids)
            sample_lens = process_data["lens"]

        splits = get_train_valid_test_split_(split, len(sample_lens))
        assert len(sample_lens) >= splits[-1], "The document nums should larger than max of splits, but %s < %s" % (
            len(sample_lens),
            splits[-1],
        )

        tokenizer_class, pretrained_name = MODEL_CLASSES[model_type]
        tokenizer = tokenizer_class.from_pretrained(pretrained_name)

        self.input_dir = input_dir
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.name = "gpt_" + mode
        self.eos_id = tokenizer.eos_token_id
        self.sample_ids = sample_ids
        self.sample_lens = sample_lens
        self.build_data_file = local_rank == 0

        if mode in mode_to_index.keys():
            index = mode_to_index[mode]
        else:
            raise ValueError("valid str value for 'mode'")

        documents = np.arange(splits[index], splits[index + 1])
        if documents is None:
            document_ids = np.arange(0, self.sample_lens.shape[0])
        else:
            document_ids = documents

        self.doc_idx, self.sample_idx, self.shuffle_idx = construct_samples_and_shuffle_data(
            self.name,
            input_prefix,
            document_ids,
            self.sample_lens,
            num_samples,
            max_seq_len,
            seed,
            self.build_data_file,
        )

        # The doc cumsum start pos
        self.start_pos = [0] + np.cumsum(self.sample_lens).tolist()

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)
        # Attention mask for the attention calulate
        # attention_mask = np.tri(seq_length, seq_length).reshape((1, seq_length,
        #  seq_length))
        # The pad and eos tokens do not contribute the loss
        loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[tokens == self.eos_id] = 0.0
        position_ids = np.arange(0, seq_length, dtype="int64")

        labels = np.array(labels).astype("int64")
        tokens = np.array(tokens).astype("int64")
        if self.mode == "Test":
            return [tokens, position_ids]
        else:
            return [tokens, position_ids, labels, loss_mask]

    def _get_single_sample_from_idx(self, doc_index_f, doc_index_l, offset_f, offset_l):
        """
        The input means:
            doc_index_f: data from the first doc.
            doc_index_l: data from the last doc.
            offset_f: offset of the first doc.
            offset_l: offset of the last doc.
        """
        # Data from the sample doc. just select the needed ids.
        if doc_index_f == doc_index_l:
            current_start_pos = self.start_pos[self.doc_idx[doc_index_f]]
            return self.sample_ids[current_start_pos + offset_f : current_start_pos + offset_l + 1].tolist()

        # Data from multi docs.
        else:
            current_start_pos = self.start_pos[self.doc_idx[doc_index_f]]
            next_start_pos = self.start_pos[self.doc_idx[doc_index_f] + 1]
            tokens = self.sample_ids[current_start_pos + offset_f : next_start_pos].tolist()
            for i in range(doc_index_f + 1, doc_index_l):
                current_start_pos = self.start_pos[self.doc_idx[i]]
                next_start_pos = self.start_pos[self.doc_idx[i] + 1]
                tokens.extend(self.sample_ids[current_start_pos:next_start_pos].tolist())
            last_start_pos = self.start_pos[self.doc_idx[doc_index_l]]
            tokens.extend(self.sample_ids[last_start_pos : last_start_pos + offset_l + 1].tolist())

        return tokens

    def __getitem__(self, index):
        idx = self.shuffle_idx[index]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        tokens = self._get_single_sample_from_idx(doc_index_f, doc_index_l, offset_f, offset_l)
        return self._construct_sample(tokens)

    def __len__(self):
        return self.sample_idx.shape[0] - 1


def get_train_data_file(input_dir):
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if (os.path.isfile(os.path.join(input_dir, f)) and str(f).endswith("_idx.npz"))
    ]
    files = [x.replace("_idx.npz", "") for x in files]
    if len(files) == 0:
        logger.warning(
            "Not found dataset with name of xxx_ids.npy and xxx_idx.npz! Try to found old compatible xxx_ids.npz file."
        )
    else:
        return files

    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if (os.path.isfile(os.path.join(input_dir, f)) and str(f).endswith("_ids.npz"))
    ]

    files = [x.replace("_ids.npz", "") for x in files]

    if len(files) == 0:
        raise RuntimeError("Not found dataset with name of xxx_ids.npz in given input_dir '{}'! ".format(input_dir))
    else:
        return files


def get_train_valid_test_split_(splits, size):
    """
    Get dataset splits from comma or '/' separated string list.
    """

    splits = [float(s) for s in splits]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def construct_samples_and_shuffle_data(
    name, data_prefix, documents, sizes, num_samples, seq_length, seed, build_data_file
):
    """
    documents: document index from 0 to len(docs)
    sizes: the length list of all docs.
    num_samples: total step*bs iterations of data.
    seq_length: the sequence length.
    sum(sizes) = tokens_per_epoch
    data_nums = num_samples *  micro_batch_size
    num_epochs = (data_nums + 1) // sum(sizes)
    len(doc_idx) = num_epochs * sum(sizes)
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # Rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    # Sava random state
    savedState = np_rng.get_state()
    # Build the indexed mapping if not exist.
    if build_data_file:
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):
            if num_epochs == 1:
                separate_last_epoch = False
            else:
                num_samples_from_epochs_minus_one = ((num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                last_epoch_num_samples = num_samples - num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, "last epoch number of samples should be non-negative."
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                assert last_epoch_num_samples < (
                    num_samples_per_epoch + 1
                ), "last epoch number of samples exceeded max value."
                separate_last_epoch = last_epoch_num_samples < int(0.80 * num_samples_per_epoch)
            # Note. len(doc_idx) = num_epochs * len(doc)
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print(
                " > elasped time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # sample-idx. pos of each seq_len of data.
            start_time = time.time()
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32

            from ppfleetx.data.data_tools.cpp import fast_index_map_helpers

            sample_idx = fast_index_map_helpers.build_sample_idx(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
            )
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                                num_epochs, tokens_per_epoch)

            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print(
                " > elasped time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )

            # shuffle-idx.
            start_time = time.time()

            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1

            # Shuffle all seq len data.
            shuffle_idx = _build_shuffle_idx(num_samples_, sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print(
                " > elasped time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time)
            )

    else:
        while True:
            if (
                (not os.path.isfile(doc_idx_filename))
                or (not os.path.isfile(sample_idx_filename))
                or (not os.path.isfile(shuffle_idx_filename))
            ):
                time.sleep(3)
            else:
                try:
                    np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
                    break
                except Exception:
                    print("%s file is still writing or damaged, please wait a moment." % shuffle_idx_filename)
                    time.sleep(3)

    # Restore random state
    np_rng.set_state(savedState)

    try:
        if paddle.distributed.get_world_size() > 1:
            if paddle.in_dynamic_mode():
                paddle.distributed.barrier()
    except AssertionError:
        pass

    # Load mappings.
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, lens):
    """Total number of tokens in the dataset."""
    return np.sum(lens[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """
    Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document.
    """
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        doc_idx[:] = documents
        # The documents repeat num_epochs times.
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    """
    num_samples + 1, pos of bs data
    the distance between two points for sample idx is bs tokens.
    """
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([int(num_samples) + 1, 2], dtype=np.int32)

    sample_index = 0
    doc_idx_index = 0
    doc_offset = 0
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            remaining_seq_length -= doc_length
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - 1
                remaining_seq_length = 0
            else:
                doc_idx_index += 1
                doc_offset = 0
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


class LM_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, input_dir, max_seq_len, overlapping_eval=None, model_type="GPT", **kwargs):
        tokenizer_class, pretrained_name = MODEL_CLASSES[model_type]
        tokenizer = tokenizer_class.from_pretrained(pretrained_name)

        with open(input_dir, "rb") as reader:
            entire_data = reader.read().decode("utf-8")

        self.num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = self._wikitext_detokenizer(entire_data)
        self.tokens = tokenizer.encode(entire_data)
        self.num_tokenized_tokens = len(self.tokens)
        print("Original Tokens: %d, Detokenized tokens: %d" % (self.num_original_tokens, self.num_tokenized_tokens))

        self.seq_len = max_seq_len
        self.pad_idx = tokenizer.eos_token_id
        self.overlapping_eval = overlapping_eval
        if self.overlapping_eval is None:
            self.overlapping_eval = self.seq_len
        self.overlapping_eval = max(1, self.overlapping_eval)

        self.total_targets = len(self.tokens) - 1
        # remove first sequence tokens
        targets = max(self.total_targets - self.overlapping_eval, 0)
        self.total_sequences = max(math.ceil(targets / self.overlapping_eval) + 1, 1)

    def __len__(self):
        return self.total_sequences

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape((1, seq_length, seq_length))

        # the pad and eos tokens do not contribute the loss
        loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[tokens == self.pad_idx] = 0.0
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        # attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        start_idx = idx * self.overlapping_eval
        end_idx = start_idx + self.seq_len
        tokens = self.tokens[start_idx : end_idx + 1]
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = self.seq_len + 1 - num_tokens
            tokens += [self.pad_idx] * num_pad
        [tokens, loss_mask, attention_mask, position_ids, labels] = self._construct_sample(tokens)
        if self.overlapping_eval != self.seq_len and idx != 0:
            loss_mask[: -self.overlapping_eval] *= 0

        return [
            tokens,
            loss_mask,
            attention_mask,
            position_ids,
            labels,
            np.array([self.num_original_tokens, self.num_tokenized_tokens]),
        ]

    def _wikitext_detokenizer(self, string):
        # contractions
        string = string.replace("s '", "s'")
        string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
        # number separators
        string = string.replace(" @-@ ", "-")
        string = string.replace(" @,@ ", ",")
        string = string.replace(" @.@ ", ".")
        # punctuation
        string = string.replace(" : ", ": ")
        string = string.replace(" ; ", "; ")
        string = string.replace(" . ", ". ")
        string = string.replace(" ! ", "! ")
        string = string.replace(" ? ", "? ")
        string = string.replace(" , ", ", ")
        # double brackets
        string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
        string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
        string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
        string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
        string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
        # miscellaneous
        string = string.replace("= = = =", "====")
        string = string.replace("= = =", "===")
        string = string.replace("= =", "==")
        string = string.replace(" " + chr(176) + " ", chr(176))
        string = string.replace(" \n", "\n")
        string = string.replace("\n ", "\n")
        string = string.replace(" N ", " 1 ")
        string = string.replace(" 's", "'s")
        return string


class Lambada_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, input_dir, max_seq_len, model_type="GPT", **kwargs):
        tokenizer_class, pretrained_name = MODEL_CLASSES[model_type]
        tokenizer = tokenizer_class.from_pretrained(pretrained_name)

        tokenized_data = []
        tokenized_label = []
        with open(input_dir, "r") as f:
            for line in f.readlines():
                text = json.loads(line)["text"]
                tokens, labels = self._get_tokens(tokenizer, text)
                tokenized_data.append(tokens)
                tokenized_label.append(labels)

        self.pad_idx = tokenizer.eos_token_id
        self.seq_len = max_seq_len
        self.tokens = tokenized_data
        self.labels = tokenized_label

    def __len__(self):
        return len(self.tokens)

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]

        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape((1, seq_length, seq_length))

        # the pad and eos tokens do not contribute the loss
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        # attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        tokens = self.tokens[idx][: self.seq_len]
        labels = self.labels[idx]
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = self.seq_len + 1 - num_tokens
            tokens += [self.pad_idx] * num_pad
        loss_mask = np.zeros(self.seq_len, dtype="float32")
        loss_mask[num_tokens - len(labels) - 1 : num_tokens - 1] = 1.0
        [tokens, attention_mask, position_ids, labels] = self._construct_sample(tokens)
        return [tokens, loss_mask, attention_mask, position_ids, labels, np.array([self.__len__()])]

    def _get_tokens(self, tokenizer, text, strict=True):
        if not strict:
            tokens = tokenizer.encode(text)
            return tokens[:-1], [tokens[-1]]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)
        beginning_tokens = tokenizer.encode(text[:start_idx].strip())
        last_token = tokenizer.encode(" " + last_token)
        return beginning_tokens, last_token
