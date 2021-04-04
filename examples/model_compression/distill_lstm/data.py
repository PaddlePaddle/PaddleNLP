# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import io
from functools import partial
import numpy as np
import jieba

import paddle
from paddlenlp.data import Stack, Tuple, Pad, Vocab
from paddlenlp.transformers import BertTokenizer
from paddlenlp.datasets import load_dataset

from utils import convert_small_example, convert_two_example, convert_pair_example


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


def ngram_sampling(words, words_2=None, p_ng=0.25, ngram_range=(2, 6)):
    if np.random.rand() < p_ng:
        ngram_len = np.random.randint(ngram_range[0], ngram_range[1] + 1)
        ngram_len = min(ngram_len, len(words))
        start = np.random.randint(0, len(words) - ngram_len + 1)
        words = words[start:start + ngram_len]
        if words_2:
            words_2 = words_2[start:start + ngram_len]
    return words if not words_2 else (words, words_2)


def flatten(list_of_list):
    final_list = []
    for each_list in list_of_list:
        final_list += each_list
    return final_list


def apply_data_augmentation(data,
                            task_name,
                            tokenizer,
                            n_iter=20,
                            p_mask=0.1,
                            p_ng=0.25,
                            ngram_range=(2, 6),
                            whole_word_mask=False,
                            seed=0):
    """
    Data Augmentation contains Masking and n-gram sampling. Tokenization and
    Masking are performed at the same time, so that the masked token can be
    directly replaced by `mask_token`, after what sampling is performed.
    """

    def _data_augmentation(data, whole_word_mask=whole_word_mask):
        # 1. Masking
        words = []
        if not whole_word_mask:
            tokenized_list = tokenizer.tokenize(data)
            words = [
                tokenizer.mask_token if np.random.rand() < p_mask else word
                for word in tokenized_list
            ]
        else:
            for word in data.split():
                words += [[tokenizer.mask_token]] if np.random.rand(
                ) < p_mask else [tokenizer.tokenize(word)]
        # 2. N-gram sampling
        words = ngram_sampling(words, p_ng=p_ng, ngram_range=ngram_range)
        words = flatten(words) if isinstance(words[0], list) else words
        new_text = " ".join(words)
        return words, new_text

    np.random.seed(seed)
    used_texts, new_data = [], []
    for example in data:
        if task_name == 'qqp':
            data_list = tokenizer.tokenize(example['sentence1'])
            data_list_2 = tokenizer.tokenize(example['sentence2'])
            new_data.append({
                "sentence1": data_list,
                "sentence2": data_list_2,
                "labels": example['labels']
            })
            used_texts.append(" ".join(data_list_2))
        else:
            data_list = tokenizer.tokenize(example['sentence'])
            new_data.append({
                "sentence": data_list,
                "labels": example['labels']
            })

        used_texts.append(" ".join(data_list))
    used_texts = set(used_texts)

    for example in data:
        for _ in range(n_iter):
            if task_name == 'qqp':
                words, new_text = _data_augmentation(example['sentence1'])
                words_2, new_text_2 = _data_augmentation(example['sentence2'])
                if new_text not in used_texts or new_text_2 not in used_texts:
                    new_data.append({
                        "sentence1": words,
                        "sentence2": words_2,
                        "labels": example['labels']
                    })
                    used_texts.add(new_text_2)
            else:
                words, new_text = _data_augmentation(example['sentence'])
                if new_text not in used_texts:
                    new_data.append({
                        "sentence": words,
                        "labels": example['labels']
                    })
            used_texts.add(new_text)
    return new_data


def apply_data_augmentation_for_cn(data,
                                   tokenizer,
                                   vocab,
                                   n_iter=20,
                                   p_mask=0.1,
                                   p_ng=0.25,
                                   ngram_range=(2, 10),
                                   seed=0):
    """
    Because BERT and jieba have different `tokenize` function, it returns
    jieba_tokenizer(example['text'], bert_tokenizer(example['text']) and
    example['label]) for each example in data.
    jieba tokenization and Masking are performed at the same time, so that the
    masked token can be directly replaced by `mask_token`, and other tokens
    could be tokenized by BERT's tokenizer, from which tokenized example for
    student model and teacher model would get at the same time.
    """
    np.random.seed(seed)
    used_texts = [example['text'] for example in data]
    used_texts = set(used_texts)
    new_data = []
    for example in data:
        words = [word for word in jieba.cut(example['text'])]
        words_bert = tokenizer.tokenize(example['text'])
        new_data.append({
            "text": words,
            "sentence": words_bert,
            "label": example['label']
        })
        for _ in range(n_iter):
            # 1. Masking
            words, words_bert = [], []
            for word in jieba.cut(example['text']):
                if np.random.rand() < p_mask:
                    words.append([vocab.unk_token])
                    words_bert.append([tokenizer.unk_token])
                else:
                    words.append([word])
                    words_bert.append(tokenizer.tokenize(word))
            # 2. N-gram sampling
            words, words_bert = ngram_sampling(words, words_bert, p_ng,
                                               ngram_range)
            words, words_bert = flatten(words), flatten(words_bert)
            new_text = "".join(words)
            if new_text not in used_texts:
                new_data.append({
                    "text": words,
                    "sentence": words_bert,
                    "label": example['label']
                })
                used_texts.add(new_text)
    return new_data


def create_data_loader_for_small_model(task_name,
                                       vocab_path,
                                       model_name=None,
                                       batch_size=64,
                                       max_seq_length=128,
                                       shuffle=True):
    """Data loader for bi-lstm, not bert."""
    if task_name == 'chnsenticorp':
        train_ds, dev_ds = load_dataset(task_name, splits=["train", "dev"])
    else:
        train_ds, dev_ds = load_dataset(
            'glue', task_name, splits=["train", "dev"])
    if task_name == 'chnsenticorp':
        vocab = Vocab.load_vocabulary(
            vocab_path,
            unk_token='[UNK]',
            pad_token='[PAD]',
            bos_token=None,
            eos_token=None, )
        pad_val = vocab['[PAD]']

    else:
        vocab = BertTokenizer.from_pretrained(model_name)
        pad_val = vocab.pad_token_id

    trans_fn = partial(
        convert_small_example,
        task_name=task_name,
        vocab=vocab,
        max_seq_length=max_seq_length,
        is_test=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_val),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    ): fn(samples)

    train_ds = train_ds.map(trans_fn, lazy=True)
    dev_ds = dev_ds.map(trans_fn, lazy=True)

    train_data_loader, dev_data_loader = create_dataloader(
        train_ds, dev_ds, batch_size, batchify_fn, shuffle)

    return train_data_loader, dev_data_loader


def create_distill_loader(task_name,
                          model_name,
                          vocab_path,
                          batch_size=64,
                          max_seq_length=128,
                          shuffle=True,
                          n_iter=20,
                          whole_word_mask=False,
                          seed=0):
    """
    Returns batch data for bert and small model.
    Bert and small model have different input representations.
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    if task_name == 'chnsenticorp':
        train_ds, dev_ds = load_dataset(task_name, splits=["train", "dev"])
        vocab = Vocab.load_vocabulary(
            vocab_path,
            unk_token='[UNK]',
            pad_token='[PAD]',
            bos_token=None,
            eos_token=None, )
        pad_val = vocab['[PAD]']
        data_aug_fn = partial(
            apply_data_augmentation_for_cn,
            tokenizer=tokenizer,
            vocab=vocab,
            n_iter=n_iter,
            seed=seed)
    else:
        train_ds, dev_ds = load_dataset(
            'glue', task_name, splits=["train", "dev"])
        vocab = tokenizer
        pad_val = tokenizer.pad_token_id
        data_aug_fn = partial(
            apply_data_augmentation,
            task_name=task_name,
            tokenizer=tokenizer,
            n_iter=n_iter,
            whole_word_mask=whole_word_mask,
            seed=seed)
    train_ds = train_ds.map(data_aug_fn, batched=True)
    print("Data augmentation has been applied.")

    trans_fn = partial(
        convert_two_example,
        task_name=task_name,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=max_seq_length,
        vocab=vocab)

    trans_fn_dev = partial(
        convert_two_example,
        task_name=task_name,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=max_seq_length,
        vocab=vocab,
        is_tokenized=False)

    if task_name == 'qqp':
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # bert input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # bert segment
            Pad(axis=0, pad_val=pad_val),  # small input_ids
            Stack(dtype="int64"),  # small seq len
            Pad(axis=0, pad_val=pad_val),  # small input_ids
            Stack(dtype="int64"),  # small seq len
            Stack(dtype="int64")  # small label
        ): fn(samples)
    else:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # bert input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # bert segment
            Pad(axis=0, pad_val=pad_val),  # small input_ids
            Stack(dtype="int64"),  # small seq len
            Stack(dtype="int64")  # small label
        ): fn(samples)

    train_ds = train_ds.map(trans_fn, lazy=True)
    dev_ds = dev_ds.map(trans_fn_dev, lazy=True)
    train_data_loader, dev_data_loader = create_dataloader(
        train_ds, dev_ds, batch_size, batchify_fn, shuffle)
    return train_data_loader, dev_data_loader


def create_pair_loader_for_small_model(task_name,
                                       model_name,
                                       vocab_path,
                                       batch_size=64,
                                       max_seq_length=128,
                                       shuffle=True,
                                       is_test=False):
    """Only support QQP now."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_ds, dev_ds = load_dataset('glue', task_name, splits=["train", "dev"])
    vocab = Vocab.load_vocabulary(
        vocab_path,
        unk_token='[UNK]',
        pad_token='[PAD]',
        bos_token=None,
        eos_token=None, )

    trans_func = partial(
        convert_pair_example,
        task_name=task_name,
        vocab=tokenizer,
        is_tokenized=False,
        max_seq_length=max_seq_length,
        is_test=is_test)
    train_ds = train_ds.map(trans_func, lazy=True)
    dev_ds = dev_ds.map(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab['[PAD]']),  # input
        Stack(),  # length
        Pad(axis=0, pad_val=vocab['[PAD]']),  # input
        Stack(),  # length
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)

    train_data_loader, dev_data_loader = create_dataloader(
        train_ds, dev_ds, batch_size, batchify_fn, shuffle)
    return train_data_loader, dev_data_loader


def create_dataloader(train_ds, dev_ds, batch_size, batchify_fn, shuffle=True):
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=batch_size, shuffle=shuffle)

    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=batch_size, shuffle=False)

    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    return train_data_loader, dev_data_loader
