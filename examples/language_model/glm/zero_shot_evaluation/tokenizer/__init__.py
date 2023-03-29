# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""utils for creating datasets"""
import os
import math
import time
import random
#import torch

#from .samplers import DistributedBatchSampler
#from .datasets import GPT2Dataset
# split_ds, ConcatDataset, SplitDataset, BertSentencepairDataset, \
#    GPT2Dataset, ShuffleDataset, XLDataset, BlockDataset
#from .lazy_loader import exists_lazy, LazyWriter, LazyLoader, exists_scatter, get_scatter_path
from .tokenization import Tokenization, CommandToken, Tokenizer, CharacterLevelTokenizer, BertWordPieceTokenizer, \
    GPT2BPETokenizer, make_tokenizer
#from . import corpora

TRAIN_DATA = 0
VAL_DATA = 1
TEST_DATA = 2


def should_split(split):
    """
    given split proportions checks if should split
    Examples:
    >>> should_split([10,0,0]) 
    False
    >>> should_split([1,.1,.2])
    True
    """
    return max(split) / sum(split) != 1.


def get_ext(path):
    """gets path extension"""
    return os.path.splitext(path)[1]


def get_dataset(name, tokenizer, pre_tokenize, data_parallel_rank, loader_scatter=None, no_lazy_loader=False,
                half_lazy_loader=False):
    """gets dataset object based on keyword args and file at `path`"""
    global_rank = torch.distributed.get_rank()
    if not supported_corpus(name):
        raise NotImplementedError('dataset %s is not supported' % name)
    dataset = corpora.NAMED_CORPORA[name]
    path = dataset.PATH
    if issubclass(dataset, corpora.PromptReader):
        if not (exists_lazy(path, data_type='prompt') and exists_lazy(path, data_type='text')) and not (
                loader_scatter is not None and exists_scatter(path, data_type='prompt',
                                                              scatter_num=loader_scatter) and exists_scatter(path,
                                                                                                             data_type='text',
                                                                                                             scatter_num=loader_scatter)):
            # create cached version of dataset for lazy loading if it doesn't exist
            if global_rank == 0:
                print(f"Creating lazy loader for dataset {name}")
                prompt_writer = LazyWriter(path, data_type='prompt', is_array=pre_tokenize)
                text_writer = LazyWriter(path, data_type='text', is_array=pre_tokenize)
                writers = {'prompt': prompt_writer, 'text': text_writer}
                reader = dataset(writers=writers, tokenizer=tokenizer, tokenize=pre_tokenize)
                reader.process()
                prompt_writer.close()
                text_writer.close()
            else:
                while not os.path.exists(LazyWriter.get_len_path(path, data_type='prompt')):
                    time.sleep(1)
        map_fn = (lambda x: x.tolist()) if pre_tokenize else None
        if loader_scatter is not None:
            if not (exists_scatter(path, data_type='prompt', scatter_num=loader_scatter) and exists_scatter(path,
                                                                                                            data_type='text',
                                                                                                            scatter_num=loader_scatter)):
                if global_rank == 0:
                    print(f"Creating scatter loader for dataset {name}")
                    prompts = LazyLoader(path, data_type='prompt', map_fn=map_fn, mem_map=True,
                                         is_array=pre_tokenize)
                    texts = LazyLoader(path, data_type='text', map_fn=map_fn, mem_map=True,
                                       is_array=pre_tokenize)
                    indices = list(range(len(texts)))
                    random.shuffle(indices)
                    segment_length = (len(indices) - 1) // loader_scatter + 1
                    for i in range(loader_scatter):
                        scatter_path = get_scatter_path(path, scatter_rank=i)
                        prompt_writer = LazyWriter(scatter_path, data_type='prompt', is_array=pre_tokenize)
                        text_writer = LazyWriter(scatter_path, data_type='text', is_array=pre_tokenize)
                        for idx in indices[i * segment_length: (i + 1) * segment_length]:
                            prompt_writer.write(prompts[idx])
                            text_writer.write(texts[idx])
                        prompt_writer.close()
                        text_writer.close()
                else:
                    while not (
                            exists_scatter(path, data_type='prompt', scatter_num=loader_scatter) and exists_scatter(
                        path, data_type='text', scatter_num=loader_scatter)):
                        time.sleep(1)
            scatter_path = get_scatter_path(path, scatter_rank=data_parallel_rank % loader_scatter)
            print(f"Rank {global_rank} is using scatter from {scatter_path}")
            prompts = LazyLoader(scatter_path, data_type='prompt', map_fn=map_fn, mem_map=True,
                                 is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
            texts = LazyLoader(scatter_path, data_type='text', map_fn=map_fn, mem_map=True,
                               is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
        else:
            prompts = LazyLoader(path, data_type='prompt', map_fn=map_fn, mem_map=True,
                                 is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
            texts = LazyLoader(path, data_type='text', map_fn=map_fn, mem_map=True,
                               is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
        text = corpora.PromptDataset(prompt_loader=prompts, text_loader=texts, tokenizer=tokenizer,
                                     to_tokenize=not pre_tokenize)
        if loader_scatter is None:
            if global_rank == 0:
                print(f"Create dataset {name} with {len(text)} documents")
                for i in range(10):
                    rand_id = i if i < 5 else random.randrange(len(text))
                    sample_tokens = text[rand_id]['tokens'][:1024]
                    print(sample_tokens)
                    print(tokenizer.DecodeIds(sample_tokens).encode('utf-8'))
        else:
            for scatter_id in range(loader_scatter):
                if data_parallel_rank % loader_scatter == scatter_id and data_parallel_rank // loader_scatter == 0:
                    print(f"Create dataset {name} at scatter {scatter_id} with {len(text)} documents")
                    for i in range(10):
                        sample_tokens = text[i]['tokens'][:1024]
                        print(sample_tokens)
                        print(tokenizer.DecodeIds(sample_tokens))
                torch.distributed.barrier()
        return text
    elif issubclass(dataset, corpora.KeyReader):
        if not (exists_lazy(path, data_type='text') and exists_lazy(path, data_type='mask')):
            # create cached version of dataset for lazy loading if it doesn't exist
            if global_rank == 0:
                text_writer = LazyWriter(path, data_type='text', is_array=pre_tokenize)
                mask_writer = LazyWriter(path, data_type='mask', is_array=True)
                writers = {'mask': mask_writer, 'text': text_writer}
                dataset(writers=writers, tokenizer=tokenizer, tokenize=pre_tokenize)
                mask_writer.close()
                text_writer.close()
            else:
                while not os.path.exists(LazyWriter.get_len_path(path, data_type='mask')):
                    time.sleep(1)
        map_fn = (lambda x: x.tolist()) if pre_tokenize else None
        masks = LazyLoader(path, data_type='mask', map_fn=map_fn, mem_map=True, is_array=True)
        texts = LazyLoader(path, data_type='text', map_fn=map_fn, mem_map=True, is_array=pre_tokenize)
        text = corpora.KeyDataset(mask_loader=masks, text_loader=texts, tokenizer=tokenizer,
                                  to_tokenize=not pre_tokenize)
        return text


def supported_corpus(corpus_name):
    """checks if corpus name is defined in `corpora.py`"""
    return corpus_name in corpora.NAMED_CORPORA


def make_dataset(path, seq_length, mem_length, shuffle=True, split=None, tokenizer=None,
                 sample_one_document=False, pre_tokenize=False, ds_type='', save_splits=None, load_splits=None,
                 save_test_data=None, no_lazy_loader=False, loader_scatter=None, data_parallel_rank=None,
                 filter_english=False, non_sentence_start=0.0, half_lazy_loader=False, **kwargs):
    """function to create datasets+tokenizers for common options"""
    if split is None:
        split = [1.]

    # get one or multiple datasets and concatenate
    if isinstance(path, str):
        ds = get_dataset(path, tokenizer=tokenizer, pre_tokenize=pre_tokenize, no_lazy_loader=no_lazy_loader,
                         loader_scatter=loader_scatter, data_parallel_rank=data_parallel_rank,
                         half_lazy_loader=half_lazy_loader)
    else:
        ds = [get_dataset(p, tokenizer=tokenizer, pre_tokenize=pre_tokenize, no_lazy_loader=no_lazy_loader,
                          loader_scatter=loader_scatter, data_parallel_rank=data_parallel_rank,
                          half_lazy_loader=half_lazy_loader) for p in path]
        ds = ConcatDataset(ds)

    # Split dataset into train/val/test (and wrap bert dataset)
    def wrap_dataset(dataset):
        if ds_type.lower() == 'bert':
            presplit_sentences = kwargs['presplit_sentences'] if 'presplit_sentences' in kwargs else False
            dataset = BertSentencepairDataset(dataset, max_seq_len=seq_length, presplit_sentences=presplit_sentences)
        elif ds_type.lower() == 'gpt-xl':
            assert pre_tokenize
            dataset = XLDataset(dataset, tokenizer, max_seq_len=seq_length, mem_len=mem_length,
                                sample_across_doc=not sample_one_document)
        elif ds_type.lower() == 'gpt2':
            dataset = GPT2Dataset(dataset, tokenizer, max_seq_len=seq_length, sample_across_doc=not sample_one_document)
        elif ds_type.lower() == 'block':
            dataset = BlockDataset(dataset, tokenizer, max_seq_len=seq_length,
                                   sample_across_doc=not sample_one_document, filter_english=filter_english,
                                   non_sentence_start=non_sentence_start)
        return dataset

    if should_split(split):
        ds = split_ds(ds, split, shuffle=shuffle, save_splits=save_splits, load_splits=load_splits)
        if save_test_data is not None and torch.distributed.get_rank() == 0:
            test_ds = ds[-1]
            with open(save_test_data, "w", encoding='utf-8') as output:
                for data in test_ds:
                    text = data['tokens']
                    text = tokenizer.DecodeIds(text)
                    output.write(text)
                    output.write("\n")
            print(f"Write test data to {save_test_data}")
        ds = [wrap_dataset(d) if d is not None else None for d in ds]
    else:
        ds = wrap_dataset(ds)
    return ds
