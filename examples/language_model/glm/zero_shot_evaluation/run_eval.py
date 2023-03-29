# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import json
import math
import os
import re
import time

from itertools import accumulate
from bisect import bisect_right

import numpy as np
import paddle
from paddle.io import DataLoader

from paddlenlp.data import Stack, Tuple
from paddlenlp.transformers import GLMModel #, GLMTokenizer
from paddlenlp.utils.log import logger

from tokenizer.tokenization import GPT2BPETokenizer
from tokenizer.utils import print_rank_0
from tokenizer.data_utils import build_input_from_ids, num_special_tokens_to_add 

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=None, type=str, required=True, help="Path to pre-trained model")
parser.add_argument("--eval_path", default=None, type=str, required=True, help="The eval file path.", )
parser.add_argument('--cloze_eval', action='store_true', help='Evaluation dataset from `--eval_path` is a cloze task.')
parser.add_argument('--overlapping_eval', type=int, default=32, help='Sliding window for overlapping eval.')
parser.add_argument("--init_checkpoint_path", default=None, type=str, help="The model checkpoint path.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--seq_length', type=int, default=512, help='Maximum sequence length to process for evaluation.')
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu", "xpu", "npu"], help="Select cpu, gpu, xpu, npu devices.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
# yapf: enable

token_kwargs = {'add_block_symbols': True, 'cache_dir': None, 'add_sentinel_token': 0, 'add_task_mask': True, 'add_decoder_mask': False}
wiki_token_kwargs = {'add_block_symbols': True, 'cache_dir': None, 'add_sentinel_token': 0, 'add_task_mask': True, 'add_decoder_mask': False}


class LMDataset(paddle.io.Dataset):
    def __init__(self, args, documents, tokenizer, num_original_tokens, num_tokenized_tokens):
        self.args = args
        self.documents = documents
        self.max_seq_len = args.seq_length - 1
        self.tokenizer = tokenizer
        self.overalapping_eval = args.overlapping_eval
        if self.overalapping_eval is None:
            self.overalapping_eval = self.max_seq_len
        self.overalapping_eval = max(1, self.overalapping_eval)
        self.num_original_tokens = num_original_tokens
        self.num_tokenized_tokens = num_tokenized_tokens
        # remove first sequence tokens
        targets = [max(len(tokens) - self.max_seq_len, 0) for tokens in self.documents]
        self.num_sequences = [max(math.ceil(target / self.overalapping_eval) + 1, 1) for target in targets]
        self.weights = list(accumulate(self.num_sequences))
        self.left_weights = [0] + self.weights[:-1]
        self.unidirectional = args.unidirectional
        self.block_lm = args.block_lm
        mask_token = "gMASK" if args.task_mask else 'MASK'
        self.mask_id = self.tokenizer.get_command(mask_token).Id

    def __len__(self):
        return sum(self.num_sequences)

    def __getitem__(self, idx):
        document_idx = bisect_right(self.weights, idx)
        idx = idx - self.left_weights[document_idx]
        start_idx = idx * self.overalapping_eval
        end_idx = start_idx + self.max_seq_len
        tokens = self.documents[document_idx][start_idx:end_idx]
        if self.block_lm:
            if idx == 0 or self.unidirectional:
                prompt, text = tokens[:1], tokens[1:]
            else:
                prompt_length = self.max_seq_len - self.overalapping_eval
                prompt, text = tokens[:prompt_length], tokens[prompt_length:]
            prompt = prompt + [self.mask_id]
            num_special_tokens = num_special_tokens_to_add(prompt, None, text, add_cls=True, add_sep=False,
                                                           add_piece=True,
                                                           add_eos=False)
            data = build_input_from_ids(prompt, None, text, self.max_seq_len + num_special_tokens + 1, self.tokenizer,
                                        args=self.args, add_cls=True, add_sep=False, add_piece=True, add_eos=False, mask_id=self.mask_id)
            ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
            if idx != 0 and self.unidirectional:
                loss_masks = np.array(loss_masks, dtype=np.int64)
                loss_masks[:-self.overalapping_eval] = 0
            return [np.array(ids, dtype=np.int64), np.array(loss_masks, dtype=np.int64), np.array(sep, dtype=np.int64), np.array(position_ids, dtype=np.int64), np.array(target_ids, dtype=np.int64)]
            return {'text': np.array(ids, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                    'attention_mask': np.array(sep, dtype=np.int64), 'loss_mask': np.array(loss_masks, dtype=np.int64),
                    "position_id": np.array(position_ids, dtype=np.int64)}
        else:
            loss_masks = [1] * len(tokens)
            if len(tokens) < self.max_seq_len:
                tokens = tokens + [0] * (self.max_seq_len - len(tokens))
                loss_masks = loss_masks + [0] * (self.max_seq_len - len(loss_masks))
            if idx != 0:
                loss_masks = np.array(loss_masks, dtype=np.int64)
                loss_masks[:-self.overalapping_eval] = 0
            return {'text': np.array(tokens, dtype=np.int64), 'loss_mask': np.array(loss_masks, dtype=np.int64)}


class LM_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, tokens, seq_len, pad_idx, mask_idx, sop_idx, cls_idx, overlapping_eval=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.sop_idx = sop_idx
        self.cls_idx = cls_idx
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
        loss_mask = np.ones(seq_length, dtype="int64")
        loss_mask[np.where(np.array(tokens) == self.pad_idx)] = 0
        position_ids = np.stack([
            np.arange(0, seq_length, dtype="int64"),
            np.zeros(seq_length, dtype="int64")
        ], axis=0)

        # -INF mask value as default
        # attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        start_idx = idx * self.overlapping_eval
        end_idx = start_idx + self.seq_len - 4
        tokens = self.tokens[start_idx : end_idx + 1]
        num_tokens = len(tokens)
        
        src_tokens, tgt_tokens = tokens[:-1], tokens[-1:]
        src_tokens = [self.cls_idx] + src_tokens + [self.mask_idx, self.pad_idx, self.sop_idx]
        content_length = len(src_tokens)
        input_tokens = src_tokens + tgt_tokens
        pad_length = self.seq_len - len(src_tokens)
        if pad_length > 0:
            input_tokens = input_tokens + [self.pad_idx] * pad_length

        position_ids = np.stack([
            np.arange(len(input_tokens), dtype="int64"),
            np.zeros(len(input_tokens), dtype="int64"),
        ])
        input_tokens = np.array(input_tokens)
        attention_mask = np.array([len(src_tokens)])
        loss_mask = np.zeros(len(input_tokens)) 
        loss_mask[len(src_tokens):len(src_tokens) + len(tgt_tokens)] = 1
        if pad_length > 0:
            labels = np.array([0] * len(src_tokens) + tgt_tokens + [0] * pad_length)
        else:
            labels = np.array([0] * len(src_tokens) + tgt_tokens)

        return [input_tokens, loss_mask, attention_mask, position_ids, labels]


class Lambada_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, data_path, tokenizer, seq_len):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def _load_data(self, data_path):
        with open(data_path, "r", encoding="utf-8") as fp:
            data = [json.loads(x)["text"] for x in fp]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx].split()
        src_text, tgt_text = "".join(example[:-1]), example[-1]
        src_tokens = self.tokenizer(src_text.strip(), add_special_tokens=False)["input_ids"]
        tgt_tokens = self.tokenizer(tgt_text, add_special_tokens=False)["input_ids"]

        # create input_ids
        max_src_length = self.seq_len - len(tgt_tokens) - 3
        src_tokens = src_tokens[-max_src_length:]
        src_tokens = [self.tokenizer.cls_token_id] + src_tokens + [self.tokenizer.gmask_token_id, self.tokenizer.eos_token_id]
        tgt_tokens = [self.tokenizer.sop_token_id] + tgt_tokens
        input_tokens = src_tokens + tgt_tokens
        pad_length = self.seq_len - len(input_tokens)
        if pad_length > 0:
            input_tokens = input_tokens + [self.tokenizer.pad_token_id] * pad_length
        input_tokens[len(src_tokens) + len(tgt_tokens) - 1] = self.tokenizer.pad_token_id

        # create position_ids
        mask_position = input_tokens.index(self.tokenizer.gmask_token_id)
        position_ids = np.concatenate([
            np.arange(len(src_tokens), dtype="int64"),
            np.full([len(input_tokens) - len(src_tokens)], mask_position, dtype="int64")
        ])
        block_position_ids = np.concatenate([
            np.zeros(len(src_tokens), dtype="int64"),
            np.arange(1, len(input_tokens) - len(src_tokens) + 1, dtype="int64")
        ])
        position_ids = np.stack([position_ids, block_position_ids]) 
       
        # create loss_mask 
        loss_mask = np.zeros(len(input_tokens))
        loss_mask[len(src_tokens): len(src_tokens) + len(tgt_tokens) - 1] = 1

        # create attention_mask
        attention_mask = np.array([len(src_tokens)])

        # create labels
        labels = np.array([0] * (len(src_tokens) - 1) + tgt_tokens + [self.tokenizer.pad_token_id] * (pad_length + 1))
        input_tokens = np.array(input_tokens)
        
        return [input_tokens, loss_mask, attention_mask, position_ids, labels]
    

class Namespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(Namespace, k, v)

class LambadaDataset(paddle.io.Dataset):
    def __init__(self, args, tokenizer, strict=True):
        self.args = args
        data_path = ".//lambada_test.jsonl"
        print_rank_0('> building lambada dataset from {} ...'.format(data_path))
        self.max_seq_length = 512
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.get_command('pad').Id
        self.strict = strict
        self.block_lm = True
        self.unidirectional = False
        mask_token = "gMASK"
        self.mask_id = self.tokenizer.get_command(mask_token).Id

        self.tokens = []
        self.labels = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = self.get_tokens(text)
                self.tokens.append(tokens)
                self.labels.append(labels)

    def get_tokens(self, text):
        if not self.strict:
            tokens = self.tokenizer.EncodeAsIds(text).tokenization
            return tokens[:-1], [tokens[-1]]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)
        beginning_tokens = self.tokenizer.EncodeAsIds(text[:start_idx].strip()).tokenization
        last_token = self.tokenizer.EncodeAsIds(' ' + last_token).tokenization
        return beginning_tokens, last_token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens, answer = self.tokens[idx], self.labels[idx]
        if self.block_lm:
            if self.unidirectional:
                tokens, answer_tokens = tokens[:1], tokens[1:] + answer
            else:
                answer_tokens = answer
            tokens = tokens + [self.mask_id]
            num_special_tokens = num_special_tokens_to_add(tokens, None, answer_tokens, add_cls=True, add_sep=False,
                                                           add_piece=True)
            left_shift = len(tokens) + len(answer_tokens) + num_special_tokens - self.max_seq_length
            if left_shift > 0:
                tokens = tokens[left_shift:]
            data = build_input_from_ids(tokens, None, answer_tokens, self.max_seq_length, self.tokenizer,
                                        args=self.args, add_cls=True, add_sep=False, add_piece=True,
                                        mask_id=self.mask_id)
            ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
            if self.unidirectional:
                loss_masks = np.array(loss_masks, dtype=np.int64)
                last_index = len(loss_masks)
                while loss_masks[last_index - 1] == 0:
                    last_index -= 1
                loss_masks[:last_index - len(answer)] = 0
            return [np.array(ids, dtype=np.int64), np.array(loss_masks, dtype=np.int64), np.array(sep, dtype=np.int64), np.array(position_ids, dtype=np.int64), np.array(target_ids, dtype=np.int64)]
            return {'input_ids': np.array(ids, dtype=np.int64), 'labels': np.array(target_ids, dtype=np.int64),
                    'attention_mask': np.array(sep, dtype=np.int64), 'loss_mask': np.array(loss_masks, dtype=np.int64),
                    "position_ids": np.array(position_ids, dtype=np.int64)}
        else:
            left_shift = len(tokens) - self.max_seq_length
            if left_shift > 0:
                tokens = tokens[left_shift:]
            ids = tokens + answer
            if len(ids) < self.max_seq_length:
                ids = ids + [0] * (self.max_seq_length - len(ids))
            loss_masks = [0] * len(tokens) + [1] * len(answer)
            if len(loss_masks) < self.max_seq_length:
                loss_masks = loss_masks + [0] * (self.max_seq_length - len(loss_masks))
            return {'input_ids': np.array(ids, dtype=np.int64), 'loss_mask': np.array(loss_masks, dtype=np.int64)}
        

def wikitext_detokenizer(string):
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


def get_tokens(tokenizer, text, strict=True):
    if not strict:
        tokens = tokenizer(text)["input_ids"]
        return tokens[:-1], [tokens[-1]]
    last_token = text.split()[-1]
    start_idx = text.rfind(last_token)
    beginning_tokens = tokenizer(" " + text[:start_idx].strip())["input_ids"]
    last_token = tokenizer(" " + last_token, add_special_tokens=False)["input_ids"]
    return beginning_tokens, last_token


def create_eval_dataset(args):
    val_dataloader = None
    eval_batch_size = args.batch_size
    seq_len = args.seq_length

    tokenizer = GPT2BPETokenizer('gpt2', **token_kwargs)
    #tokenizer = GLMTokenizer.from_pretrained(args.model_name)
    if not args.cloze_eval:
        with open(args.eval_path, "rb") as reader:
            entire_data = reader.read().decode("utf-8")
        num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = wikitext_detokenizer(entire_data)
        #tokenized_data = tokenizer(entire_data, add_special_tokens=False)["input_ids"]
        tokenized_data = tokenizer.EncodeAsIds(entire_data).tokenization
        num_tokenized_tokens = len(tokenized_data)
        print("Original Tokens: %d, Detokenized tokens: %d" % (num_tokenized_tokens, num_original_tokens))
        token_args = Namespace(DDP_impl='torch', adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-08, adapet=False, attention_dropout=0.1, attention_scale=1.0, avg_block_length=3, batch_size=4, bert_prob=0.5, blank_maskratio=0.1, block_lm=True, block_lm_ratio=0.0, block_mask_prob=0.0, cache_dir=None, checkpoint_activations=True, checkpoint_num_layers=1, clip_grad=1.0, cloze_eval=True, context_mask_ratio=0.0, continuous_prompt=False, cpu_optimizer=False, cpu_torch_adam=False, cuda=True, data_dir=None, deep_init=False, deepscale=False, deepscale_config=None, deepspeed=True, deepspeed_activation_checkpointing=False, deepspeed_config=None, deepspeed_mpi=False, delim=',', distributed_backend='nccl', dynamic_loss_scale=True, encoder_decoder=False, eod_token=50256, epochs=None, eval_batch_size=16, eval_epoch=1, eval_interval=1000, eval_iters=100, eval_max_preds_per_seq=None, eval_seq_length=None, eval_text_key=None, eval_valid=False, experiment_name='blocklm-2B-wikitext', fast_decode=False, few_superglue=False, filter_english=False, finetune=True, fix_command_token=False, fp16=True, fp32_allreduce=False, fp32_embedding=False, fp32_layernorm=False, fp32_tokentypes=False, freeze_transformer=False, gap_sentence_prob=0.0, gap_sentence_ratio=0.15, gpt_infill_prob=0.5, gpt_min_ratio=0.5, gradient_accumulation_steps=1, half_lazy_loader=False, hidden_dropout=0.1, hidden_size=2048, hysteresis=2, input_data_sizes_file='sizes.txt', intermediate_size=None, label_smoothing=0.0, layernorm_epsilon=1e-05, length_penalty=0.0, load=None, load_pretrained='.//blocklm-2b-512', load_splits=None, loader_scatter=None, local_rank=0, log_interval=100, loose_json=False, loss_func='cross_entropy', loss_scale=None, loss_scale_window=1000, lr=0.0001, lr_decay_iters=None, lr_decay_ratio=0.1, lr_decay_style='linear', make_vocab_size_divisible_by=128, masked_lm=False, master_ip='localhost', master_port='38792', max_position_embeddings=1024, max_preds_per_seq=None, mem_length=0, min_scale=1, min_tgt_length=0, model_parallel_size=1, multi_batch_size=None, multi_seq_length=None, multi_task_data=None, multi_task_ratio=0.0, multi_token=False, new_save_directory=False, no_block_position=False, no_deepspeed_load=False, no_lazy_loader=False, no_load_lr_scheduler=False, no_load_optim=False, no_load_rng=False, no_pre_tokenize=False, no_repeat_ngram_size=0, no_save_optim=False, no_save_rng=False, no_shuffle_block=False, no_validation=False, non_sentence_start=0.0, num_attention_heads=32, num_beams=1, num_layers=36, num_prompt_tokens=0, num_workers=2, optimizer='adam', out_seq_length=256, output_dropout=0.1, overlapping_eval=256, overwrite=True, pattern_id=0, pool_token='cls', prefix_prompt=0, presplit_sentences=False, pretrained_bert=False, prompt_func='lstm', prompt_init=False, random_position=False, rank=0, reset_attention_mask=False, reset_position_ids=False, resume_dataloader=False, sample_one_document=False, save='./blocklm-2B-wikitext', save_epoch=1, save_interval=5000, save_splits=None, save_test_data=None, seed=1234, segment_length=0, select_topk=False, sentinel_token=False, seq_length=1024, short_seq_prob=0.0, shuffle=False, single_span_prob=0.0, split='1000,1,1', src_seq_length=None, summary_dir='', switch_linear=False, task='wikitext', task_mask=True, temperature=1.0, test_data=None, text_key='sentence', tgt_seq_length=None, tokenizer_model_type=None, tokenizer_path='tokenizer.model', tokenizer_type='GPT2BPETokenizer', top_k=0, top_p=0.0, train_data=None, train_iters=0, transformer_xl=False, tune_prefix_layers=None, unidirectional=False, use_tfrecords=False, valid_data=['./wiki.test.tokens'], validation_metric=None, vocab_size=50304, warmup=0.01, weight_decay=0.01, world_size=1, wsc_negative=False)
        val_dataset = LMDataset(token_args, [tokenized_data], tokenizer, num_original_tokens, num_tokenized_tokens) 
        #val_dataset = LM_Eval_Dataset(
        #    tokenized_data,
        #    seq_len,
        #    tokenizer.pad_token_id,
        #    tokenizer.gmask_token_id,
        #    tokenizer.sop_token_id,
        #    tokenizer.cls_token_id,
        #    args.overlapping_eval
        #)
    else:
        #val_dataset = Lambada_Eval_Dataset(
        #    args.eval_path,
        #    tokenizer,
        #    seq_len,
        #)
        token_args = Namespace(DDP_impl='torch', adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-08, adapet=False, attention_dropout=0.1, attention_scale=1.0, avg_block_length=3, batch_size=4, bert_prob=0.5, blank_maskratio=0.1, block_lm=True, block_lm_ratio=0.0, block_mask_prob=0.0, cache_dir=None, checkpoint_activations=True, checkpoint_num_layers=1, clip_grad=1.0, cloze_eval=True, context_mask_ratio=0.0, continuous_prompt=False, cpu_optimizer=False, cpu_torch_adam=False, cuda=True, data_dir=None, deep_init=False, deepscale=False, deepscale_config=None, deepspeed=True, deepspeed_activation_checkpointing=False, deepspeed_config=None, deepspeed_mpi=False, delim=',', distributed_backend='nccl', dynamic_loss_scale=True, encoder_decoder=False, eod_token=50256, epochs=None, eval_batch_size=16, eval_epoch=1, eval_interval=1000, eval_iters=100, eval_max_preds_per_seq=None, eval_seq_length=None, eval_text_key=None, eval_valid=False, experiment_name='blocklm-2B-lambda', fast_decode=False, few_superglue=False, filter_english=False, finetune=True, fix_command_token=False, fp16=True, fp32_allreduce=False, fp32_embedding=False, fp32_layernorm=False, fp32_tokentypes=False, freeze_transformer=False, gap_sentence_prob=0.0, gap_sentence_ratio=0.15, gpt_infill_prob=0.5, gpt_min_ratio=0.5, gradient_accumulation_steps=1, half_lazy_loader=False, hidden_dropout=0.1, hidden_size=2048, hysteresis=2, input_data_sizes_file='sizes.txt', intermediate_size=None, label_smoothing=0.0, layernorm_epsilon=1e-05, length_penalty=0.0, load=None, load_pretrained='.//blocklm-2b-512', load_splits=None, loader_scatter=None, local_rank=0, log_interval=100, loose_json=False, loss_func='cross_entropy', loss_scale=None, loss_scale_window=1000, lr=0.0001, lr_decay_iters=None, lr_decay_ratio=0.1, lr_decay_style='linear', make_vocab_size_divisible_by=128, masked_lm=False, master_ip='localhost', master_port='38792', max_position_embeddings=1024, max_preds_per_seq=None, mem_length=0, min_scale=1, min_tgt_length=0, model_parallel_size=1, multi_batch_size=None, multi_seq_length=None, multi_task_data=None, multi_task_ratio=0.0, multi_token=False, new_save_directory=False, no_block_position=False, no_deepspeed_load=False, no_lazy_loader=False, no_load_lr_scheduler=False, no_load_optim=False, no_load_rng=False, no_pre_tokenize=False, no_repeat_ngram_size=0, no_save_optim=False, no_save_rng=False, no_shuffle_block=False, no_validation=False, non_sentence_start=0.0, num_attention_heads=32, num_beams=1, num_layers=36, num_prompt_tokens=0, num_workers=2, optimizer='adam', out_seq_length=256, output_dropout=0.1, overlapping_eval=32, overwrite=True, pattern_id=0, pool_token='cls', prefix_prompt=0, presplit_sentences=False, pretrained_bert=False, prompt_func='lstm', prompt_init=False, random_position=False, rank=0, reset_attention_mask=False, reset_position_ids=False, resume_dataloader=False, sample_one_document=False, save='./blocklm-2B-lambda', save_epoch=1, save_interval=5000, save_splits=None, save_test_data=None, seed=1234, segment_length=0, select_topk=False, sentinel_token=False, seq_length=512, short_seq_prob=0.0, shuffle=False, single_span_prob=0.0, split='1000,1,1', src_seq_length=None, summary_dir='', switch_linear=False, task='lambda', task_mask=True, temperature=1.0, test_data=None, text_key='sentence', tgt_seq_length=None, tokenizer_model_type=None, tokenizer_path='tokenizer.model', tokenizer_type='GPT2BPETokenizer', top_k=0, top_p=0.0, train_data=None, train_iters=0, transformer_xl=False, tune_prefix_layers=None, unidirectional=False, use_tfrecords=False, valid_data=['.//lambada_test.jsonl'], validation_metric=None, vocab_size=50304, warmup=0.01, weight_decay=0.01, world_size=1, wsc_negative=False)
        val_dataset = LambadaDataset(token_args, tokenizer, strict=True)
        
        num_tokenized_tokens = 0
        num_original_tokens = 0

    args.num_examples = len(val_dataset)
    args.num_original_tokens = num_original_tokens
    args.num_tokenized_tokens = num_tokenized_tokens
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        drop_last=False,
        collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()),
    )

    return val_dataloader


def do_eval(args):
    paddle.set_device(args.device)
    def empty_func(self, *args, **kwargs):
        return

    # 屏蔽init_weights 
    GLMModel.init_weights = empty_func
    paddle.set_default_dtype("float16")

    #tokenizer = GLMTokenizer.from_pretrained(args.model_name)
    tokenizer = GPT2BPETokenizer('gpt2', **token_kwargs)

    print(args.model_name)
    if args.init_checkpoint_path is not None:
        print('step 1')
        model = GLMModel.from_pretrained(args.model_name, dtype="float16", low_cpu_mem_usage=True)
        #model = GLMModel.from_pretrained(args.model_name)

        logger.info("Load model checkpoint from %s" % args.init_checkpoint_path)
        model_dict = paddle.load(os.path.join(args.init_checkpoint_path))
        model.set_dict(model_dict)
    else:
        print('step 2')
        model = GLMModel.from_pretrained(args.model_name, dtype="float16", low_cpu_mem_usage=True)
        #model = GLMModel.from_pretrained(args.model_name)

    tic_eval = time.time()
    eval_data_loader = create_eval_dataset(args)
    model.eval()
    total_score = 0
    score_name = "loss" if not args.cloze_eval else "number correct"
    with paddle.no_grad():
        for step, batch in enumerate(eval_data_loader):
            tokens, loss_mask, attention_mask, position_ids, labels = batch
            #print(tokens.tolist())
            #print(position_ids.tolist())
            #print(attention_mask.tolist())
            preds = model(tokens, position_ids, attention_mask)
            if isinstance(preds, tuple):
                preds = preds[0]
            #print(preds.sum())
            if not args.cloze_eval:
                masked_lm_loss = paddle.nn.functional.cross_entropy(preds, labels, reduction="none")
                loss = paddle.sum(masked_lm_loss * loss_mask)
                total_score += loss.numpy() / (args.num_tokenized_tokens - 1)
            else:
                outputs = paddle.argmax(preds, -1)
                #for i, m, x, y, a in zip(tokens, loss_mask, outputs, labels, attention_mask):
                    #print("inputs", i[m > 0])
                    #print("preds", x[m > 0])
                    #print("label", y[m > 0])
                acc = paddle.cast(outputs == labels, "float32")
                acc = paddle.where(paddle.cast(loss_mask, "bool"), acc, paddle.ones_like(acc)) 
                acc = paddle.sum(paddle.prod(acc, -1)) 
                total_score += acc.numpy()
            if step % args.logging_steps == 0:
                logger.info(
                    "step %d, batch: %d, %s: %f, speed: %.2f step/s"
                    % (step, step, score_name, total_score, args.logging_steps / (time.time() - tic_eval))
                )
                tic_eval = time.time()

    if not args.cloze_eval:
        total_loss = float(total_score)
        ppl = math.exp(min(20, total_loss))
        token_ratio = (args.num_tokenized_tokens - 1) / (args.num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, total_loss * token_ratio))
        string = " validation results on {} | ".format(args.eval_path)
        string += "avg loss: {:.4E} | ".format(total_loss)
        string += "ppl: {:.4E} | ".format(ppl)
        string += "adjusted ppl: {:.4E} | ".format(adjusted_ppl)
        string += "token ratio: {} |".format(token_ratio)
    else:
        num_correct = float(total_score)
        acc = float(num_correct / args.num_examples)
        string = " validation results on {} | ".format(args.eval_path)
        string += "number correct: {:.4E} | ".format(num_correct)
        string += "total examples: {:.4E} | ".format(args.num_examples)
        string += "avg accuracy: {:.4E}".format(acc)
    logger.info(string)


def run():
    args = parser.parse_args()
    do_eval(args)


if __name__ == "__main__":
    run()
