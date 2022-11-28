#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import ast
import time
import argparse
import logging

import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddlenlp.transformers import ErnieForGeneration
from paddlenlp.transformers import ErnieTokenizer, ErnieTinyTokenizer, BertTokenizer, ElectraTokenizer, RobertaTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import Rouge1, Rouge2
from paddlenlp.utils.log import logger

from encode import convert_example, after_padding
from decode import beam_search_infilling, post_process, greedy_search_infilling

# yapf: disable
parser = argparse.ArgumentParser('seq2seq model with ERNIE-GEN')
parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: "+ ", ".join(list(ErnieTokenizer.pretrained_init_configuration.keys())))
parser.add_argument('--max_encode_len', type=int, default=24, help="The max encoding sentence length")
parser.add_argument('--max_decode_len', type=int, default=72, help="The max decoding sentence length")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument('--beam_width', type=int, default=3, help="Beam search width")
parser.add_argument('--length_penalty', type=float, default=1.0, help="The length penalty during decoding")
parser.add_argument('--init_checkpoint', type=str, default=None, help='Checkpoint to warm start from')
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
# yapf: enable

args = parser.parse_args()


def predict():
    paddle.set_device(args.device)

    model = ErnieForGeneration.from_pretrained(args.model_name_or_path)
    if "ernie-tiny" in args.model_name_or_path:
        tokenizer = ErnieTinyTokenizer.from_pretrained(args.model_name_or_path)
    elif "ernie" in args.model_name_or_path:
        tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)
    elif "roberta" in args.model_name_or_path or "rbt" in args.model_name_or_path:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    elif "electra" in args.model_name_or_path:
        tokenizer = ElectraTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    dev_dataset = load_dataset('poetry', splits=('dev'), lazy=False)
    attn_id = tokenizer.vocab[
        '[ATTN]'] if '[ATTN]' in tokenizer.vocab else tokenizer.vocab['[MASK]']
    tgt_type_id = model.sent_emb.weight.shape[0] - 1

    trans_func = convert_example(tokenizer=tokenizer,
                                 attn_id=attn_id,
                                 tgt_type_id=tgt_type_id,
                                 max_encode_len=args.max_encode_len,
                                 max_decode_len=args.max_decode_len)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_sids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_sids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # attn_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_labels
    ): after_padding(fn(samples))

    dev_dataset = dev_dataset.map(trans_func)
    test_batch_sampler = paddle.io.BatchSampler(dev_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False)
    data_loader = DataLoader(dataset=dev_dataset,
                             batch_sampler=test_batch_sampler,
                             collate_fn=batchify_fn,
                             num_workers=0,
                             return_list=True)

    if args.init_checkpoint:
        model_state = paddle.load(args.init_checkpoint)
        model.set_state_dict(model_state)

    model.eval()
    vocab = tokenizer.vocab
    eos_id = vocab[tokenizer.sep_token]
    sos_id = vocab[tokenizer.cls_token]
    pad_id = vocab[tokenizer.pad_token]
    unk_id = vocab[tokenizer.unk_token]
    vocab_size = len(vocab)
    evaluated_sentences = []
    evaluated_sentences_ids = []
    logger.info("Predicting...")
    for data in data_loader:
        (src_ids, src_sids, src_pids, _, _, _, _, _, _, _, _,
         raw_tgt_labels) = data  # never use target when infer
        # Use greedy_search_infilling or beam_search_infilling to get predictions
        output_ids = beam_search_infilling(model,
                                           src_ids,
                                           src_sids,
                                           eos_id=eos_id,
                                           sos_id=sos_id,
                                           attn_id=attn_id,
                                           pad_id=pad_id,
                                           unk_id=unk_id,
                                           vocab_size=vocab_size,
                                           max_decode_len=args.max_decode_len,
                                           max_encode_len=args.max_encode_len,
                                           beam_width=args.beam_width,
                                           length_penalty=args.length_penalty,
                                           tgt_type_id=tgt_type_id)

        for source_ids, target_ids, predict_ids in zip(
                src_ids.numpy().tolist(),
                raw_tgt_labels.numpy().tolist(), output_ids.tolist()):
            if eos_id in predict_ids:
                predict_ids = predict_ids[:predict_ids.index(eos_id)]
            source_sentence = ''.join(
                map(post_process,
                    vocab.to_tokens(source_ids[1:source_ids.index(eos_id)])))
            tgt_sentence = ''.join(
                map(post_process,
                    vocab.to_tokens(target_ids[1:target_ids.index(eos_id)])))
            predict_ids = ''.join(
                map(post_process, vocab.to_tokens(predict_ids)))
            print("source :%s\ntarget :%s\npredict:%s\n" %
                  (source_sentence, tgt_sentence, predict_ids))


if __name__ == "__main__":
    predict()
