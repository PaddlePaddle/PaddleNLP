#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import json
import copy
import sys
from uie.extraction.predict_parser.utils import fix_unk_from_text_with_tokenizer
from uie.seq2seq_paddle.t5_bert_tokenizer import T5BertTokenizer

tokenizer = T5BertTokenizer.from_pretrained(
    "pd_models/t5-char-100g-small-30w-zh_match_0.5-50w/")
to_remove_token_list = list()
if tokenizer.eos_token:
    to_remove_token_list += [tokenizer.eos_token]
if tokenizer.pad_token:
    to_remove_token_list += [tokenizer.pad_token]


def postprocess_text(x_str):
    # Clean `bos` `eos` `pad` for cleaned text
    for to_remove_token in to_remove_token_list:
        x_str = x_str.replace(to_remove_token, '')
    return x_str


def read_span(filename):
    counter = Counter()
    for line in open(filename):
        span_set = list()
        instance = json.loads(line.strip())
        for event in instance['event']:
            span_set.append(event['text'])
            for arg in event['args']:
                span_set.append(arg['text'])
        for span in span_set:
            counter.update(['span'])
            tokenized = tokenizer.encode(span)['input_ids'][:-1]
            decoded_span = postprocess_text(
                tokenizer.decode(
                    tokenized,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False))
            if '<unk>' in decoded_span:
                fixed_unk_span = postprocess_text(
                    fix_unk_from_text_with_tokenizer(
                        decoded_span,
                        instance['text'],
                        unk='<unk>',
                        tokenizer=tokenizer))
                if fixed_unk_span != span:
                    if decoded_span.startswith(
                            '<unk>') or decoded_span.endswith('<unk>'):
                        counter.update(['bound'])
                    else:
                        counter.update(['cannot fix unk'])
                    print('[UNK SPAN]', span)
                    print('[DEC SPAN]', decoded_span)
                    print('[   FIX  ]', fixed_unk_span)
                    print('[  TEXT  ]', instance['text'])
            elif decoded_span != span:
                counter.update(['no unk'])
                # print('[UNK SPAN]', list(span))
                # print('[DEC SPAN]', list(decoded_span))
                # print('[  TEXT  ]', instance['text'])
                # print('============')
    print(counter)


if __name__ == "__main__":
    read_span(sys.argv[1])
