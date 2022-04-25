#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from tabulate import tabulate
from tqdm import tqdm
from uie.seq2seq.t5_bert_tokenizer import T5BertTokenizer
from uie.extraction.dataset_processer import PrefixGenerator
from uie.extraction.record_schema import RecordSchema


def find_key(count):
    if count > 512:
        return '7.>512'
    elif 384 < count <= 512:
        return "6.384-512"
    elif 320 < count <= 384:
        return "5.320-384"
    elif 256 < count <= 320:
        return "4.256-320"
    elif 192 < count <= 256:
        return "3.192-256"
    elif 128 < count <= 192:
        return "2.128-192"
    elif 64 < count <= 128:
        return "1. 64-128"
    elif count == 0:
        return "8. =0"
    else:
        return "0.    <64"


def get_acc_list(counter):
    sum_instance = float(sum(counter.values()))
    acc_list = list()
    acc_counter = defaultdict(int)
    for k in sorted(counter.keys()):
        v = counter[k]
        acc_counter[find_key(k)] += v
    acc = 0
    for k in sorted(acc_counter.keys()):
        acc += acc_counter[k]
        acc_list += [(k, acc, "%.2f" % (acc / sum_instance * 100))]
    return acc_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True, nargs='+')
    parser.add_argument('-tokenize', default='hf_models/t5-small')
    parser.add_argument('-fast', action='store_true')
    parser.add_argument('-key', default='record')
    options = parser.parse_args()

    if "t5-char" in options.tokenize:
        tokenizer = T5BertTokenizer.from_pretrained(options.tokenize)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            options.tokenize, use_fast=options.fast)
    print("Load tokenize: ", options.tokenize)

    to_add_special_token = list()
    for special_token in [
            '<extra_id_0>', '<extra_id_1>', '<extra_id_2>', '<extra_id_3>',
            '<extra_id_4>', '<extra_id_5>', '<spot>', '<asoc>'
    ]:
        if special_token not in tokenizer.get_vocab():
            to_add_special_token += [special_token]
    tokenizer.add_special_tokens({
        "additional_special_tokens": to_add_special_token
    })

    for data_folder in options.data:
        print(data_folder)

        record_schema = RecordSchema.read_from_file(data_folder +
                                                    '/record.schema')
        schema_prefix = PrefixGenerator.get_schema_prefix(record_schema)
        len_schema_prefix = len(tokenizer.tokenize(schema_prefix))
        print("Schema Propmt: %s" % schema_prefix)
        print("Schema Propmt After Toknized: %s" %
              tokenizer.tokenize(schema_prefix))
        print("Schema Prompt Length: %s" % len_schema_prefix)
        for file_type in {"train", "val", "test", "align"}:
            counter = defaultdict(Counter)
            filename = os.path.join(data_folder, file_type + '.json')
            if not os.path.exists(filename):
                print('Skip %s' % filename)
                continue

            for line in tqdm(open(filename).readlines(), unit='line'):
                instance = json.loads(line)
                text = instance['text']
                record = instance[options.key]
                counter['Text'].update([len(tokenizer.tokenize(text))])
                counter['Record'].update([len(tokenizer.tokenize(record))])
                counter['Text + Schema'].update(
                    [len(tokenizer.tokenize(text)) + len_schema_prefix])
                counter['Record + Schema Prompt'].update(
                    [len(tokenizer.tokenize(record)) + len_schema_prefix])
                if len(tokenizer.tokenize(record)) > 512:
                    print("[Length > 512 Text  ]:", text)
                    print("[Length > 512 Record]:", record)

            for k, v in counter.items():
                print(file_type, k)
                table = get_acc_list(v)
                print(tabulate(table))
                print(f"Min: {min(v.keys())}")
                print(f"Max: {max(v.keys())}")


if __name__ == "__main__":
    main()
