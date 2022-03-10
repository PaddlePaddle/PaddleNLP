'''
@Author: your name
@Date: 2021-12-08 13:59:15
@LastEditTime: 2021-12-08 13:59:16
@LastEditors: your name
@Description: In User Settings Edit
@FilePath: /shenyaozong/baidu/personal-code/shenyaozong/Interpret_tasks/similarity-ch/simnet/gen_vocab.py
'''
#!/usr/bin/env python
# coding=utf-8

# @Copyright (c) 2019 Baidu.com, Inc. All rights reserved
# @Author: zhangshuai28@baidu.com
# @Date: 2021-10-12 16:58:42
# @LastEditTime: 2021-10-12 17:17:29
# @Description:
import sys
sys.path.append("../../..")
from paddlenlp.datasets import load_dataset
from collections import defaultdict
import spacy

if sys.argv[1] == 'ch':
    train_ds, dev_ds, test_ds = load_dataset(
        "lcqmc", splits=["train", "dev", "test"])

    vocab = defaultdict(int)
    for example in train_ds.data:
        query = example['query']
        title = example['title']
        for c in query:
            vocab[c] += 1
        for c in title:
            vocab[c] += 1
    with open("vocab.char", "w") as f:
        for k, v in vocab.items():
            if v > 3:
                f.write(k + '\n')
                
else:
    tokenizer = spacy.load('en_core_web_sm')
    vocab = defaultdict(int)

    with open('../data/QQP/train/train.tsv', 'r') as f_dataset:
        for idx, line in enumerate(f_dataset.readlines()):
            if idx == 0:
                continue
            line_split = line.strip().split('\t')
            query = [token.text for token in tokenizer(line_split[0])]
            title = [token.text for token in tokenizer(line_split[1])]

            for word in query:
                vocab[word] += 1
            for word in title:
                vocab[word] += 1
        
    with open("vocab_QQP", "w") as f:
        for k, v in vocab.items():
            if v > 3:
                f.write(k + '\n')
