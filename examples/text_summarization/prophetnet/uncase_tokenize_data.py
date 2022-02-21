import argparse
import os

import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer

from paddlenlp.transformers.prophetnet.tokenizer import ProphetNetTokenizer


def uncased_preocess(fin, fout, keep_sep=False, max_len=512):
    tokenizer = ProphetNetTokenizer(vocab_file="prophetnet.tokenizer")
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    twd = TreebankWordDetokenizer()
    for line in tqdm.tqdm(fin.readlines()):
        line = line.strip().replace('``', '"').replace('\'\'',
                                                       '"').replace('`', '\'')
        s_list = [
            twd.detokenize(x.strip().split(' '), convert_parentheses=True)
            for x in line.split('<S_SEP>')
        ]
        if keep_sep:
            output_string = " [X_SEP] ".join(s_list)
        else:
            output_string = " ".join(s_list)
        encoded_string = tokenizer(output_string,
                                   return_attention_mask=True,
                                   max_seq_len=max_len)
        ids, attention_mask_ids = encoded_string[
            "input_ids"][:max_len], encoded_string["attention_mask"][:max_len]
        output_string = "$1$".join([
            " ".join([str(i) for i in ids]),
            " ".join([str(i) for i in attention_mask_ids])
        ])
        fout.write('{}\n'.format(output_string))


def tokenize_with_bert_uncase(fin, fout, max_len=512):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tokenizer = ProphetNetTokenizer(vocab_file="prophetnet.tokenizer")
    for line in tqdm.tqdm(fin.readlines()):
        encoded_string = tokenizer(line,
                                   return_attention_mask=True,
                                   max_seq_len=max_len)
        ids, attention_mask_ids = encoded_string[
            "input_ids"][:max_len], encoded_string["attention_mask"][:max_len]
        output_string = "$1$".join([
            " ".join([str(i) for i in ids]),
            " ".join([str(i) for i in attention_mask_ids])
        ])
        fout.write('{}\n'.format(output_string))


def tokenize_data(dataset):
    dataset = dataset + "_data"
    input_dir = './data/%s' % (dataset)
    output_dir = './data/%s/uncased_tok_data' % (dataset)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if dataset == 'cnndm':
        uncased_preocess('%s/train.src' % input_dir,
                         '%s/train.src' % output_dir,
                         keep_sep=False)
        uncased_preocess('%s/dev.src' % input_dir,
                         '%s/dev.src' % output_dir,
                         keep_sep=False)
        uncased_preocess('%s/test.src' % input_dir,
                         '%s/test.src' % output_dir,
                         keep_sep=False)
        uncased_preocess('%s/train.tgt' % input_dir,
                         '%s/train.tgt' % output_dir,
                         keep_sep=True,
                         max_len=128)
        uncased_preocess('%s/dev.tgt' % input_dir,
                         '%s/dev.tgt' % output_dir,
                         keep_sep=True)
        uncased_preocess('%s/test.tgt' % input_dir,
                         '%s/test.tgt' % output_dir,
                         keep_sep=True)
    else:
        tokenize_with_bert_uncase('%s/train.src' % input_dir,
                                  '%s/train.src' % output_dir)
        tokenize_with_bert_uncase('%s/train.tgt' % input_dir,
                                  '%s/train.tgt' % output_dir)
        tokenize_with_bert_uncase('%s/dev.src' % input_dir,
                                  '%s/dev.src' % output_dir)
        tokenize_with_bert_uncase('%s/dev.tgt' % input_dir,
                                  '%s/dev.tgt' % output_dir)
        tokenize_with_bert_uncase('%s/test.src' % input_dir,
                                  '%s/test.src' % output_dir)
        tokenize_with_bert_uncase('%s/test.tgt' % input_dir,
                                  '%s/test.tgt' % output_dir)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="choose dataset from all, or 1 of 8 datasets: cnndm, gigaword")
args = parser.parse_args()

DATASET_LIST = ['cnndm', 'gigaword']

if args.dataset != 'all' and args.dataset not in DATASET_LIST:
    print('please choose dataset from all, or 1 of 8 datasets: cnndm, gigaword')
    exit()
else:
    if args.dataset == 'all':
        dataset_list = DATASET_LIST
    else:
        dataset_list = [args.dataset]

print(dataset_list)
for dataset in dataset_list:
    tokenize_data(dataset)
