import os
import argparse
import json
import multiprocessing
import pickle
import mmap
import numpy as np
from paddlenlp.transformers import BigBirdTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', type=str, required=True, help='Path to input JSON')
    parser.add_argument(
        '--model_name', type=str, required=True, help='What model to use.')
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker processes to launch')
    args = parser.parse_args()
    return args


class Converter(object):
    def __init__(self, model_name):
        tokenizer = BigBirdTokenizer.from_pretrained(model_name)
        Converter.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.vocab)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        text = data["text"]
        doc_ids = []
        tokens = self.tokenizer(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids, len(ids)


def main():
    args = get_args()
    fin = open(args.input_path, 'r', encoding='utf-8')
    convert = Converter(args.model_name)
    pool = multiprocessing.Pool(args.workers)
    encoded_docs = pool.imap(convert.encode, fin, 25)
    all_doc_ids = []
    lens = []
    for tokens, sizes in encoded_docs:
        all_doc_ids.extend(tokens)
        lens.append(sizes)
    # save the mmap for the tokens and lens 
    save_dtype = None
    if convert.vocab_size < 65500:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32
    all_doc_ids = np.array(all_doc_ids, dtype=save_dtype)
    lens = np.array(lens, dtype=save_dtype)
    np.savez(args.input_path + "_ids.npz", ids=all_doc_ids, lens=lens)


if __name__ == "__main__":
    main()
