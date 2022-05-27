from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from config import top_k

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--recall_path", type=str, required=True,
    help="recall data set.")

parser.add_argument("--tag_path", type=str, required=True,
    help="tag data set.")

parser.add_argument("--threshold", type=float, default=0.3,
    help="threshold for generate tags.")

args = parser.parse_args()
# yapf: enable


def gen_id2corpus(corpus_file):
    id2corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            id2corpus[idx] = line.rstrip()
    return id2corpus


def extract_tag(list_data, threshold):
    first_tag = []
    second_tag = []
    ans = []
    first = False
    for item in list_data:
        arr = item.strip().split('\t')
        tag1 = arr[-3]
        tag2 = arr[-2]
        product = float(arr[-1])
        # tag1,tag2,distance= arr
        if (product >= threshold or first == False):
            first = True
            first_tag.append(tag1)
            second_tag.append(tag2)
    counter = Counter(first_tag)
    for item in counter.most_common(1):
        ans.append(item[0])
    counter = Counter(second_tag)
    for item in counter.most_common(1):
        ans.append(item[0])
    return ans


def generate_tags(recall_path, tag_path, threshold=0.3):
    labels = []
    with open(recall_path) as f:
        list_data = []
        for i, item in enumerate(f.readlines()):
            list_data.append(item)
            if ((i + 1) % top_k == 0):
                label = extract_tag(list_data, threshold)
                list_data = []
                labels.append(label)

    with open(tag_path, 'w') as f:
        for item in labels:
            f.write('\t'.join(item) + '\n')


if __name__ == "__main__":
    generate_tags(args.recall_path, args.tag_path, args.threshold)
