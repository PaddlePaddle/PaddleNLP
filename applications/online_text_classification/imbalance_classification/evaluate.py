import os
import argparse
from collections import defaultdict

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--predict_file", type=str, required=True,
    help="The directory to static model.")

parser.add_argument("--target_file", type=str, required=True,
    help="The corpus_file path.")

args = parser.parse_args()
# yapf: enable


def read_text(file_path):
    file = open(file_path)
    list_data = []
    for idx, data in enumerate(file.readlines()):
        item = data.strip().split('\t')[-2:]
        list_data.append(item)
    return list_data


def read_pred_text(file_path):
    file = open(file_path)
    list_data = []
    for idx, data in enumerate(file.readlines()):
        item = data.strip().split('\t')
        list_data.append(item)
    return list_data


def evaluate(predict_path, target_path):
    target_labels = read_text(target_path)
    print(target_labels[:5])
    pred_labels = read_pred_text(predict_path)
    print(pred_labels[:5])

    counts = defaultdict(int)
    for pred, tgt in zip(pred_labels, target_labels):
        for idx, (i, j) in enumerate(zip(pred, tgt)):
            if (i == j):
                counts[idx] += 1
    for k, v in counts.items():
        print('{}级分类'.format(k + 1))
        total_count = len(pred_labels)
        print(v / total_count)


if __name__ == "__main__":
    evaluate(args.predict_file, args.target_file)
