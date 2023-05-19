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

import numpy as np
from data import label2ids
from metric import MetricReport
from tqdm import tqdm

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--label_path", type=str,
                    default='data/label.txt', help="The full path of label file")
parser.add_argument("--recall_result_file", type=str,
                    default='./recall_result_dir/recall_result.txt', help="The full path of recall result file")
parser.add_argument("--similar_text_pair", default='data/dev.txt',
                    help="The full path of similar pair file")

parser.add_argument("--threshold", default=0.5, type=float,
                    help="The threshold for selection the labels")

args = parser.parse_args()
# yapf: enable


def evaluate(label2id):
    metric = MetricReport()
    text2similar = {}
    # Encoding labels as one hot
    with open(args.similar_text_pair, "r", encoding="utf-8") as f:
        for line in f:
            text, similar_text = line.rstrip().rsplit("\t", 1)
            text2similar[text] = np.zeros(len(label2id))
            # One hot Encoding
            for label in similar_text.strip().split(","):
                text2similar[text][label2id[label]] = 1
    pred_labels = {}
    # Convert predicted labels into one hot encoding
    with open(args.recall_result_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            text_arr = line.rstrip().split("\t")
            text, labels, cosine_sim = text_arr
            # One hot Encoding
            if text not in pred_labels:
                pred_labels[text] = np.zeros(len(label2id))
            if float(cosine_sim) > args.threshold:
                for label in labels.split(","):
                    pred_labels[text][label2id[label]] = float(cosine_sim)

        for text, probs in tqdm(pred_labels.items()):
            metric.update(probs, text2similar[text])

        micro_f1_score, macro_f1_score = metric.accumulate()
        print("Micro fl score: {}".format(micro_f1_score * 100))
        print("Macro f1 score: {}".format(macro_f1_score * 100))


if __name__ == "__main__":
    label2id = label2ids(args.label_path)
    evaluate(label2id)
