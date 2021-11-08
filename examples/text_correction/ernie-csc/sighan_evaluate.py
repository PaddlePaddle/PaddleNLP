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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", "-p", required=True, type=str, help="")
parser.add_argument("--truth_file", "-t", required=True, type=str, help="")
args = parser.parse_args()


def main(args):
    detect_tp, correct_tp, pos, neg, fp = 0, 0, 0, 0, 0

    pred_dict = dict()
    truth_dict = dict()
    fpred = open(args.pred_file, 'r', encoding='utf-8')
    ftruth = open(args.truth_file, 'r', encoding='utf-8')
    for idx, (pred, truth) in enumerate(zip(fpred, ftruth)):
        pred_tokens = pred.strip().split(" ")
        truth_tokens = truth.strip().split(" ")

        pred_id = pred_tokens[0]
        truth_id = truth_tokens[0]

        pred_tokens = pred_tokens[1:]
        truth_tokens = truth_tokens[1:]

        detect_truth_positions = [
            int(truth_token.strip(","))
            for i, truth_token in enumerate(truth_tokens) if i % 2 == 0
        ]
        correct_truth_tokens = [
            truth_token.strip(",") for i, truth_token in enumerate(truth_tokens)
            if i % 2 == 1
        ]
        detect_pred_positions = [
            int(pred_token.strip(","))
            for i, pred_token in enumerate(pred_tokens) if i % 2 == 0
        ]
        correct_pred_tokens = [
            pred_token.strip(",") for i, pred_token in enumerate(pred_tokens)
            if i % 2 == 1
        ]

        pred_dict[pred_id] = (detect_pred_positions, correct_pred_tokens)
        truth_dict[truth_id] = (detect_truth_positions, correct_truth_tokens)

    assert sorted(pred_dict.keys()) == sorted(truth_dict.keys(
    )), "Prediction file should have all prediction result in truth file"

    for pid, predition in pred_dict.items():
        truth = truth_dict[pid]
        if predition[0][0] != 0:
            pos += 1
            if sorted(zip(*predition)) == sorted(zip(*truth)):
                correct_tp += 1
            if truth[0][0] == 0:
                fp += 1

        if truth[0][0] != 0:
            if sorted(predition[0]) == sorted(truth[0]):
                detect_tp += 1
            neg += 1

    eps = 1e-9

    # Detection level
    detect_pos = detect_tp + fp
    if detect_pos > 0 and neg > 0:
        detect_precision = detect_tp * 1.0 / detect_pos
        detect_recall = detect_tp * 1.0 / neg
        detect_f1 = 2. * detect_precision * detect_recall / (
            detect_precision + detect_recall + eps)
    else:
        detect_precision = 0
        detect_recall = 0
        detect_f1 = 0

    # Correction level
    correct_pos = correct_tp + fp
    if correct_pos > 0 and neg > 0:
        correct_precision = correct_tp * 1.0 / correct_pos
        correct_recall = correct_tp * 1.0 / neg
        correct_f1 = 2. * correct_precision * correct_recall / (
            correct_precision + correct_recall + eps)
    else:
        correct_precision = 0
        correct_recall = 0
        correct_f1 = 0

    print("==========================================================")
    print("Overall Performance")
    print("==========================================================")
    print("\nDetection Level")
    print("\tPrecision = {:.4f} ({}/{})".format(detect_precision, detect_tp,
                                                detect_pos))
    print("\tRecall = {:.4f} ({}/{})".format(detect_recall, detect_tp, neg))
    print("\tF1-Score = {:.4f} ((2*{:.4f}*{:.4f})/({:.4f}+{:.4f}))".format(
        detect_f1, detect_precision, detect_recall, detect_precision,
        detect_recall))

    print("\nCorrection Level")
    print("\tPrecision = {:.4f} ({}/{})".format(correct_precision, correct_tp,
                                                correct_pos))
    print("\tRecall = {:.4f} ({}/{})".format(correct_recall, correct_tp, neg))
    print("\tF1-Score = {:.4f} ((2*{:.4f}*{:.4f})/({:.4f}+{:.4f}))".format(
        correct_f1, correct_precision, correct_recall, correct_precision,
        correct_recall))
    print("==========================================================\n")


if __name__ == "__main__":
    main(args)
