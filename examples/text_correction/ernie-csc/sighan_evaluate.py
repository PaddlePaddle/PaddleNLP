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


def main():
    detect_sent_TP, sent_P, sent_N, correct_sent_TP, sent_FP = 0, 0, 0, 0, 0
    for idx, (pred, actual
              ) in enumerate(zip(open(args.pred_file), open(args.truth_file))):
        pred_tokens = pred.strip().split(" ")
        actual_tokens = actual.strip().split(" ")

        pred_tokens = pred_tokens[1:]
        actual_tokens = actual_tokens[1:]
        detect_actual_tokens = [
            int(actual_token.strip(","))
            for i, actual_token in enumerate(actual_tokens) if i % 2 == 0
        ]
        correct_actual_tokens = [
            actual_token.strip(",")
            for i, actual_token in enumerate(actual_tokens) if i % 2 == 1
        ]
        detect_pred_tokens = [
            int(pred_token.strip(","))
            for i, pred_token in enumerate(pred_tokens) if i % 2 == 0
        ]
        correct_pred_tokens = [
            pred_token.strip(",") for i, pred_token in enumerate(pred_tokens)
            if i % 2 == 1
        ]

        correct_pred_zip = zip(detect_pred_tokens, correct_pred_tokens)
        correct_actual_zip = zip(detect_actual_tokens, correct_actual_tokens)

        if detect_pred_tokens[0] != 0:
            sent_P += 1
            if sorted(correct_pred_zip) == sorted(correct_actual_zip):
                correct_sent_TP += 1
            if detect_actual_tokens[0] == 0:
                sent_FP += 1

        if detect_actual_tokens[0] != 0:
            if sorted(detect_actual_tokens) == sorted(detect_pred_tokens):
                detect_sent_TP += 1
            sent_N += 1

    detect_sent_TP_FP = detect_sent_TP + sent_FP
    detect_sent_precision = detect_sent_TP * 1.0 / (detect_sent_TP_FP)
    detect_sent_recall = detect_sent_TP * 1.0 / (sent_N)
    detect_sent_F1 = 2. * detect_sent_precision * detect_sent_recall / (
        (detect_sent_precision + detect_sent_recall) + 1e-8)

    correct_sent_TP_FP = correct_sent_TP + sent_FP
    correct_sent_precision = correct_sent_TP * 1.0 / (correct_sent_TP_FP)
    correct_sent_recall = correct_sent_TP * 1.0 / (sent_N)
    correct_sent_F1 = 2. * correct_sent_precision * correct_sent_recall / (
        (correct_sent_precision + correct_sent_recall) + 1e-8)

    detect_sent_FP = sent_P - detect_sent_TP

    print("==========================================================")
    print("Overall Performance")
    print("==========================================================")
    print("\nDetection Level")
    print("\tPrecision = {:.4f} ({}/{})".format(
        detect_sent_precision, detect_sent_TP, detect_sent_TP_FP))
    print("\tRecall = {:.4f} ({}/{})".format(detect_sent_recall, detect_sent_TP,
                                             sent_N))
    print("\tF1-Score = {:.4f} ((2*{:.4f}*{:.4f})/({:.4f}+{:.4f}))".format(
        detect_sent_F1, detect_sent_precision, detect_sent_recall,
        detect_sent_precision, detect_sent_recall))

    print("\nCorrection Level")
    print("\tPrecision = {:.4f} ({}/{})".format(
        correct_sent_precision, correct_sent_TP, correct_sent_TP_FP))
    print("\tRecall = {:.4f} ({}/{})".format(correct_sent_recall,
                                             correct_sent_TP, sent_N))
    print("\tF1-Score = {:.4f} ((2*{:.4f}*{:.4f})/({:.4f}+{:.4f}))".format(
        correct_sent_F1, correct_sent_precision, correct_sent_recall,
        correct_sent_precision, correct_sent_recall))
    print("==========================================================\n")


if __name__ == "__main__":
    main()
