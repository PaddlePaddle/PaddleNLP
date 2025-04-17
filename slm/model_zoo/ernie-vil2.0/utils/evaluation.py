# -*- coding: utf-8 -*-

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

"""
This script computes the recall scores given the ground-truth annotations and predictions.
"""

import json
import os
import sys

NUM_K = 10


def read_submission(submit_path, reference, k=5):
    # Check whether the path of submitted file exists
    if not os.path.exists(submit_path):
        raise Exception("The file is not found!")

    submission_dict = {}
    ref_qids = set(reference.keys())

    with open(submit_path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            try:
                pred_obj = json.loads(line)
            except Exception:
                raise Exception("Cannot parse this line into json object: {}".format(line))
            if "text_id" not in pred_obj:
                raise Exception("There exists one line not containing text_id: {}".format(line))
            if not isinstance(pred_obj["text_id"], int):
                raise Exception(
                    "Found an invalid text_id , it should be an integer (not string), please check your schema"
                )
            qid = pred_obj["text_id"]
            if "image_ids" not in pred_obj:
                raise Exception("There exists one line not containing the predicted image_ids: {}".format(line))
            image_ids = pred_obj["image_ids"]
            if not isinstance(image_ids, list):
                raise Exception(
                    "The image_ids field of text_id {} is not a list, please check your schema".format(qid)
                )
            # Check whether there are K products for each text
            if len(image_ids) != k:
                raise Exception(
                    "Text_id {} has wrong number of predicted image_ids! Require {}, but {} founded.".format(
                        qid, k, len(image_ids)
                    )
                )
            # Check whether there are duplicate predicted products for a single text
            if len(set(image_ids)) != k:
                raise Exception(
                    "Text_id {} has duplicate products in your prediction. Pleace check again!".format(qid)
                )
            submission_dict[qid] = image_ids  # here we save the list of product ids

    # Check if any text is missing in the submission
    pred_qids = set(submission_dict.keys())
    nopred_qids = ref_qids - pred_qids
    if len(nopred_qids) != 0:
        raise Exception(
            "The following text_ids have no prediction in your submission, please check again: {}".format(
                ", ".join([str(idx) for idx in nopred_qids])
            )
        )

    return submission_dict


def dump_2_json(info, path):
    with open(path, "w", encoding="utf-8") as output_json_file:
        json.dump(info, output_json_file)


def report_error_msg(detail, showMsg, out_p):
    error_dict = dict()
    error_dict["errorDetail"] = detail
    error_dict["errorMsg"] = showMsg
    error_dict["score"] = 0
    error_dict["scoreJson"] = {}
    error_dict["success"] = False
    dump_2_json(error_dict, out_p)


def report_score(r1, r5, r10, out_p):
    result = dict()
    result["success"] = True
    mean_recall = (r1 + r5 + r10) / 3.0
    result["score"] = mean_recall * 100
    result["scoreJson"] = {
        "score": mean_recall * 100,
        "mean_recall": mean_recall * 100,
        "r1": r1 * 100,
        "r5": r5 * 100,
        "r10": r10 * 100,
    }
    dump_2_json(result, out_p)


def read_reference(path):
    fin = open(path, encoding="utf-8")
    reference = dict()
    for line in fin:
        line = line.strip()
        obj = json.loads(line)
        reference[obj["text_id"]] = obj["image_ids"]
    return reference


def compute_score(golden_file, predict_file):
    # Read ground-truth
    reference = read_reference(golden_file)

    # Read predictions
    k = 10
    predictions = read_submission(predict_file, reference, k)

    # Compute score for each text
    r1_stat, r5_stat, r10_stat = 0, 0, 0
    for qid in reference.keys():
        ground_truth_ids = set(reference[qid])
        top10_pred_ids = predictions[qid]
        if any([idx in top10_pred_ids[:1] for idx in ground_truth_ids]):
            r1_stat += 1
        if any([idx in top10_pred_ids[:5] for idx in ground_truth_ids]):
            r5_stat += 1
        if any([idx in top10_pred_ids[:10] for idx in ground_truth_ids]):
            r10_stat += 1
    # The higher score, the better
    r1, r5, r10 = r1_stat * 1.0 / len(reference), r5_stat * 1.0 / len(reference), r10_stat * 1.0 / len(reference)
    mean_recall = (r1 + r5 + r10) / 3.0
    result = [mean_recall, r1, r5, r10]
    result = [score * 100 for score in result]
    return result


if __name__ == "__main__":
    # The path of answer json file (eg. test_queries_answers.jsonl)
    standard_path = sys.argv[1]
    # The path of prediction file (eg. example_pred.jsonl)
    submit_path = sys.argv[2]
    # The score will be dumped into this output json file
    out_path = sys.argv[3]

    print("Read standard from %s" % standard_path)
    print("Read user submit file from %s" % submit_path)

    try:
        # Read ground-truth
        reference = read_reference(standard_path)

        # Read predictions
        k = 10
        predictions = read_submission(submit_path, reference, k)

        # Compute score for each text
        r1_stat, r5_stat, r10_stat = 0, 0, 0
        for qid in reference.keys():
            ground_truth_ids = set(reference[qid])
            top10_pred_ids = predictions[qid]
            if any([idx in top10_pred_ids[:1] for idx in ground_truth_ids]):
                r1_stat += 1
            if any([idx in top10_pred_ids[:5] for idx in ground_truth_ids]):
                r5_stat += 1
            if any([idx in top10_pred_ids[:10] for idx in ground_truth_ids]):
                r10_stat += 1
        # The higher score, the better
        r1, r5, r10 = r1_stat * 1.0 / len(reference), r5_stat * 1.0 / len(reference), r10_stat * 1.0 / len(reference)
        report_score(r1, r5, r10, out_path)
        print("The evaluation finished successfully.")
    except Exception as e:
        report_error_msg(e.args[0], e.args[0], out_path)
        print("The evaluation failed: {}".format(e.args[0]))
