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

import numpy as np
from scipy import stats
from sklearn import metrics


def eval_metrics(labels, sims):

    eval_res = {}
    spearman_corr = stats.spearmanr(labels, sims).correlation
    eval_res["spearman_corr"] = spearman_corr
    print("Spearman corr --> {:.5f}\n".format(spearman_corr))

    labels_2_cls = np.where(np.greater_equal(labels, 3), np.ones_like(labels),
                            np.zeros_like(labels))

    pre, rec, thr = metrics.precision_recall_curve(labels_2_cls,
                                                   sims,
                                                   pos_label=1)
    pre_rec = np.sum([pre, rec], axis=0)
    best_pre_rec = np.max(pre_rec)
    best_pre = pre[np.argmax(pre_rec)]
    best_rec = rec[np.argmax(pre_rec)]
    best_thr = thr[np.argmax(pre_rec)]
    eval_res["best_pre_rec_thr"] = [best_pre_rec, best_pre, best_rec, best_thr]
    print(
        "2-CLS best pre+rec={:.5f}, pre --> {:.5f}, rec --> {:.5f}, thr --> {:.5f}"
        .format(best_pre_rec, best_pre, best_rec, best_thr))

    pre_rec_thr = list(zip(pre, rec, thr))
    pre_rec_thr.sort(key=lambda x: x[1], reverse=True)
    pre_rec_thr_at_K = None
    at_K = 0.95
    for x in pre_rec_thr:
        if x[0] >= at_K:
            pre_rec_thr_at_K = x
            break
    eval_res["best_pre_rec_thr_at_K"] = pre_rec_thr_at_K
    print(
        "2-CLS best pre@{:.2f}, pre --> {:.5f}, rec --> {:.5f}, thr --> {:.5f}".
        format(at_K, pre_rec_thr_at_K[0], pre_rec_thr_at_K[1],
               pre_rec_thr_at_K[2]))

    preds_2_cls = np.where(np.greater_equal(sims, best_thr), np.ones_like(sims),
                           np.zeros_like(sims))

    acc_2_cls = metrics.accuracy_score(labels_2_cls, preds_2_cls)
    auc_2_cls = metrics.roc_auc_score(labels_2_cls, sims)
    report_2_cls = metrics.classification_report(labels_2_cls,
                                                 preds_2_cls,
                                                 labels=[0, 1],
                                                 target_names=["0", "1"],
                                                 digits=5,
                                                 output_dict=True)
    eval_res["acc_2_cls"] = acc_2_cls
    eval_res["auc_2_cls"] = auc_2_cls
    eval_res["report_2_cls"] = report_2_cls
    print("2-CLS metrics are evaluated on the best thr --> {}".format(best_thr))
    print("2-CLS accuracy --> {}".format(acc_2_cls))
    print("2-CLS roc-auc --> {}".format(auc_2_cls))
    print("\t".join(["label", "precision", "recall", "f1-score", "support"]))
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
        "2-CLS 0", report_2_cls["0"]["precision"], report_2_cls["0"]["recall"],
        report_2_cls["0"]["f1-score"], report_2_cls["0"]["support"]))
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}\n".format(
        "2-CLS 1", report_2_cls["1"]["precision"], report_2_cls["1"]["recall"],
        report_2_cls["1"]["f1-score"], report_2_cls["1"]["support"]))

    best_thr_6_cls = []
    for pivot in [5, 4, 3, 2, 1]:
        labels_pivot = np.where(np.greater_equal(labels, pivot),
                                np.ones_like(labels), np.zeros_like(labels))
        pre_pivot, rec_pivot, thr_pivot = metrics.precision_recall_curve(
            labels_pivot, sims, pos_label=1)
        pre_rec_pivot = np.sum([pre_pivot, rec_pivot], axis=0)
        best_pre_rec_pivot = np.max(pre_rec_pivot)
        best_pre_pivot = pre_pivot[np.argmax(pre_rec_pivot)]
        best_rec_pivot = rec_pivot[np.argmax(pre_rec_pivot)]
        best_thr_pivot = thr_pivot[np.argmax(pre_rec_pivot)]
        best_thr_6_cls.append(best_thr_pivot)
        eval_res["best_pre_rec_thr_6_cls_at_{}_pivot".format(pivot)] = [
            best_pre_rec_pivot, best_pre_pivot, best_rec_pivot, best_thr_pivot
        ]
        print(
            "6-CLS [split at {} pivot] best pre+rec={:.5f}, pre --> {:.5f}, rec --> {:.5f}, thr --> {:.5f}"
            .format(pivot, best_pre_rec_pivot, best_pre_pivot, best_rec_pivot,
                    best_thr_pivot))

    preds = []
    for s in sims:
        s = max(s, 0)
        for pivot, thr in list(zip([5, 4, 3, 2, 1, 0], best_thr_6_cls + [0.0])):
            if s + 1e-8 >= thr:
                preds.append(pivot)
                break

    acc_6_cls = metrics.accuracy_score(labels, preds)

    report_6_cls = metrics.classification_report(
        labels,
        preds,
        labels=[0, 1, 2, 3, 4, 5],
        target_names=["0", "1", "2", "3", "4", "5"],
        digits=5,
        output_dict=True)
    eval_res["acc_6_cls"] = acc_6_cls
    eval_res["report_6_cls"] = report_6_cls
    print("6-CLS metrics are evaluated on the best thr --> {}".format(
        best_thr_6_cls))
    print("6-CLS accuracy --> {}".format(acc_6_cls))
    print("\t".join(["label", "precision", "recall", "f1-score", "support"]))
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
        "6-CLS 0", report_6_cls["0"]["precision"], report_6_cls["0"]["recall"],
        report_6_cls["0"]["f1-score"], report_6_cls["0"]["support"]))
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
        "6-CLS 1", report_6_cls["1"]["precision"], report_6_cls["1"]["recall"],
        report_6_cls["1"]["f1-score"], report_6_cls["1"]["support"]))
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
        "6-CLS 2", report_6_cls["2"]["precision"], report_6_cls["2"]["recall"],
        report_6_cls["2"]["f1-score"], report_6_cls["2"]["support"]))
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
        "6-CLS 3", report_6_cls["3"]["precision"], report_6_cls["3"]["recall"],
        report_6_cls["3"]["f1-score"], report_6_cls["3"]["support"]))
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
        "6-CLS 4", report_6_cls["4"]["precision"], report_6_cls["4"]["recall"],
        report_6_cls["4"]["f1-score"], report_6_cls["4"]["support"]))
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
        "6-CLS 5", report_6_cls["5"]["precision"], report_6_cls["5"]["recall"],
        report_6_cls["5"]["f1-score"], report_6_cls["5"]["support"]))

    return eval_res


def read_infer_result(infer_result):
    sims = []
    with open(infer_result, 'r') as f:
        for line in f:
            line = line.strip()
            if " --> " in line:
                sim = float(line.split(' --> ')[1])
                sims.append(sim)
    return sims


def read_infer_label(infer_label):
    labels = []
    with open(infer_label, 'r') as f:
        for line in f:
            line = line.strip()
            label = int(line.split('\t')[2])
            labels.append(label)
    return labels


if __name__ == "__main__":
    """
    example
    """
    labels = read_infer_label("./data/test_v1.txt")
    sims = read_infer_result("infer_res.txt")
    eval_metrics(labels, sims)
