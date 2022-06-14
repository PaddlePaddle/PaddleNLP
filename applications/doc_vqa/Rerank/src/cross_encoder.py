#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import time
import logging
import numpy as np
from six.moves import xrange
from scipy.stats import pearsonr, spearmanr

import paddle.fluid as fluid

from model.ernie import ErnieModel

log = logging.getLogger(__name__)


def create_model(args,
                 pyreader_name,
                 ernie_config,
                 is_prediction=False,
                 task_name=""):
    pyreader = fluid.layers.py_reader(capacity=50,
                                      shapes=[[-1, args.max_seq_len, 1],
                                              [-1, args.max_seq_len, 1],
                                              [-1, args.max_seq_len, 1],
                                              [-1, args.max_seq_len, 1],
                                              [-1, args.max_seq_len, 1],
                                              [-1, 1], [-1, 1]],
                                      dtypes=[
                                          'int64', 'int64', 'int64', 'int64',
                                          'float32', 'int64', 'int64'
                                      ],
                                      lod_levels=[0, 0, 0, 0, 0, 0, 0],
                                      name=task_name + "_" + pyreader_name,
                                      use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, task_ids, input_mask, labels,
     qids) = fluid.layers.read_file(pyreader)

    def _model(is_noise=False):
        ernie = ErnieModel(src_ids=src_ids,
                           position_ids=pos_ids,
                           sentence_ids=sent_ids,
                           task_ids=task_ids,
                           input_mask=input_mask,
                           config=ernie_config,
                           is_noise=is_noise)

        cls_feats = ernie.get_pooled_output()
        if not is_noise:
            cls_feats = fluid.layers.dropout(
                x=cls_feats,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=cls_feats,
            size=args.num_labels,
            param_attr=fluid.ParamAttr(
                name=task_name + "_cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name=task_name + "_cls_out_b",
                initializer=fluid.initializer.Constant(0.)))
        """
        if is_prediction:
            probs = fluid.layers.softmax(logits)
            feed_targets_name = [
                src_ids.name, sent_ids.name, pos_ids.name, input_mask.name
            ]
            if ernie_version == "2.0":
                feed_targets_name += [task_ids.name]
            return pyreader, probs, feed_targets_name
        """

        num_seqs = fluid.layers.create_tensor(dtype='int64')
        ## add focal loss
        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=labels, return_softmax=True)
        loss = fluid.layers.mean(x=ce_loss)
        accuracy = fluid.layers.accuracy(input=probs,
                                         label=labels,
                                         total=num_seqs)
        graph_vars = {
            "loss": loss,
            "probs": probs,
            "accuracy": accuracy,
            "labels": labels,
            "num_seqs": num_seqs,
            "qids": qids
        }
        return graph_vars

    if not is_prediction:
        graph_vars = _model(is_noise=True)
        old_loss = graph_vars["loss"]
        token_emb = fluid.default_main_program().global_block().var(
            "word_embedding")
        # print(token_emb)
        token_emb.stop_gradient = False
        token_gradient = fluid.gradients(old_loss, token_emb)[0]
        token_gradient.stop_gradient = False
        epsilon = 1e-8
        norm = (fluid.layers.sqrt(
            fluid.layers.reduce_sum(fluid.layers.square(token_gradient)) +
            epsilon))
        gp = (0.01 * token_gradient) / norm
        gp.stop_gradient = True
        fluid.layers.assign(token_emb + gp, token_emb)
        graph_vars = _model()
        fluid.layers.assign(token_emb - gp, token_emb)
    else:
        graph_vars = _model()

    return pyreader, graph_vars


def evaluate_mrr(preds):
    last_qid = None
    total_mrr = 0.0
    qnum = 0.0
    rank = 0.0
    correct = False
    for qid, score, label in preds:
        if qid != last_qid:
            rank = 0.0
            qnum += 1
            correct = False
            last_qid = qid

        rank += 1
        if not correct and label != 0:
            total_mrr += 1.0 / rank
            correct = True

    return total_mrr / qnum


def evaluate(exe,
             test_program,
             test_pyreader,
             graph_vars,
             eval_phase,
             use_multi_gpu_test=False,
             metric='simple_accuracy'):
    train_fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["num_seqs"].name
    ]

    if eval_phase == "train":
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list, program=test_program)
        ret = {"loss": np.mean(outputs[0]), "accuracy": np.mean(outputs[1])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[3][0])
        return ret

    test_pyreader.start()
    total_cost = 0.0
    total_acc = 0.0
    total_num_seqs = 0.0
    total_label_pos_num = 0.0
    total_pred_pos_num = 0.0
    total_correct_num = 0.0
    qids, labels, scores, preds = [], [], [], []
    time_begin = time.time()

    fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["probs"].name, graph_vars["labels"].name,
        graph_vars["num_seqs"].name, graph_vars["qids"].name
    ]
    while True:
        try:
            if use_multi_gpu_test:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                    fetch_list=fetch_list)
            else:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                    program=test_program, fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_acc += np.sum(np_acc * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            scores.extend(np_probs[:, 1].reshape(-1).tolist())
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            preds.extend(np_preds)
            total_label_pos_num += np.sum(np_labels)
            total_pred_pos_num += np.sum(np_preds)
            total_correct_num += np.sum(np.dot(np_preds, np_labels))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    cost = total_cost / total_num_seqs
    elapsed_time = time_end - time_begin

    evaluate_info = ""
    if metric == 'acc_and_f1':
        ret = acc_and_f1(preds, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, ave_acc: %f, f1: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret['acc'], ret['f1'], total_num_seqs, elapsed_time)
    elif metric == 'matthews_corrcoef':
        ret = matthews_corrcoef(preds, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, matthews_corrcoef: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret, total_num_seqs, elapsed_time)
    elif metric == 'pearson_and_spearman':
        ret = pearson_and_spearman(scores, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, pearson:%f, spearman:%f, corr:%f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret['pearson'], ret['spearman'], ret['corr'], total_num_seqs, elapsed_time)
    elif metric == 'simple_accuracy':
        ret = simple_accuracy(preds, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, acc:%f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret, total_num_seqs, elapsed_time)
    elif metric == "acc_and_f1_and_mrr":
        ret_a = acc_and_f1(preds, labels)
        preds = sorted(zip(qids, scores, labels),
                       key=lambda elem: (elem[0], -elem[1]))
        ret_b = evaluate_mrr(preds)
        evaluate_info = "[%s evaluation] ave loss: %f, acc: %f, f1: %f, mrr: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret_a['acc'], ret_a['f1'], ret_b, total_num_seqs, elapsed_time)
    else:
        raise ValueError('unsupported metric {}'.format(metric))
    return evaluate_info


def matthews_corrcoef(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))

    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc


def f1_score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return f1


def pearson_and_spearman(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def acc_and_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    acc = simple_accuracy(preds, labels)
    f1 = f1_score(preds, labels)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


def predict(exe, test_program, test_pyreader, graph_vars, dev_count=1):
    test_pyreader.start()
    qids, scores, probs = [], [], []
    preds = []

    fetch_list = [graph_vars["probs"].name, graph_vars["qids"].name]

    while True:
        try:
            if dev_count == 1:
                np_probs, np_qids = exe.run(program=test_program,
                                            fetch_list=fetch_list)
            else:
                np_probs, np_qids = exe.run(fetch_list=fetch_list)

            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            preds.extend(np_preds)
            probs.append(np_probs)

        except fluid.core.EOFException:
            test_pyreader.reset()
            break

    probs = np.concatenate(probs, axis=0).reshape([len(preds), -1])

    return qids, preds, probs
