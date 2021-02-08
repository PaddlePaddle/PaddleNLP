# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"""evaluate task metrics"""

import sys
import io


class EvalDA(object):
    """
    evaluate da testset, swda|mrda
    """

    def __init__(self, task_name, pred, refer):
        """
        predict file
        """
        self.pred_file = pred
        self.refer_file = refer

    def load_data(self):
        """
        load reference label and predict label
        """
        pred_label = []
        refer_label = []
        fr = io.open(self.refer_file, 'r', encoding="utf8")
        for line in fr:
            label = line.rstrip('\n').split('\t')[1]
            refer_label.append(int(label))
        idx = 0
        fr = io.open(self.pred_file, 'r', encoding="utf8")
        for line in fr:
            elems = line.rstrip('\n').split('\t')
            if len(elems) != 2 or not elems[0].isdigit():
                continue
            tag_id = int(elems[1])
            pred_label.append(tag_id)
        return pred_label, refer_label

    def evaluate(self):
        """
        calculate acc metrics
        """
        pred_label, refer_label = self.load_data()
        common_num = 0
        total_num = len(pred_label)
        for i in range(total_num):
            if pred_label[i] == refer_label[i]:
                common_num += 1
        acc = float(common_num) / total_num
        return acc


class EvalATISIntent(object):
    """
    evaluate da testset, swda|mrda
    """

    def __init__(self, pred, refer):
        """
        predict file
        """
        self.pred_file = pred
        self.refer_file = refer

    def load_data(self):
        """
        load reference label and predict label
        """
        pred_label = []
        refer_label = []
        fr = io.open(self.refer_file, 'r', encoding="utf8")
        for line in fr:
            label = line.rstrip('\n').split('\t')[0]
            refer_label.append(int(label))
        idx = 0
        fr = io.open(self.pred_file, 'r', encoding="utf8")
        for line in fr:
            elems = line.rstrip('\n').split('\t')
            if len(elems) != 2 or not elems[0].isdigit():
                continue
            tag_id = int(elems[1])
            pred_label.append(tag_id)
        return pred_label, refer_label

    def evaluate(self):
        """
        calculate acc metrics
        """
        pred_label, refer_label = self.load_data()
        common_num = 0
        total_num = len(pred_label)
        for i in range(total_num):
            if pred_label[i] == refer_label[i]:
                common_num += 1
        acc = float(common_num) / total_num
        return acc


class EvalATISSlot(object):
    """
    evaluate atis slot
    """

    def __init__(self, pred, refer):
        """
        pred file
        """
        self.pred_file = pred
        self.refer_file = refer

    def load_data(self):
        """
        load reference label and predict label
        """
        pred_label = []
        refer_label = []
        fr = io.open(self.refer_file, 'r', encoding="utf8")
        for line in fr:
            labels = line.rstrip('\n').split('\t')[1].split()
            labels = [int(l) for l in labels]
            refer_label.append(labels)
        fr = io.open(self.pred_file, 'r', encoding="utf8")
        for line in fr:
            if len(line.split('\t')) != 2 or not line[0].isdigit():
                continue
            labels = line.rstrip('\n').split('\t')[1].split()[1:]
            labels = [int(l) for l in labels]
            pred_label.append(labels)
        pred_label_equal = []
        refer_label_equal = []
        assert len(refer_label) == len(pred_label)
        for i in range(len(refer_label)):
            num = len(refer_label[i])
            refer_label_equal.extend(refer_label[i])
            pred_label[i] = pred_label[i][:num]
            pred_label_equal.extend(pred_label[i])

        return pred_label_equal, refer_label_equal

    def evaluate(self):
        """
        evaluate f1_micro score
        """
        pred_label, refer_label = self.load_data()
        tp = dict()
        fn = dict()
        fp = dict()
        for i in range(len(refer_label)):
            if refer_label[i] == pred_label[i]:
                if refer_label[i] not in tp:
                    tp[refer_label[i]] = 0
                tp[refer_label[i]] += 1
            else:
                if pred_label[i] not in fp:
                    fp[pred_label[i]] = 0
                fp[pred_label[i]] += 1
                if refer_label[i] not in fn:
                    fn[refer_label[i]] = 0
                fn[refer_label[i]] += 1

        results = ["label    precision    recall"]
        for i in range(0, 130):
            if i not in tp:
                results.append(" %s:    0.0     0.0" % i)
                continue
            if i in fp:
                precision = float(tp[i]) / (tp[i] + fp[i])
            else:
                precision = 1.0
            if i in fn:
                recall = float(tp[i]) / (tp[i] + fn[i])
            else:
                recall = 1.0
            results.append(" %s:    %.4f    %.4f" % (i, precision, recall))
        tp_total = sum(tp.values())
        fn_total = sum(fn.values())
        fp_total = sum(fp.values())
        p_total = float(tp_total) / (tp_total + fp_total)
        r_total = float(tp_total) / (tp_total + fn_total)
        f_micro = 2 * p_total * r_total / (p_total + r_total)
        results.append("f1_micro: %.4f" % (f_micro))
        return "\n".join(results)


class EvalUDC(object):
    """
    evaluate udc
    """

    def __init__(self, pred, refer):
        """
        predict file
        """
        self.pred_file = pred
        self.refer_file = refer

    def load_data(self):
        """
        load reference label and predict label
        """
        data = []
        refer_label = []
        fr = io.open(self.refer_file, 'r', encoding="utf8")
        for line in fr:
            label = line.rstrip('\n').split('\t')[0]
            refer_label.append(label)
        idx = 0
        fr = io.open(self.pred_file, 'r', encoding="utf8")
        for line in fr:
            elems = line.rstrip('\n').split('\t')
            if len(elems) != 2 or not elems[0].isdigit():
                continue
            match_prob = elems[1]
            data.append((float(match_prob), int(refer_label[idx])))
            idx += 1
        return data

    def get_p_at_n_in_m(self, data, n, m, ind):
        """
        calculate precision in recall n
        """
        pos_score = data[ind][0]
        curr = data[ind:ind + m]
        curr = sorted(curr, key=lambda x: x[0], reverse=True)

        if curr[n - 1][0] <= pos_score:
            return 1
        return 0

    def evaluate(self):
        """
        calculate udc data
        """
        data = self.load_data()
        assert len(data) % 10 == 0

        p_at_1_in_2 = 0.0
        p_at_1_in_10 = 0.0
        p_at_2_in_10 = 0.0
        p_at_5_in_10 = 0.0

        length = int(len(data) / 10)

        for i in range(0, length):
            ind = i * 10
            assert data[ind][1] == 1

            p_at_1_in_2 += self.get_p_at_n_in_m(data, 1, 2, ind)
            p_at_1_in_10 += self.get_p_at_n_in_m(data, 1, 10, ind)
            p_at_2_in_10 += self.get_p_at_n_in_m(data, 2, 10, ind)
            p_at_5_in_10 += self.get_p_at_n_in_m(data, 5, 10, ind)

        metrics_out = [p_at_1_in_2 / length, p_at_1_in_10 / length, \
                p_at_2_in_10 / length, p_at_5_in_10 / length]
        return metrics_out


class EvalDSTC2(object):
    """
    evaluate dst testset, dstc2
    """

    def __init__(self, task_name, pred, refer):
        """
        predict file
        """
        self.task_name = task_name
        self.pred_file = pred
        self.refer_file = refer

    def load_data(self):
        """
        load reference label and predict label
        """
        pred_label = []
        refer_label = []
        fr = io.open(self.refer_file, 'r', encoding="utf8")
        for line in fr:
            line = line.strip('\n')
            labels = [int(l) for l in line.split('\t')[-1].split()]
            labels = sorted(list(set(labels)))
            refer_label.append(" ".join([str(l) for l in labels]))
        all_pred = []
        fr = io.open(self.pred_file, 'r', encoding="utf8")
        for line in fr:
            line = line.strip('\n')
            all_pred.append(line)
        all_pred = all_pred[len(all_pred) - len(refer_label):]
        for line in all_pred:
            labels = [int(l) for l in line.split('\t')[-1].split()]
            labels = sorted(list(set(labels)))
            pred_label.append(" ".join([str(l) for l in labels]))
        return pred_label, refer_label

    def evaluate(self):
        """
        calculate joint acc && overall acc
        """
        overall_all = 0.0
        correct_joint = 0
        pred_label, refer_label = self.load_data()
        for i in range(len(refer_label)):
            if refer_label[i] != pred_label[i]:
                continue
            correct_joint += 1
        joint_all = float(correct_joint) / len(refer_label)
        metrics_out = [joint_all, overall_all]
        return metrics_out


def evaluate(task_name, pred_file, refer_file):
    """evaluate task metrics"""
    if task_name.lower() == 'udc':
        eval_inst = EvalUDC(pred_file, refer_file)
        eval_metrics = eval_inst.evaluate()
        print("MATCHING TASK: %s metrics in testset: " % task_name)
        print("R1@2: %s" % eval_metrics[0])
        print("R1@10: %s" % eval_metrics[1])
        print("R2@10: %s" % eval_metrics[2])
        print("R5@10: %s" % eval_metrics[3])

    elif task_name.lower() in ['swda', 'mrda']:
        eval_inst = EvalDA(task_name.lower(), pred_file, refer_file)
        eval_metrics = eval_inst.evaluate()
        print("DA TASK: %s metrics in testset: " % task_name)
        print("ACC: %s" % eval_metrics)

    elif task_name.lower() == 'atis_intent':
        eval_inst = EvalATISIntent(pred_file, refer_file)
        eval_metrics = eval_inst.evaluate()
        print("INTENTION TASK: %s metrics in testset: " % task_name)
        print("ACC: %s" % eval_metrics)

    elif task_name.lower() == 'atis_slot':
        eval_inst = EvalATISSlot(pred_file, refer_file)
        eval_metrics = eval_inst.evaluate()
        print("SLOT FILLING TASK: %s metrics in testset: " % task_name)
        print(eval_metrics)
    elif task_name.lower() in ['dstc2', 'dstc2_asr']:
        eval_inst = EvalDSTC2(task_name.lower(), pred_file, refer_file)
        eval_metrics = eval_inst.evaluate()
        print("DST TASK: %s metrics in testset: " % task_name)
        print("JOINT ACC: %s" % eval_metrics[0])
    elif task_name.lower() == "multi-woz":
        eval_inst = EvalMultiWoz(pred_file, refer_file)
        eval_metrics = eval_inst.evaluate()
        print("DST TASK: %s metrics in testset: " % task_name)
        print("JOINT ACC: %s" % eval_metrics[0])
        print("OVERALL ACC: %s" % eval_metrics[1])
    else:
        print(
            "task name not in [udc|swda|mrda|atis_intent|atis_slot|dstc2|dstc2_asr|multi-woz]"
        )


if __name__ == "__main__":
    if len(sys.argv[1:]) < 3:
        print("please input task_name predict_file reference_file")

    task_name = sys.argv[1]
    pred_file = sys.argv[2]
    refer_file = sys.argv[3]

    evaluate(task_name, pred_file, refer_file)
