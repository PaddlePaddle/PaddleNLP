# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class MetricsCalculator():

    def __init__(self, shaking_tagger):
        self.shaking_tagger = shaking_tagger
        self.last_weights = None  # for exponential moving averaging

    def GHM(self, gradient, bins=10, beta=0.9):
        '''
        gradient_norm: gradient_norms of all examples in this batch; (batch_size, shaking_seq_len)
        '''
        avg = paddle.mean(gradient)
        std = paddle.std(gradient) + 1e-12
        gradient_norm = paddle.sigmoid(
            (gradient - avg) /
            std)  # normalization and pass through sigmoid to 0 ~ 1.

        min_, max_ = paddle.min(gradient_norm), paddle.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        gradient_norm = paddle.clamp(
            gradient_norm, 0,
            0.9999999)  # ensure elements in gradient_norm != 1.

        example_sum = paddle.flatten(gradient_norm).size()[0]  # N

        # calculate weights
        current_weights = paddle.zeros(bins).to(gradient.device)
        hits_vec = paddle.zeros(bins).to(gradient.device)
        count_hits = 0  # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = paddle.sum((gradient_norm <= bar)) - count_hits
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / (hits.item() +
                                                       example_sum / bins)
        # EMA: exponential moving averaging


#         print()
#         print("hits_vec: {}".format(hits_vec))
#         print("current_weights: {}".format(current_weights))
        if self.last_weights is None:
            self.last_weights = paddle.ones(bins).to(
                gradient.device)  # init by ones
        current_weights = self.last_weights * beta + (1 -
                                                      beta) * current_weights
        self.last_weights = current_weights
        #         print("ema current_weights: {}".format(current_weights))

        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins)).long()[:, :, None]
        weights_rp = current_weights[None,
                                     None, :].repeat(gradient_norm.size()[0],
                                                     gradient_norm.size()[1], 1)
        weights4examples = paddle.gather(weights_rp, -1,
                                         weight_pk_idx).squeeze(-1)
        weights4examples /= paddle.sum(weights4examples)
        return weights4examples * gradient  # return weighted gradients

    # loss func
    def _multilabel_categorical_crossentropy(self, y_pred, y_true, ghm=True):
        """
        y_pred: (batch_size, shaking_seq_len, type_size)
        y_true: (batch_size, shaking_seq_len, type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = (1 -
                  2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (
            1 - y_true) * 1e12  # mask the pred outputs of neg classes
        zeros = paddle.zeros_like(y_pred[..., :1])  # st - st
        y_pred_neg = paddle.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = paddle.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = paddle.logsumexp(y_pred_neg, dim=-1)
        pos_loss = paddle.logsumexp(y_pred_pos, dim=-1)

        if ghm:
            return (self.GHM(neg_loss + pos_loss, bins=1000)).sum()
        else:
            return (neg_loss + pos_loss).mean()

    def loss_func(self, y_pred, y_true, ghm):
        return self._multilabel_categorical_crossentropy(y_pred,
                                                         y_true,
                                                         ghm=ghm)

    def get_sample_accuracy(self, pred, truth):
        '''
        计算该batch的pred与truth全等的样本比例
        '''
        pred = pred.view(pred.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = paddle.sum(paddle.eq(truth, pred).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = paddle.eq(
            correct_tag_num,
            paddle.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = paddle.mean(sample_acc_)

        return sample_acc

    def get_mark_sets_event(self, event_list):
        trigger_iden_set, trigger_class_set, arg_iden_set, arg_class_set = set(
        ), set(), set(), set()
        for event in event_list:
            event_type = event["trigger_type"]
            trigger_offset = event["trigger_tok_span"]
            trigger_iden_set.add("{}\u2E80{}".format(trigger_offset[0],
                                                     trigger_offset[1]))
            trigger_class_set.add("{}\u2E80{}\u2E80{}".format(
                event_type, trigger_offset[0], trigger_offset[1]))
            for arg in event["argument_list"]:
                argument_offset = arg["tok_span"]
                argument_role = arg["type"]
                arg_iden_set.add("{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(
                    event_type, trigger_offset[0], trigger_offset[1],
                    argument_offset[0], argument_offset[1]))
                arg_class_set.add(
                    "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(
                        event_type, trigger_offset[0], trigger_offset[1],
                        argument_offset[0], argument_offset[1], argument_role))

        return trigger_iden_set, \
             trigger_class_set, \
             arg_iden_set, \
             arg_class_set

    def get_mark_sets_rel(self, rel_list, ent_list, pattern="only_head_text"):
        if pattern == "only_head_index":
            rel_set = set([
                "{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                            rel["predicate"],
                                            rel["obj_tok_span"][0])
                for rel in rel_list
            ])
            ent_set = set([
                "{}\u2E80{}".format(ent["tok_span"][0], ent["type"])
                for ent in ent_list
            ])

        elif pattern == "whole_span":
            rel_set = set([
                "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(
                    rel["subj_tok_span"][0], rel["subj_tok_span"][1],
                    rel["predicate"], rel["obj_tok_span"][0],
                    rel["obj_tok_span"][1]) for rel in rel_list
            ])
            ent_set = set([
                "{}\u2E80{}\u2E80{}".format(ent["tok_span"][0],
                                            ent["tok_span"][1], ent["type"])
                for ent in ent_list
            ])

        elif pattern == "whole_text":
            rel_set = set([
                "{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"],
                                            rel["object"]) for rel in rel_list
            ])
            ent_set = set([
                "{}\u2E80{}".format(ent["text"], ent["type"])
                for ent in ent_list
            ])

        elif pattern == "only_head_text":
            rel_set = set([
                "{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0],
                                            rel["predicate"],
                                            rel["object"].split(" ")[0])
                for rel in rel_list
            ])
            ent_set = set([
                "{}\u2E80{}".format(ent["text"].split(" ")[0], ent["type"])
                for ent in ent_list
            ])

        return rel_set, ent_set

    def _cal_cpg(self, pred_set, gold_set, cpg):
        '''
        cpg is a list: [correct_num, pred_num, gold_num]
        '''
        for mark_str in pred_set:
            if mark_str in gold_set:
                cpg[0] += 1
        cpg[1] += len(pred_set)
        cpg[2] += len(gold_set)

    def cal_rel_cpg(self, pred_rel_list, pred_ent_list, gold_rel_list,
                    gold_ent_list, ere_cpg_dict, pattern):
        '''
        ere_cpg_dict = {
                "rel_cpg": [0, 0, 0],
                "ent_cpg": [0, 0, 0],
                }
        pattern: metric pattern
        '''
        gold_rel_set, gold_ent_set = self.get_mark_sets_rel(
            gold_rel_list, gold_ent_list, pattern)
        pred_rel_set, pred_ent_set = self.get_mark_sets_rel(
            pred_rel_list, pred_ent_list, pattern)

        self._cal_cpg(pred_rel_set, gold_rel_set, ere_cpg_dict["rel_cpg"])
        self._cal_cpg(pred_ent_set, gold_ent_set, ere_cpg_dict["ent_cpg"])

    def cal_event_cpg(self, pred_event_list, gold_event_list, ee_cpg_dict):
        '''
        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        '''
        pred_trigger_iden_set, \
        pred_trigger_class_set, \
        pred_arg_iden_set, \
        pred_arg_class_set = self.get_mark_sets_event(pred_event_list)

        gold_trigger_iden_set, \
        gold_trigger_class_set, \
        gold_arg_iden_set, \
        gold_arg_class_set = self.get_mark_sets_event(gold_event_list)

        self._cal_cpg(pred_trigger_iden_set, gold_trigger_iden_set,
                      ee_cpg_dict["trigger_iden_cpg"])
        self._cal_cpg(pred_trigger_class_set, gold_trigger_class_set,
                      ee_cpg_dict["trigger_class_cpg"])
        self._cal_cpg(pred_arg_iden_set, gold_arg_iden_set,
                      ee_cpg_dict["arg_iden_cpg"])
        self._cal_cpg(pred_arg_class_set, gold_arg_class_set,
                      ee_cpg_dict["arg_class_cpg"])

    def get_cpg(self,
                sample_list,
                tok2char_span_list,
                batch_pred_shaking_tag,
                pattern="only_head_text"):
        '''
        return correct number, predict number, gold number (cpg)
        '''

        ee_cpg_dict = {
            "trigger_iden_cpg": [0, 0, 0],
            "trigger_class_cpg": [0, 0, 0],
            "arg_iden_cpg": [0, 0, 0],
            "arg_class_cpg": [0, 0, 0],
        }
        ere_cpg_dict = {
            "rel_cpg": [0, 0, 0],
            "ent_cpg": [0, 0, 0],
        }

        # go through all sentences
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_shaking_tag = batch_pred_shaking_tag[ind]

            pred_rel_list, pred_ent_list = self.shaking_tagger.decode_rel(
                text, pred_shaking_tag, tok2char_span)  # decoding
            gold_rel_list = sample["relation_list"]
            gold_ent_list = sample["entity_list"]

            if pattern == "event_extraction":
                pred_event_list = self.shaking_tagger.trans2ee(
                    pred_rel_list, pred_ent_list)  # transform to event list
                gold_event_list = sample["event_list"]
                self.cal_event_cpg(pred_event_list, gold_event_list,
                                   ee_cpg_dict)
            else:
                self.cal_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list,
                                 gold_ent_list, ere_cpg_dict, pattern)

        if pattern == "event_extraction":
            return ee_cpg_dict
        else:
            return ere_cpg_dict

    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-12
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1
