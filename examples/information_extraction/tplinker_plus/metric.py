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


class MetricCalculator():
    '''Metric for TPLinkerPlus'''

    def __init__(self, shaking_tagger, task_type="relation_extraction"):
        super(MetricCalculator, self).__init__()
        self.shaking_tagger = shaking_tagger
        self.task_type = task_type
        self.total_sample_acc = 0.
        self.batch_cnt = 0
        self.total_cpg_dict = {
            "rel_cpg": [0, 0, 0],
            "ent_cpg": [0, 0, 0],
        }

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

    def compute(self,
                sample_list,
                tok2char_span_list,
                batch_pred_shaking_tag,
                pattern="only_head_text"):
        '''
        return correct number, predict number, gold number (cpg)
        '''

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
            if len(pred_rel_list) != 0 or len(pred_ent_list) != 0:
                print(pred_rel_list)
                print(pred_ent_list)
                print(gold_rel_list)
                print(gold_ent_list)
            self.cal_rel_cpg(pred_rel_list, pred_ent_list, gold_rel_list,
                             gold_ent_list, ere_cpg_dict, pattern)

        return ere_cpg_dict

    def get_prf_scores(self, correct_num, pred_num, gold_num):
        precision = float(correct_num / pred_num) if pred_num else 0.
        recall = float(correct_num / gold_num) if gold_num else 0.
        f1 = float(2 * precision * recall /
                   (precision + recall)) if correct_num else 0.
        return precision, recall, f1

    def update(self, cpg_dict):
        for k, cpg in cpg_dict.items():
            for i in range(len(cpg)):
                self.total_cpg_dict[k][i] += cpg[i]

    def accumulate(self):
        if self.task_type == "relation_extraction":
            rel_prf = self.get_prf_scores(self.total_cpg_dict["rel_cpg"][0],
                                          self.total_cpg_dict["rel_cpg"][1],
                                          self.total_cpg_dict["rel_cpg"][2])
            ent_prf = self.get_prf_scores(self.total_cpg_dict["ent_cpg"][0],
                                          self.total_cpg_dict["ent_cpg"][1],
                                          self.total_cpg_dict["ent_cpg"][2])
            log_dict = {
                "rel_prec": rel_prf[0],
                "rel_recall": rel_prf[1],
                "rel_f1": rel_prf[2],
                "ent_prec": ent_prf[0],
                "ent_recall": ent_prf[1],
                "ent_f1": ent_prf[2],
            }
        elif self.task_type == "ner":
            ent_prf = self.get_prf_scores(self.total_cpg_dict["ent_cpg"][0],
                                          self.total_cpg_dict["ent_cpg"][1],
                                          self.total_cpg_dict["ent_cpg"][2])
            log_dict = {
                "ent_prec": ent_prf[0],
                "ent_recall": ent_prf[1],
                "ent_f1": ent_prf[2],
            }
        return log_dict

    def reset(self):
        self.total_cpg_dict = {
            "rel_cpg": [0, 0, 0],
            "ent_cpg": [0, 0, 0],
        }
