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

import operator

import attr
import pyrsistent
import paddle

from text2sql.models.sql_decoder.tree_traversal import TreeTraversal


@attr.s
class ChoiceHistoryEntry:
    """ChoiceHistoryEntry"""
    rule_left = attr.ib()
    choices = attr.ib()
    probs = attr.ib()
    valid_choices = attr.ib()


class TrainTreeTraversal(TreeTraversal):
    """TrainTreeTraversal"""

    @attr.s(frozen=True)
    class XentChoicePoint:
        """XentChoicePoint"""
        logits = attr.ib()
        weight = attr.ib(default=1.0)

        def compute_loss(self, outer, idx, extra_indices):
            """compute loss"""
            if extra_indices:
                logprobs = paddle.nn.functional.log_softmax(self.logits, axis=1)
                valid_logprobs = logprobs[:, [idx] + extra_indices]
                return self.weight * outer.model.multi_loss_reduction(
                    valid_logprobs)
            else:
                # idx shape: batch (=1)
                idx = outer.model._tensor([idx])
                # loss_piece shape: batch (=1)
                loss = outer.model.xent_loss(self.logits, idx)
                return self.weight * loss

    @attr.s(frozen=True)
    class TokenChoicePoint:
        """TokenChoicePoint"""
        lstm_output = attr.ib()
        gen_logodds = attr.ib()

        def compute_loss(self, outer, token, extra_tokens):
            """compute loss"""
            return outer.model.gen_token_loss(self.lstm_output,
                                              self.gen_logodds, token,
                                              outer.desc_enc)

    def __init__(self, model, desc_enc, debug=False):
        """__init__"""
        super().__init__(model, desc_enc)
        self.choice_point = None
        self.loss = pyrsistent.pvector()

        self.debug = debug
        self.history = pyrsistent.pvector()

    def clone(self):
        """clone"""
        super_clone = super().clone()
        super_clone.choice_point = self.choice_point
        super_clone.loss = self.loss
        super_clone.debug = self.debug
        super_clone.history = self.history
        return super_clone

    def rule_choice(self, node_type, rule_logits):
        """rule_choice"""
        self.choice_point = self.XentChoicePoint(rule_logits)
        if self.debug:
            choices = []
            probs = []
            for rule_idx, logprob in sorted(self.model.rule_infer(
                    node_type, rule_logits),
                                            key=operator.itemgetter(1),
                                            reverse=True):
                _, rule = self.model.preproc.all_rules[rule_idx]
                choices.append(rule)
                probs.append(logprob.exp().item())
            self.history = self.history.append(
                ChoiceHistoryEntry(node_type, choices, probs, None))

    def token_choice(self, output, gen_logodds):
        """token_choice"""
        self.choice_point = self.TokenChoicePoint(output, gen_logodds)

    def pointer_choice(self, node_type, logits, attention_logits):
        """pointer_choice"""
        loss_weight = 1.0
        if node_type == 'value':
            loss_weight = 2.0
        self.choice_point = self.XentChoicePoint(logits, weight=loss_weight)
        self.attention_choice = self.XentChoicePoint(attention_logits,
                                                     weight=loss_weight)

    def update_using_last_choice(self, last_choice, extra_choice_info,
                                 attention_offset):
        """update_using_last_choice"""
        super().update_using_last_choice(last_choice, extra_choice_info,
                                         attention_offset)
        if last_choice is None:
            return

        if self.debug and isinstance(self.choice_point, self.XentChoicePoint):
            valid_choice_indices = [last_choice] + (
                [] if extra_choice_info is None else extra_choice_info)
            self.history[-1].valid_choices = [
                self.model.preproc.all_rules[rule_idx][1]
                for rule_idx in valid_choice_indices
            ]

        self.loss = self.loss.append(
            self.choice_point.compute_loss(self, last_choice,
                                           extra_choice_info))

        if attention_offset is not None and self.attention_choice is not None:
            self.loss = self.loss.append(
                self.attention_choice.compute_loss(self,
                                                   attention_offset,
                                                   extra_indices=None))

        self.choice_point = None
        self.attention_choice = None
