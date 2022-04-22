#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Counter, Tuple


class PredictParser:
    def __init__(self, label_constraint=None, tokenizer=None):
        self.predicate_set = label_constraint.type_list if label_constraint else list(
        )
        self.role_set = label_constraint.role_list if label_constraint else list(
        )
        self.tokenizer = tokenizer

    def decode(self, gold_list, pred_list, text_list=None,
               raw_list=None) -> Tuple[List, Counter]:
        pass
