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
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_event -> [(type1, trigger1), (type2, trigger2), ...]
                gold_event -> [(type1, trigger1), (type2, trigger2), ...]
                pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                pred_record -> [{'type': type1, 'trigger': trigger1, 'roles': [(type1, role1, argument1), ...]},
                                {'type': type2, 'trigger': trigger2, 'roles': [(type2, role2, argument2), ...]},
                                ]
                gold_record -> [{'type': type1, 'trigger': trigger1, 'roles': [(type1, role1, argument1), ...]},
                                {'type': type2, 'trigger': trigger2, 'roles': [(type2, role2, argument2), ...]},
                                ]
            Counter:
        """
        pass

    @staticmethod
    def count_multi_event_role_in_instance(instance, counter):
        if len(instance['gold_event']) != len(set(instance['gold_event'])):
            counter.update(['multi-same-event-gold'])

        if len(instance['gold_role']) != len(set(instance['gold_role'])):
            counter.update(['multi-same-role-gold'])

        if len(instance['pred_event']) != len(set(instance['pred_event'])):
            counter.update(['multi-same-event-pred'])

        if len(instance['pred_role']) != len(set(instance['pred_role'])):
            counter.update(['multi-same-role-pred'])
