#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List
import sys


def tuple_offset(offset):
    if isinstance(offset, tuple):
        return offset
    else:
        return tuple(offset)


def warning_tp_increment(gold, pred, prefix):
    sys.stderr.write(
        f"{prefix} TP Increment Warning, Gold Offset: {gold['offset']}\n")
    sys.stderr.write(
        f"{prefix} TP Increment Warning, Pred Offset: {pred['offset']}\n")
    sys.stderr.write(
        f"{prefix} TP Increment Warning, Gold String: {gold['string']}\n")
    sys.stderr.write(
        f"{prefix} TP Increment Warning, Pred String: {pred['string']}\n")
    sys.stderr.write(f"===============\n")


class Metric:
    """ Tuple Metric """

    def __init__(self, verbose=False, match_mode='normal'):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.
        self.verbose = verbose
        self.match_mode = match_mode
        assert self.match_mode in {'set', 'normal', 'multimatch'}

    def __repr__(self) -> str:
        return f"tp: {self.tp}, gold: {self.gold_num}, pred: {self.pred_num}"

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {
            prefix + 'tp': tp,
            prefix + 'gold': gold_num,
            prefix + 'pred': pred_num,
            prefix + 'P': p * 100,
            prefix + 'R': r * 100,
            prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
        }

    def count_instance(self, gold_list, pred_list):
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            self.tp += len(gold_list & pred_list)

        else:
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)

            if len(gold_list) > 0 and len(pred_list) > 0:
                # guarantee length same
                assert len(gold_list[0]) == len(pred_list[0])

            dup_gold_list = deepcopy(gold_list)
            for pred in pred_list:
                if pred in dup_gold_list:
                    self.tp += 1
                    if self.match_mode == 'normal':
                        # Each Gold Instance can be matched one time
                        dup_gold_list.remove(pred)

    def count_batch_instance(self, batch_gold_list, batch_pred_list):
        for gold_list, pred_list in zip(batch_gold_list, batch_pred_list):
            self.count_instance(gold_list=gold_list, pred_list=pred_list)


class Scorer:
    @staticmethod
    def load_gold_list(gold_list, offset_key=None):
        raise NotImplementedError

    @staticmethod
    def load_pred_list(pred_list):
        raise NotImplementedError

    @staticmethod
    def eval_instance_list(gold_instance_list,
                           pred_instance_list,
                           verbose=False,
                           match_mode='normal'):
        raise NotImplementedError


class EntityScorer(Scorer):
    @staticmethod
    def load_gold_list(gold_list: List[List[Dict]]):
        """ Load gold instance to `string` and `offset`

        Args:
            gold_list (List[List[Dict]]): [description]
                [
                    [
                        {'type': 'Geo-political', 'offset': [7], 'text': 'seattle'},
                        {'type': 'Location', 'offset': [11], 'text': 'lot'},
                        {'type': 'Geo-political', 'offset': [14], 'text': 'city'}
                    ],
                    [...]
                ]

        Returns:
            List[Dict]: each instance has `offset` and `string`
                [
                    {
                        'offset': [('Geo-political', (7,)), ('Location', (11,)), ('Geo-political', (14,))],
                        'string': [('Geo-political', 'seattle'), ('Location', 'lot'), ('Geo-political', 'city')]
                    },
                    {...}, ...
                ]
        """
        gold_instance_list = []
        for gold in gold_list:
            gold_offset = list()
            gold_string = list()
            for span in gold:
                span_label = span['type']
                span_offset = span['offset']
                span_text = span['text']
                gold_offset += [(span_label, tuple_offset(span_offset))]
                gold_string += [(span_label, span_text)]
            gold_instance = {
                'offset': gold_offset,
                'string': gold_string,
            }
            gold_instance_list += [gold_instance]
        return gold_instance_list

    @staticmethod
    def load_pred_list(pred_list: List[Dict]):
        """[summary]

        Args:
            pred_list (List[Dict]): [description]
                [
                    {
                        'offset': [['Geo-political', [7]], ['Geo-political', [14]]],
                        'string': [['Geo-political', 'seattle'], ['Geo-political', 'city']]
                    },
                    {...},
                ]
        Returns:
            List[Dict] : each relation instance has `offset` and `string`
                [
                    {
                        'offset': [('Geo-political', (7,)), ('Geo-political', (14,))],
                        'string': [('Geo-political', 'seattle'), ('Geo-political', 'city')]
                    }
                ]
        """
        pred_instance_list = list()
        for pred in pred_list:
            for offset_pred in pred['offset']:
                if not isinstance(offset_pred[1], tuple):
                    offset_pred[1] = tuple_offset(offset_pred[1])
            pred['offset'] = [tuple_offset(p) for p in pred['offset']]
            pred['string'] = [tuple_offset(p) for p in pred['string']]
            pred_instance_list += [pred]
        return pred_instance_list

    @staticmethod
    def eval_instance_list(gold_instance_list: List[Dict],
                           pred_instance_list: List[Dict],
                           verbose=False,
                           match_mode='normal'):
        """[summary]

        Args:
            gold_instance_list (List[Dict]): [description]
                [
                    {
                        'offset': [('Geo-political', (7,)), ('Location', (11,)), ('Geo-political', (14,))],
                        'string': [('Geo-political', 'seattle'), ('Location', 'lot'), ('Geo-political', 'city')]
                    },
                    {...}, ...
                ]
            pred_instance_list (List[Dict]): [description]
                [
                    {
                        'offset': [('Geo-political', (7,)), ('Geo-political', (14,))],
                        'string': [('Geo-political', 'seattle'), ('Geo-political', 'city')]
                    }
                ]
            verbose (bool, optional): [description]. Defaults to False.
            match_mode (string, optional): [description]. Defaults to `normal` .

        Returns:
            Dict: Result of Evaluation
                (offset, string) X (gold, pred, tp, P, R, F1)
        """
        metrics = {
            'string': Metric(
                verbose=verbose, match_mode=match_mode),
            'offset': Metric(
                verbose=verbose, match_mode=match_mode),
        }
        for pred, gold in zip(pred_instance_list, gold_instance_list):

            pre_string_tp, pre_offset_tp = metrics['string'].tp, metrics[
                'offset'].tp

            for eval_key in metrics:
                metrics[eval_key].count_instance(
                    gold_list=gold.get(eval_key, []),
                    pred_list=pred.get(eval_key, []))

            post_string_tp, post_offset_tp = metrics['string'].tp, metrics[
                'offset'].tp
            if verbose and post_offset_tp - pre_offset_tp != post_string_tp - pre_string_tp:
                warning_tp_increment(gold=gold, pred=pred, prefix='Entity')

        results = dict()
        for eval_key in metrics:
            results.update(metrics[eval_key].compute_f1(prefix=eval_key +
                                                        '-ent-'))

        return results


class RelationScorer(Scorer):
    @staticmethod
    def load_gold_list(gold_list: List[List[Dict]]):
        """[summary]

        Args:
            gold_list (List[List[Dict]]): List of Sentece, each sentence contains a List of Relation Dict
                [
                    [
                        {
                            'type': 'Part-whole',
                            'args': [{'type': 'Location', 'offset': [11], 'text': 'lot'}, {'type': 'Geo-political', 'offset': [14], 'text': 'city'}]
                        }, ...
                    ],
                    [...],
                ]

        Returns:
            List[Dict]: List of Sentece, each sentence contains two List (offset, string) of Relation Tuple
                [
                    {
                        'offset': [('Part-whole', 'Geo-political', (0,), 'Geo-political', (2,)), ... ],
                        'string': [('Part-whole', 'Geo-political', 'MULTAN', 'Geo-political', 'Pakistan'), ...]
                    }
                ]
        """
        gold_instance_list = []
        for gold in gold_list:
            gold_instance = defaultdict(list)
            for record in gold:
                assert len(record['args']) == 2
                gold_instance['offset'] += [(
                    record['type'],
                    record['args'][0]['type'],
                    tuple_offset(record['args'][0]['offset']),
                    record['args'][1]['type'],
                    tuple_offset(record['args'][1]['offset']), )]
                gold_instance['string'] += [(
                    record['type'],
                    record['args'][0]['type'],
                    record['args'][0]['text'],
                    record['args'][1]['type'],
                    record['args'][1]['text'], )]
            gold_instance_list += [gold_instance]

        return gold_instance_list

    @staticmethod
    def load_pred_list(pred_list):
        """[summary]

        Args:
            pred_list (List[Dict]): List of Sentece, each sentence contains two List (offset, string) of Relation List
                [
                    {
                        'offset': [['Part-whole', 'Geo-political', [0], 'Geo-political', [2]]],
                        'string': [['Part-whole', 'Geo-political', 'MULTAN', 'Geo-political', 'Pakistan']],
                    }, ...
                ]
        Returns:
            List[Dict]: List of Sentece, each sentence contains two List (offset, string) of Relation Tuple
                [
                    {
                        'offset': [('Part-whole', 'Geo-political', (0,), 'Geo-political', (2,))],
                        'string': [('Part-whole', 'Geo-political', 'MULTAN', 'Geo-political', 'Pakistan')]
                    }, ...
                ]
        """
        pred_instance_list = list()
        for pred in pred_list:
            for offset_pred in pred['offset']:

                if not isinstance(offset_pred[2], tuple):
                    offset_pred[2] = tuple_offset(offset_pred[2])

                if not isinstance(offset_pred[4], tuple):
                    offset_pred[4] = tuple_offset(offset_pred[4])

            pred['offset'] = [tuple_offset(p) for p in pred['offset']]
            pred['string'] = [tuple_offset(p) for p in pred['string']]
            pred_instance_list += [pred]
        return pred_instance_list

    @staticmethod
    def eval_instance_list(gold_instance_list,
                           pred_instance_list,
                           verbose=False,
                           match_mode='normal'):
        """[summary]

        Args:
            gold_instance_list (List[Dict]): List of Sentece, each sentence contains two List (offset, string) of Relation Tuple
                [
                    {
                        'offset': [('Part-whole', 'Geo-political', (0,), 'Geo-political', (2,)), ... ],
                        'string': [('Part-whole', 'Geo-political', 'MULTAN', 'Geo-political', 'Pakistan'), ...]
                    }
                ]
            pred_instance_list ([type]): List of Sentece, each sentence contains two List (offset, string) of Relation Tuple
                [
                    {
                        'offset': [('Part-whole', 'Geo-political', (0,), 'Geo-political', (2,))],
                        'string': [('Part-whole', 'Geo-political', 'MULTAN', 'Geo-political', 'Pakistan')]
                    }, ...
                ]
            verbose (bool, optional): Defaults to False.
            match_mode (string, optional): [description]. Defaults to `normal` .

        Returns:
            Dict: Result of Evaluation
                (offset, string) X (boundary, strict) X (gold, pred, tp, P, R, F1)
        """
        # Span Boundary and Type
        metrics = {
            'offset': Metric(
                verbose=verbose, match_mode=match_mode),
            'string': Metric(
                verbose=verbose, match_mode=match_mode),
        }
        # Span Boundary Only
        boundary_metrics = {
            'offset': Metric(
                verbose=verbose, match_mode=match_mode),
            'string': Metric(
                verbose=verbose, match_mode=match_mode),
        }
        for pred, gold in zip(pred_instance_list, gold_instance_list):

            pre_string_tp, pre_offset_tp = metrics['string'].tp, metrics[
                'offset'].tp

            for eval_key in metrics:
                # Span Boundary and Type
                metrics[eval_key].count_instance(
                    gold_list=gold.get(eval_key, []),
                    pred_list=pred.get(eval_key, []), )

            post_string_tp, post_offset_tp = metrics['string'].tp, metrics[
                'offset'].tp
            if verbose and (post_offset_tp - pre_offset_tp !=
                            post_string_tp - pre_string_tp):
                warning_tp_increment(
                    gold=gold, pred=pred, prefix='Relation Strict')

            pre_string_tp, pre_offset_tp = boundary_metrics[
                'string'].tp, boundary_metrics['offset'].tp

            for eval_key in boundary_metrics:
                # Span Boundary Only
                boundary_metrics[eval_key].count_instance(
                    gold_list=[(x[0], x[2], x[4])
                               for x in gold.get(eval_key, [])],
                    pred_list=[(x[0], x[2], x[4])
                               for x in pred.get(eval_key, [])], )
            post_string_tp, post_offset_tp = boundary_metrics[
                'string'].tp, boundary_metrics['offset'].tp
            if verbose and post_offset_tp - pre_offset_tp != post_string_tp - pre_string_tp:
                warning_tp_increment(
                    gold=gold, pred=pred, prefix='Relation Boundary')

        results = dict()
        for eval_key in metrics:
            results.update(metrics[eval_key].compute_f1(prefix=eval_key +
                                                        '-rel-strict-'))
        for eval_key in boundary_metrics:
            results.update(boundary_metrics[eval_key].compute_f1(
                prefix=eval_key + '-rel-boundary-'))
        return results


class EventScorer(Scorer):
    @staticmethod
    def load_gold_list(gold_list):
        """[summary]

        Args:
            gold_list (List[List[Dict]]): List of Sentece, each sentence contains a List of Event Dict
                [
                    [ # Sentance
                        { # Event Record
                            'type': 'Die',
                            'offset': [16],
                            'text': 'shot',
                            'args': [
                                {'type': 'Victim', 'offset': [17], 'text': 'himself'},
                                {'type': 'Agent', 'offset': [5, 6], 'text': 'John Joseph'},
                                {'type': 'Place', 'offset': [23], 'text': 'court'}
                            ]
                        },
                    ]
                ]

        Returns:
            List[Dict]: List of Sentece, each sentence contains Four List of Event Tuple
                [
                    {
                        'offset_trigger': [('Die', (16,)), ('Convict', (30,))],
                        'string_trigger': [('Die', 'shot'), ('Convict', 'convicted')],
                        'offset_role': [('Die', 'Victim', (17,)), ('Die', 'Agent', (5, 6)), ('Die', 'Place', (23,))],
                        'string_role': [('Die', 'Victim', 'himself'), ('Die', 'Agent', 'John Joseph'), ('Die', 'Place', 'court')]
                    },
                    ...
                ]
        """
        gold_instance_list = []
        for gold in gold_list:
            gold_instance = defaultdict(list)
            for record in gold:
                gold_instance['offset_trigger'] += [
                    (record['type'], tuple_offset(record['offset']))
                ]
                gold_instance['string_trigger'] += [
                    (record['type'], record['text'])
                ]
                for arg in record['args']:
                    gold_instance['offset_role'] += [(
                        record['type'], arg['type'],
                        tuple_offset(arg['offset']))]
                    gold_instance['string_role'] += [
                        (record['type'], arg['type'], arg['text'])
                    ]
            gold_instance_list += [gold_instance]
        return gold_instance_list

    @staticmethod
    def load_pred_list(pred_list):
        """[summary]

        Args:
            pred_list (List[Dict]): List of Sentece, each sentence contains two List (offset, string) of Event List
                [
                    {
                        'offset': [{'type': 'Attack', 'roles': [['Attacker', [5, 6]], ['Place', [23]], ['Target', [17]]], 'trigger': [16]}],
                        'string': [{'roles': [['Attacker', 'John Joseph'], ['Place', 'court'], ['Target', 'himself']], 'type': 'Attack', 'trigger': 'shot'}],
                    },
                    ...
                ]
        Returns:
            List[Dict]: List of Sentece, each sentence contains four List (offset, string) X (trigger, role) of Event List
                [
                    {
                        'offset_trigger': [('Attack', (16,))],
                        'offset_role': [('Attack', 'Attacker', (5, 6)), ('Attack', 'Place', (23,)), ('Attack', 'Target', (17,))],
                        'string_trigger': [('Attack', 'shot')],
                        'string_role': [('Attack', 'Attacker', 'John Joseph'), ('Attack', 'Place', 'court'), ('Attack', 'Target', 'himself')],
                    },
                    ...
                ]
        """
        pred_instance_list = list()
        for pred in pred_list:
            pred_instance = defaultdict(list)

            for offset_pred in pred['offset']:
                event_type, trigger_offset = offset_pred['type'], tuple_offset(
                    offset_pred['trigger'])
                pred_instance['offset_trigger'] += [(event_type, trigger_offset)
                                                    ]
                for role_type, role_offset in offset_pred['roles']:
                    pred_instance['offset_role'] += [(
                        event_type, role_type, tuple_offset(role_offset))]

            for string_pred in pred['string']:
                event_type, trigger_string = string_pred['type'], string_pred[
                    'trigger']
                pred_instance['string_trigger'] += [(event_type, trigger_string)
                                                    ]
                for role_type, role_string in string_pred['roles']:
                    pred_instance['string_role'] += [(event_type, role_type,
                                                      role_string)]
            pred_instance_list += [pred_instance]
        return pred_instance_list

    @staticmethod
    def eval_instance_list(gold_instance_list,
                           pred_instance_list,
                           verbose=False,
                           match_mode='normal'):
        """[summary]

        Args:
            gold_instance_list (List[Dict]): List of Sentece, each sentence contains Four List of Event Tuple
                [
                    {
                        'offset_trigger': [('Die', (16,)), ('Convict', (30,))],
                        'string_trigger': [('Die', 'shot'), ('Convict', 'convicted')],
                        'offset_role': [('Die', 'Victim', (17,)), ('Die', 'Agent', (5, 6)), ('Die', 'Place', (23,))],
                        'string_role': [('Die', 'Victim', 'himself'), ('Die', 'Agent', 'John Joseph'), ('Die', 'Place', 'court')]
                    },
                    ...
                ]
            pred_instance_list (List[Dict]): List of Sentece, each sentence contains four List (offset, string) X (trigger, role) of Event List
                [
                    {
                        'offset_trigger': [('Attack', (16,))],
                        'offset_role': [('Attack', 'Attacker', (5, 6)), ('Attack', 'Place', (23,)), ('Attack', 'Target', (17,))],
                        'string_trigger': [('Attack', 'shot')],
                        'string_role': [('Attack', 'Attacker', 'John Joseph'), ('Attack', 'Place', 'court'), ('Attack', 'Target', 'himself')],
                    },
                    ...
                ]
            verbose (bool, optional): [description]. Defaults to False.
            match_mode (string, optional): [description]. Defaults to `normal`.

        Returns:
            Dict: Result of Evaluation
                (offset, string) X (trigger, role) X (gold, pred, tp, P, R, F1)
        """
        trigger_metrics = {
            'offset': Metric(
                verbose=verbose, match_mode=match_mode),
            'string': Metric(
                verbose=verbose, match_mode=match_mode),
        }
        role_metrics = {
            'offset': Metric(
                verbose=verbose, match_mode=match_mode),
            'string': Metric(
                verbose=verbose, match_mode=match_mode),
        }

        for pred, gold in zip(pred_instance_list, gold_instance_list):

            pre_string_tp, pre_offset_tp = trigger_metrics[
                'string'].tp, trigger_metrics['offset'].tp

            for eval_key in trigger_metrics:
                trigger_metrics[eval_key].count_instance(
                    gold_list=gold.get(eval_key + '_trigger', []),
                    pred_list=pred.get(eval_key + '_trigger', []))

            post_string_tp, post_offset_tp = trigger_metrics[
                'string'].tp, trigger_metrics['offset'].tp
            if verbose and post_offset_tp - pre_offset_tp != post_string_tp - pre_string_tp:
                warning_tp_increment(gold=gold, pred=pred, prefix='Trigger')

            pre_string_tp, pre_offset_tp = role_metrics[
                'string'].tp, role_metrics['offset'].tp

            for eval_key in role_metrics:
                role_metrics[eval_key].count_instance(
                    gold_list=gold.get(eval_key + '_role', []),
                    pred_list=pred.get(eval_key + '_role', []))

            post_string_tp, post_offset_tp = role_metrics[
                'string'].tp, role_metrics['offset'].tp
            if verbose and post_offset_tp - pre_offset_tp != post_string_tp - pre_string_tp:
                warning_tp_increment(gold=gold, pred=pred, prefix='Role')

        results = dict()
        for eval_key in trigger_metrics:
            results.update(trigger_metrics[eval_key].compute_f1(
                prefix=f'{eval_key}-evt-trigger-'))
        for eval_key in role_metrics:
            results.update(role_metrics[eval_key].compute_f1(
                prefix=f'{eval_key}-evt-role-'))

        return results
