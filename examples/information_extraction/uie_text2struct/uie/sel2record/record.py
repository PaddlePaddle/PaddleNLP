#!/usr/bin/env python
# -*- coding:utf-8 -*-
from asyncio.log import logger
from typing import Tuple
import numpy
import logging

logger = logging.getLogger("__main__")


def match_sublist(the_list, to_match):
    """
    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match: [1, 2]
    :return:
        [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def check_overlap(x, y):
    if x[0] > y[1] or y[0] > x[1]:
        return False
    else:
        return True


def get_index_tuple(matched: Tuple[int, int]):
    return tuple(range(matched[0], matched[1] + 1))


def span_to_token(text, span_to_token_strategy='space'):
    if span_to_token_strategy == 'space':
        return text.split(' ')
    elif span_to_token_strategy == 'list':
        return list(text)
    else:
        raise NotImplementedError(
            f"The span to token strategy {span_to_token_strategy} is not implemented."
        )


class MapConfig:
    def __init__(self,
                 map_strategy: str='first',
                 de_duplicate: bool=True,
                 span_to_token: str='space') -> None:
        self.map_strategy = map_strategy
        self.de_duplicate = de_duplicate
        self.span_to_token = span_to_token

    def __repr__(self) -> str:
        repr_list = [
            f"map_strategy: {self.map_strategy}",
            f"de_duplicate: {self.de_duplicate}",
            f"span_to_token: {self.span_to_token}",
        ]
        return ', '.join(repr_list)

    @staticmethod
    def load_from_yaml(config_file):
        import yaml
        with open(config_file) as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        return MapConfig(
            map_strategy=config['map_strategy'],
            de_duplicate=config['de_duplicate'],
            span_to_token=config['span_to_token'], )


class Record:
    def __init__(self, map_config) -> None:
        self._map_config = map_config

    def span_to_token(self, text):
        return span_to_token(
            text, span_to_token_strategy=self._map_config.span_to_token)


class EntityRecord(Record):
    @staticmethod
    def to_string(pred_record_list):
        entity_list = list()
        for pred_record in pred_record_list:
            record_type, record_text = pred_record['type'], pred_record[
                'trigger']
            if record_text == "":
                logger.warning(f"Empty Extraction {pred_record}")
                continue
            entity_list += [(record_type, record_text)]
        return entity_list

    def to_offset(self, instance, tokens):
        # map_strategy='first', de_duplicate=True
        map_strategy_dict = {
            'first': self.record_to_offset_first_role,
            'closest': self.record_to_offset_closest_role,
            'longer_first': self.record_to_offset_longer_first,
        }

        if self._map_config.map_strategy in map_strategy_dict:
            map_function = map_strategy_dict[self._map_config.map_strategy]
            return map_function(
                instance=instance,
                token_list=tokens, )
        else:
            raise NotImplementedError(
                f"The map strategy {self._map_config.map_strategy} in {self.__class__} is not implemented."
            )

    def record_to_offset_closest_role(
            self,
            instance,
            token_list, ):
        """
        Find Role's offset using closest matched with trigger work.
        :param instance:
        :return:
        """
        return self.record_to_offset_first_role(instance, token_list=token_list)

    def record_to_offset_first_role(self, instance, token_list):
        """
        Find Entity's offset using first matched in the sentence.
        :param instance:
        :return:
        """
        entity_list = list()

        entity_matched_set = set()
        for pred_record in instance:
            record_type, record_text = pred_record['type'], pred_record[
                'trigger']
            if record_text == "":
                logger.warning(f"Empty Extraction {pred_record}")
                continue
            matched_list = match_sublist(token_list,
                                         self.span_to_token(record_text))
            for matched in matched_list:
                if (record_type, matched) not in entity_matched_set:
                    entity_list += [(record_type,
                                     tuple(range(matched[0], matched[1] + 1)))]
                    entity_matched_set.add((record_type, matched))
                    break

        return entity_list

    def record_to_offset_longer_first(self, instance, token_list):
        """
        Find Entity's offset using first matched in the sentence.
        :param instance:
        :return:
        """
        entity_list = list()

        entity_matched_set = set()
        for x in instance:
            x['length'] = len(x['trigger'])
        instance.sort(reverse=True, key=lambda x: x['length'])

        for pred_record in instance:
            record_type, record_text = pred_record['type'], pred_record[
                'trigger']
            if record_text == "":
                logger.warning(f"Empty Extraction {pred_record}")
                continue

            matched_list = match_sublist(token_list,
                                         self.span_to_token(record_text))
            for matched in matched_list:
                flag = False
                for _, g in entity_matched_set:
                    if check_overlap(g, matched):
                        flag = True
                if flag:
                    continue

                if (record_type, matched) not in entity_matched_set:
                    entity_list += [(record_type,
                                     tuple(range(matched[0], matched[1] + 1)))]
                    entity_matched_set.add((record_type, matched))
                    break

        return entity_list


class RelationRecord(Record):
    def to_offset(self, instance, tokens):
        map_strategy_dict = {
            'first': self.record_to_offset_first_role,
            'closest': self.record_to_offset_closest_role,
            'longer_first': self.record_to_offset_closest_role,
        }
        if self._map_config.map_strategy in map_strategy_dict:
            map_function = map_strategy_dict[self._map_config.map_strategy]
            return map_function(
                instance=instance,
                token_list=tokens, )
        else:
            raise NotImplementedError(
                f"The map strategy {self._map_config.map_strategy} in {self.__class__} is not implemented."
            )

    @staticmethod
    def to_string(instance):
        relation_list = list()
        for record in instance:
            relation_type = record['type']
            relation = [relation_type]
            if len(record['roles']) < 2:
                continue
            for role_type, text_str in record['roles'][:2]:
                relation += [role_type, text_str]
            relation_list += [tuple(relation)]
        return relation_list

    def record_to_offset_first_role(self, instance, token_list):
        """
        Find Role's offset using first matched in the sentence.
        :param instance:
        :return:
        """
        relation_list = list()

        for record in instance:
            relation_type = record['type']

            if len(record['roles']) < 2:
                continue

            relation = [relation_type]
            for role_type, text_str in record['roles'][:2]:
                matched_list = match_sublist(token_list,
                                             self.span_to_token(text_str))
                if len(matched_list) == 0:
                    logger.warning("[Cannot reconstruct]: %s %s\n" %
                                   (text_str, token_list))
                    break
                relation += [role_type, get_index_tuple(matched_list[0])]
            if len(relation) != 5 or (self._map_config.de_duplicate and
                                      tuple(relation) in relation_list):
                continue
            relation_list += [tuple(relation)]

        return relation_list

    def record_to_offset_closest_role(self, instance, token_list):
        """
        Find Role's offset using closest matched with trigger work.
        :param instance:
        :return:
        """
        relation_list = list()

        for record in instance:
            relation_type = record['type']

            if len(record['roles']) < 2:
                continue

            arg1_type, arg1_text = record['roles'][0]
            arg2_type, arg2_text = record['roles'][1]
            arg1_matched_list = match_sublist(token_list,
                                              self.span_to_token(arg1_text))
            arg2_matched_list = match_sublist(token_list,
                                              self.span_to_token(arg2_text))

            if len(arg1_matched_list) == 0:
                logger.warning("[Cannot reconstruct]: %s %s\n" %
                               (arg1_text, token_list))
                break
            if len(arg2_matched_list) == 0:
                logger.warning("[Cannot reconstruct]: %s %s\n" %
                               (arg2_text, token_list))
                break

            distance_tuple = list()
            for arg1_match in arg1_matched_list:
                for arg2_match in arg2_matched_list:
                    distance = abs(arg1_match[0] - arg2_match[0])
                    distance_tuple += [(distance, arg1_match, arg2_match)]
            distance_tuple.sort()

            relation = [
                relation_type,
                arg1_type,
                get_index_tuple(distance_tuple[0][1]),
                arg2_type,
                get_index_tuple(distance_tuple[0][2]),
            ]
            if self._map_config.de_duplicate and tuple(
                    relation) in relation_list:
                continue
            relation_list += [tuple(relation)]

        return relation_list


class EventRecord(Record):
    def to_offset(self, instance, tokens):
        map_strategy_dict = {
            'first': self.record_to_offset_first_role,
            'closest': self.record_to_offset_closest_role,
            'longer_first': self.record_to_offset_closest_role,
        }
        if self._map_config.map_strategy in map_strategy_dict:
            map_function = map_strategy_dict[self._map_config.map_strategy]
            return map_function(
                instance=instance,
                token_list=tokens, )
        else:
            raise NotImplementedError(
                f"The map strategy {self._map_config.map_strategy} in {self.__class__} is not implemented."
            )

    @staticmethod
    def to_string(instance):
        """
        {'type': 'Justice:Appeal',
         'trigger': 'appeal',
         'roles': [
            ('Adjudicator', 'court'),
            ('Plaintiff', 'Anwar')
            ], }
        """
        return instance

    def record_to_offset_first_role(self, instance, token_list):
        """
        Find Role's offset using first matched in the sentence.
        :param instance:
        :return:
        """
        record_list = list()

        trigger_matched_set = set()
        for record in instance:
            event_type = record['type']
            trigger = record['trigger']
            matched_list = match_sublist(token_list,
                                         self.span_to_token(trigger))

            if len(matched_list) == 0:
                logger.warning("[Cannot reconstruct]: %s %s\n" %
                               (trigger, token_list))
                continue

            trigger_offset = None
            for matched in matched_list:
                if matched not in trigger_matched_set:
                    trigger_offset = get_index_tuple(matched)
                    trigger_matched_set.add(matched)
                    break

            # No trigger word, skip the record
            if trigger_offset is None:
                break

            pred_record = {
                'type': event_type,
                'roles': [],
                'trigger': trigger_offset
            }

            for role_type, text_str in record['roles']:
                matched_list = match_sublist(token_list,
                                             self.span_to_token(text_str))
                if len(matched_list) == 0:
                    logger.warning("[Cannot reconstruct]: %s %s\n" %
                                   (text_str, token_list))
                    continue
                pred_record['roles'] += [(role_type,
                                          get_index_tuple(matched_list[0]))]

            record_list += [pred_record]

        return record_list

    def record_to_offset_closest_role(self, instance, token_list):
        """
        Find Role's offset using closest matched with trigger work.
        :param instance:
        :return:
        """
        record_list = list()

        trigger_matched_set = set()
        for record in instance:
            event_type = record['type']
            trigger = record['trigger']
            matched_list = match_sublist(token_list,
                                         self.span_to_token(trigger))

            if len(matched_list) == 0:
                logger.warning("[Cannot reconstruct]: %s %s\n" %
                               (trigger, token_list))
                continue

            trigger_offset = None
            for matched in matched_list:
                if matched not in trigger_matched_set:
                    trigger_offset = get_index_tuple(matched)
                    trigger_matched_set.add(matched)
                    break

            # No trigger word, skip the record
            if trigger_offset is None or len(trigger_offset) == 0:
                break

            pred_record = {
                'type': event_type,
                'roles': [],
                'trigger': trigger_offset
            }

            for role_type, text_str in record['roles']:
                matched_list = match_sublist(token_list,
                                             self.span_to_token(text_str))
                # if len(matched_list) == 1:
                #     pred_record['roles'] += [(role_type, get_index_tuple(matched_list[0]))]
                if len(matched_list) == 0:
                    logger.warning("[Cannot reconstruct]: %s %s\n" %
                                   (text_str, token_list))
                else:
                    abs_distances = [
                        abs(match[0] - trigger_offset[0])
                        for match in matched_list
                    ]
                    closest_index = numpy.argmin(abs_distances)
                    pred_record['roles'] += [(
                        role_type,
                        get_index_tuple(matched_list[closest_index]))]

            record_list += [pred_record]
        return record_list
