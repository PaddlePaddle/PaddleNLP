#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from typing import Tuple, List, Dict
from collections import defaultdict, OrderedDict, Counter
import os
import numpy
import logging
import re
import json
from nltk.tree import ParentedTree
from uie.evaluation.constants import (span_start, type_start, type_end,
                                      null_span, offset_map_strategy)
from uie.evaluation.scorer import EntityScorer, RelationScorer, EventScorer

logger = logging.getLogger("__main__")

left_bracket = '【'
right_bracket = '】'
brackets = left_bracket + right_bracket

split_bracket = re.compile(r"<extra_id_\d>")


def proprocessing_graph_record(graph, schema_dict):
    """ Mapping generated spot-asoc result to Entity/Relation/Event
    """
    records = {
        'entity': list(),
        'relation': list(),
        'event': list(),
    }

    entity_dict = OrderedDict()

    # 根据不同任务的 Schema 将不同的 Spot 对应到不同抽取结果： Entity/Event
    # Mapping generated spot result to Entity/Event
    for record in graph['pred_record']:

        if record['type'] in schema_dict['entity'].type_list:
            records['entity'] += [{
                'text': record['spot'],
                'type': record['type']
            }]
            entity_dict[record['spot']] = record['type']

        elif record['type'] in schema_dict['event'].type_list:
            records['event'] += [{
                'trigger': record['spot'],
                'type': record['type'],
                'roles': record['asocs']
            }]

        else:
            print("Type `%s` invalid." % record['type'])

    # 根据不同任务的 Schema 将不同的 Asoc 对应到不同抽取结果： Relation/Argument
    # Mapping generated asoc result to Relation/Argument
    for record in graph['pred_record']:
        if record['type'] in schema_dict['entity'].type_list:
            for role in record['asocs']:
                records['relation'] += [{
                    'type':
                    role[0],
                    'roles': [
                        (record['type'], record['spot']),
                        (entity_dict.get(role[1], record['type']), role[1]),
                    ]
                }]

    if len(entity_dict) > 0:
        for record in records['event']:
            if record['type'] in schema_dict['event'].type_list:
                new_role_list = list()
                for role in record['roles']:
                    if role[1] in entity_dict:
                        new_role_list += [role]
                record['roles'] = new_role_list

    return records


def match_sublist(the_list, to_match):
    """ Find sublist in the whole list

    Args:
        the_list (list(str)): the whole list
            - [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
        to_match (list(str)): the sublist
            - [1, 2]

    Returns:
        list(tuple): matched (start, end) position list
            - [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def check_overlap(x, y):
    """Check two span whether overlap or not

    Args:
        x (Tuple[int, int]): start, end including position of span x
        y (Tuple[int, int]): start, end including position of span y

    x: (3, 4), y: (4, 5) -> True
    x: (3, 3), y: (4, 5) -> False

    Returns:
        bool: two span whether overlap or not
    """
    # x start > y end or y start > x end, no overlap
    if x[0] > y[1] or y[0] > x[1]:
        return False
    else:
        return True


def get_index_tuple(matched: Tuple[int, int]):
    """ Convert start, end (inlcuding) tuple to index list

    Args:
        matched (Tuple[int, int]): start and end position tuple

    (3, 4) -> [3, 4]
    (3, 3) -> [3]

    Returns:
        List[int]: List of index
    """
    return tuple(range(matched[0], matched[1] + 1))


def span_to_token(text, span_to_token_strategy='space'):
    """Convert text span string to token list

    Args:
        text (string): text span string
        span_to_token_strategy (str, optional): Defaults to 'space'.
            - space: split text to tokens using space
            - list: split text to toekns as list

    Raises:
        NotImplementedError: No implemented span_to_token_strategy

    Returns:
        list(str): list of token
    """
    if span_to_token_strategy == 'space':
        return text.split(' ')
    elif span_to_token_strategy == 'list':
        return list(text)
    else:
        raise NotImplementedError(
            f"The span to token strategy {span_to_token_strategy} is not implemented."
        )


class MapConfig:
    """ Config of mapping string to offset
    """

    def __init__(self,
                 map_strategy: str = 'first',
                 de_duplicate: bool = True,
                 span_to_token: str = 'space') -> None:
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
    def load_by_name(config_name):
        offset_map = offset_map_strategy[config_name]
        return MapConfig(
            map_strategy=offset_map['map_strategy'],
            de_duplicate=offset_map['de_duplicate'],
            span_to_token=offset_map['span_to_token'],
        )


class RecordSchema:
    """ Record Schema Class
    type_list: list of spot name
    role_list: list of asoc name
    type_role_dict: the mapping of spot-to-asoc
    """

    def __init__(self, type_list, role_list, type_role_dict):
        self.type_list = type_list
        self.role_list = role_list
        self.type_role_dict = type_role_dict

    def __repr__(self) -> str:
        repr_list = [
            f"Type: {self.type_list}\n", f"Role: {self.role_list}\n",
            f"Map: {self.type_role_dict}"
        ]
        return '\n'.join(repr_list)

    @staticmethod
    def get_empty_schema():
        return RecordSchema(type_list=list(),
                            role_list=list(),
                            type_role_dict=dict())

    @staticmethod
    def read_from_file(filename):
        lines = open(filename, encoding='utf8').readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        return RecordSchema(type_list, role_list, type_role_dict)

    def write_to_file(self, filename):
        with open(filename, 'w', encoding='utf8') as output:
            output.write(json.dumps(self.type_list, ensure_ascii=False) + '\n')
            output.write(json.dumps(self.role_list, ensure_ascii=False) + '\n')
            output.write(
                json.dumps(self.type_role_dict, ensure_ascii=False) + '\n')


def merge_schema(schema_list: List[RecordSchema]):
    """Merge list of schema

    Args:
        schema_list (List[RecordSchema]): list of record schema

    Returns:
        RecordSchema: A merged schema
    """
    type_set = set()
    role_set = set()
    type_role_dict = defaultdict(list)

    for schema in schema_list:

        for type_name in schema.type_list:
            type_set.add(type_name)

        for role_name in schema.role_list:
            role_set.add(role_name)

        for type_name in schema.type_role_dict:
            type_role_dict[type_name] += schema.type_role_dict[type_name]

    for type_name in type_role_dict:
        type_role_dict[type_name] = list(set(type_role_dict[type_name]))

    return RecordSchema(
        type_list=list(type_set),
        role_list=list(role_set),
        type_role_dict=type_role_dict,
    )


class Record:
    """ Record for converting generated string to information record
    """

    def __init__(self, map_config) -> None:
        self._map_config = map_config

    def span_to_token(self, text):
        return span_to_token(
            text, span_to_token_strategy=self._map_config.span_to_token)


class EntityRecord(Record):
    """ Record for converting generated string to information record <type, span>
    """

    @staticmethod
    def to_string(pred_record_list):
        entity_list = list()
        for pred_record in pred_record_list:
            record_type, record_text = pred_record['type'], pred_record['text']
            if record_text == "":
                logger.warning(f"Empty Extraction {pred_record}")
                continue
            entity_list += [(record_type, record_text)]
        return entity_list

    def to_offset(self, instance, tokens):
        map_strategy_dict = {
            'first': self.record_to_offset_first_role,
            'closest': self.record_to_offset_closest_role,
            'longer_first': self.record_to_offset_longer_first,
        }

        if self._map_config.map_strategy in map_strategy_dict:
            map_function = map_strategy_dict[self._map_config.map_strategy]
            return map_function(
                instance=instance,
                token_list=tokens,
            )
        else:
            raise NotImplementedError(
                f"The map strategy {self._map_config.map_strategy} in {self.__class__} is not implemented."
            )

    def record_to_offset_closest_role(
        self,
        instance,
        token_list,
    ):
        """
        Find Role's offset using closest matched with trigger word.
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
            record_type, record_text = pred_record['type'], pred_record['text']
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
            x['length'] = len(x['text'])
        instance.sort(reverse=True, key=lambda x: x['length'])

        for pred_record in instance:
            record_type, record_text = pred_record['type'], pred_record['text']
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
    """ Record for converting generated string to information record
    <type, arg1_type, arg1_span, arg2_type, arg2_span>
    """

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
                token_list=tokens,
            )
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
            if len(relation) != 5 or (self._map_config.de_duplicate
                                      and tuple(relation) in relation_list):
                continue
            relation_list += [tuple(relation)]

        return relation_list

    def record_to_offset_closest_role(self, instance, token_list):
        """
        Find Role's offset using closest matched with trigger word.
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
    """ Record for converting generated string to information record in predicate-arguments
    {
        type: pred_type,
        trigger: predicate_span,
        args: [(arg_type, arg_span), ...]
    }
    """

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
                token_list=tokens,
            )
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
        Find Role's offset using closest matched with trigger word.
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
                if len(matched_list) == 0:
                    logger.warning("[Cannot reconstruct]: %s %s\n" %
                                   (text_str, token_list))
                else:
                    abs_distances = [
                        abs(match[0] - trigger_offset[0])
                        for match in matched_list
                    ]
                    closest_index = numpy.argmin(abs_distances)
                    pred_record['roles'] += [
                        (role_type,
                         get_index_tuple(matched_list[closest_index]))
                    ]

            record_list += [pred_record]
        return record_list


class SEL2Record:
    """ Converting sel expression to information records
    """

    def __init__(self,
                 schema_dict,
                 map_config: MapConfig,
                 tokenizer=None) -> None:
        self._schema_dict = schema_dict
        self._predict_parser = SpotAsocPredictParser(
            record_schema=schema_dict['record'],
            tokenizer=tokenizer,
        )
        self._map_config = map_config
        self._tokenizer = tokenizer

    def __repr__(self) -> str:
        return f"## {self._map_config}"

    def sel2record(self, pred, text, tokens):
        """ Converting sel expression to information records

        Args:
            pred (str): sel expression
            text (str): input text
            tokens (list(str)): token list

        Returns:
            _type_: _description_
        """
        # Parsing generated SEL to String-level Record
        # 将生成的结构表达式解析成 String 级别的 Record
        well_formed_list, counter = self._predict_parser.decode(
            gold_list=[],
            pred_list=[pred],
            text_list=[text],
        )

        # Convert String-level Record to Entity/Relation/Event
        # 将抽取的 Spot-Asoc Record 结构
        # 根据不同的 Schema 转换成 Entity/Relation/Event 结果
        pred_records = proprocessing_graph_record(well_formed_list[0],
                                                  self._schema_dict)

        task_record_map = {
            'entity': EntityRecord,
            'relation': RelationRecord,
            'event': EventRecord,
        }

        parsed_record = defaultdict(dict)
        # Mapping String-level record to Offset-level record
        # 将 String 级别的 Record 回标成 Offset 级别的 Record
        for task in task_record_map:
            record_map = task_record_map[task](map_config=self._map_config, )

            parsed_record[task]['offset'] = record_map.to_offset(
                instance=pred_records.get(task, []),
                tokens=tokens,
            )

            parsed_record[task]['string'] = record_map.to_string(
                pred_records.get(task, []), )
        return parsed_record

    @staticmethod
    def load_schema_dict(schema_folder):
        schema_dict = dict()
        for schema_key in ['record', 'entity', 'relation', 'event']:
            schema_filename = os.path.join(schema_folder,
                                           f'{schema_key}.schema')
            if os.path.exists(schema_filename):
                schema_dict[schema_key] = RecordSchema.read_from_file(
                    schema_filename)
            else:
                logger.warning(f"{schema_filename} is empty, ignore.")
                schema_dict[schema_key] = RecordSchema.get_empty_schema()
        return schema_dict


def fix_unk_from_text(span, text, unk='<unk>', tokenizer=None):
    """Fixing unknown tokens `unk` in the generated expression

    Args:
        span (str): generated span
        text (str): raw text
        unk (str, optional): symbol of unk token
        tokenizer (Tokenizer, optional): the tokenizer

    Returns:
        fixed span
    """
    if tokenizer is not None:
        return fix_unk_from_text_with_tokenizer(span,
                                                text,
                                                unk=unk,
                                                tokenizer=tokenizer)
    else:
        return fix_unk_from_text_without_tokenizer(span, text, unk=unk)


def fix_unk_from_text_without_tokenizer(span, text, unk='<unk>'):
    """
    Find span from the text to fix unk in the generated span
    Example:
    span = "<unk> colo e Bengo"
    text = "Angola International Airport is located at Ícolo e Bengo"
    fixed_span = "Ícolo e Bengo"
    """
    if unk not in span:
        return span

    def clean_wildcard(x):
        sp = ".*?()[]+"
        return re.sub("(" + "|".join([f"\\{s}" for s in sp]) + ")", "\\\\\g<1>",
                      x)

    match = r'\s*[^，？。\s]+\s*'.join(
        [clean_wildcard(item.strip()) for item in span.split(unk)])

    if len(match) > 100:
        # Too long regular expression may lead re problem
        return span

    result = re.search(match, text)

    if not result:
        return span
    return result.group().strip()


def fix_unk_from_text_with_tokenizer(span, text, tokenizer, unk='<unk>'):
    unk_id = tokenizer.vocab.to_indices(unk)
    tokenized_span = tokenizer.encode(span,
                                      add_special_tokens=False,
                                      return_token_type_ids=None)['input_ids']
    tokenized_text = tokenizer.encode(text,
                                      add_special_tokens=False,
                                      return_token_type_ids=None)['input_ids']

    matched = match_sublist(tokenized_text, tokenized_span)
    if len(matched) == 0:
        return span

    if tokenized_span[0] == unk_id and matched[0][0] > 0:
        previous_token = [tokenized_text[matched[0][0] - 1]]
        pre_strip = tokenizer.vocab.to_tokens(previous_token[0])
    else:
        previous_token = []
        pre_strip = ""

    if tokenized_span[-1] == unk_id and matched[0][1] < len(tokenized_text) - 1:
        next_token = [tokenized_text[matched[0][1] + 1]]
        next_strip = tokenizer.vocab.to_tokens(next_token[0])
    else:
        next_token = []
        next_strip = ""

    extend_span = tokenized_span
    if len(previous_token) > 0:
        extend_span = previous_token + extend_span
    if len(next_token) > 0:
        extend_span = extend_span + next_token

    extend_span = tokenizer.decode(extend_span)
    fixed_span = fix_unk_from_text_without_tokenizer(extend_span, text, unk)
    return fixed_span.rstrip(next_strip).lstrip(pre_strip)


def add_space(text):
    """
    add space between special token
    """
    new_text_list = list()
    for item in zip(split_bracket.findall(text), split_bracket.split(text)[1:]):
        new_text_list += item
    return ' '.join(new_text_list)


def convert_bracket(text):
    text = add_space(text)
    for start in [type_start]:
        text = text.replace(start, left_bracket)
    for end in [type_end]:
        text = text.replace(end, right_bracket)
    return text


def find_bracket_num(tree_str):
    """
    Count Bracket Number (num_left - num_right), 0 indicates num_left = num_right
    """
    count = 0
    for char in tree_str:
        if char == left_bracket:
            count += 1
        elif char == right_bracket:
            count -= 1
        else:
            pass
    return count


def check_well_form(tree_str):
    return find_bracket_num(tree_str) == 0


def clean_text(tree_str):
    count = 0
    sum_count = 0

    tree_str_list = tree_str.split()

    for index, char in enumerate(tree_str_list):
        if char == left_bracket:
            count += 1
            sum_count += 1
        elif char == right_bracket:
            count -= 1
            sum_count += 1
        else:
            pass
        if count == 0 and sum_count > 0:
            return ' '.join(tree_str_list[:index + 1])
    return ' '.join(tree_str_list)


def resplit_label_span(label, span, split_symbol=span_start):
    label_span = label + ' ' + span

    if split_symbol in label_span:
        splited_label_span = label_span.split(split_symbol)
        if len(splited_label_span) == 2:
            return splited_label_span[0].strip(), splited_label_span[1].strip()

    return label, span


def add_bracket(tree_str):
    """add right bracket to fix ill-formed expression
    """
    tree_str_list = tree_str.split()
    bracket_num = find_bracket_num(tree_str_list)
    tree_str_list += [right_bracket] * bracket_num
    return ' '.join(tree_str_list)


def get_tree_str(tree):
    """get str from sel tree
    """
    str_list = list()
    for element in tree:
        if isinstance(element, str):
            str_list += [element]
    return ' '.join(str_list)


def rewrite_label_span(label, span, label_set=None, text=None, tokenizer=None):

    # Invalid Type
    if label_set and label not in label_set:
        logger.debug('Invalid Label: %s' % label)
        return None, None

    # Fix unk using Text
    if text is not None and '<unk>' in span:
        span = fix_unk_from_text(span, text, unk='<unk>', tokenizer=tokenizer)

    # Invalid Text Span
    if text is not None and span not in text:
        logger.debug('Invalid Text Span: %s\n%s\n' % (span, text))
        return None, None

    return label, span


def convert_spot_asoc(spot_asoc_instance, structure_maker):
    """Convert spot asoc instance to target string
    """
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['label'],
            structure_maker.target_span_start,
            spot['span'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_label,
                structure_maker.target_span_start,
                asoc_span,
                structure_maker.span_end,
            ]
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [
            ' '.join([
                structure_maker.record_start,
                ' '.join(spot_str_rep),
                structure_maker.record_end,
            ])
        ]
    target_text = ' '.join([
        structure_maker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_maker.sent_end,
    ])
    return target_text


class SpotAsocPredictParser:
    """ Parser for converting generated sel to extraction record
    """

    def __init__(self, record_schema: RecordSchema = None, tokenizer=None):
        self.spot_set = set(record_schema.type_list) if record_schema else None
        self.asoc_set = set(record_schema.role_list) if record_schema else None
        self.tokenizer = tokenizer

    def decode(
        self,
        gold_list,
        pred_list,
        text_list=None,
        raw_list=None,
    ) -> Tuple[List[Dict], Counter]:
        counter = Counter()
        well_formed_list = []

        if gold_list is None or len(gold_list) == 0:
            gold_list = ["%s%s" % (type_start, type_end)] * len(pred_list)

        if text_list is None:
            text_list = [None] * len(pred_list)

        if raw_list is None:
            raw_list = [None] * len(pred_list)

        for gold, pred, text, raw_data in zip(gold_list, pred_list, text_list,
                                              raw_list):
            gold = convert_bracket(gold)
            pred = convert_bracket(pred)

            pred = clean_text(pred)

            try:
                gold_tree = ParentedTree.fromstring(gold, brackets=brackets)
            except ValueError:
                logger.warning(f"Ill gold: {gold}")
                logger.warning(f"Fix gold: {add_bracket(gold)}")
                gold_tree = ParentedTree.fromstring(add_bracket(gold),
                                                    brackets=brackets)
                counter.update(['gold_tree add_bracket'])

            instance = {
                'gold': gold,
                'pred': pred,
                'gold_tree': gold_tree,
                'text': text,
                'raw_data': raw_data
            }

            counter.update(['gold_tree' for _ in gold_tree])

            _, _, instance['gold_record'] = self.get_record_list(
                sel_tree=instance["gold_tree"], text=instance['text'])

            try:
                if not check_well_form(pred):
                    pred = add_bracket(pred)
                    counter.update(['fixed'])

                pred_tree = ParentedTree.fromstring(pred, brackets=brackets)
                counter.update(['pred_tree' for _ in pred_tree])

                instance['pred_tree'] = pred_tree
                counter.update(['well-formed'])

            except ValueError:
                counter.update(['ill-formed'])
                logger.debug(f'ill-formed: {pred}')
                instance['pred_tree'] = ParentedTree.fromstring(
                    left_bracket + right_bracket, brackets=brackets)

            _, _, instance['pred_record'] = self.get_record_list(
                sel_tree=instance["pred_tree"], text=instance['text'])

            well_formed_list += [instance]

        return well_formed_list, counter

    def get_record_list(self, sel_tree, text=None):
        """ Convert single sel expression to extraction records

        Args:
            sel_tree (Tree): sel tree
            text (str, optional): _description_. Defaults to None.

        Returns:
            spot_list: list of (spot_type: str, spot_span: str)
            asoc_list: list of (spot_type: str, asoc_label: str, asoc_text: str)
            record_list: list of {'asocs': list(), 'type': spot_type, 'spot': spot_text}
        """
        spot_list = list()
        asoc_list = list()
        record_list = list()

        for spot_tree in sel_tree:

            # Drop incomplete tree
            if isinstance(spot_tree, str) or len(spot_tree) == 0:
                continue

            spot_type = spot_tree.label()
            spot_text = get_tree_str(spot_tree)
            spot_type, spot_text = resplit_label_span(spot_type, spot_text)
            spot_type, spot_text = rewrite_label_span(
                label=spot_type,
                span=spot_text,
                label_set=self.spot_set,
                text=text,
                tokenizer=self.tokenizer,
            )

            # Drop empty generated span
            if spot_text is None or spot_text == null_span:
                continue
            # Drop empty generated type
            if spot_type is None:
                continue
            # Drop invalid spot type
            if self.spot_set is not None and spot_type not in self.spot_set:
                continue

            record = {'asocs': list(), 'type': spot_type, 'spot': spot_text}

            for asoc_tree in spot_tree:
                if isinstance(asoc_tree, str) or len(asoc_tree) < 1:
                    continue

                asoc_label = asoc_tree.label()
                asoc_text = get_tree_str(asoc_tree)
                asoc_label, asoc_text = resplit_label_span(
                    asoc_label, asoc_text)
                asoc_label, asoc_text = rewrite_label_span(
                    label=asoc_label,
                    span=asoc_text,
                    label_set=self.asoc_set,
                    text=text,
                    tokenizer=self.tokenizer,
                )

                # Drop empty generated span
                if asoc_text is None or asoc_text == null_span:
                    continue
                # Drop empty generated type
                if asoc_label is None:
                    continue
                # Drop invalid asoc type
                if self.asoc_set is not None and asoc_label not in self.asoc_set:
                    continue

                asoc_list += [(spot_type, asoc_label, asoc_text)]
                record['asocs'] += [(asoc_label, asoc_text)]

            spot_list += [(spot_type, spot_text)]
            record_list += [record]

        return spot_list, asoc_list, record_list


def evaluate_extraction_results(gold_instances,
                                pred_instances,
                                eval_match_mode='normal'):
    task_scorer_dict = {
        'entity': EntityScorer,
        'relation': RelationScorer,
        'event': EventScorer
    }
    # Score Record
    results = dict()
    for task, scorer in task_scorer_dict.items():

        gold_list = [x[task] for x in gold_instances]
        pred_list = [x[task] for x in pred_instances]

        gold_instance_list = scorer.load_gold_list(gold_list)
        pred_instance_list = scorer.load_pred_list(pred_list)
        sub_results = scorer.eval_instance_list(
            gold_instance_list=gold_instance_list,
            pred_instance_list=pred_instance_list,
            verbose=False,
            match_mode=eval_match_mode,
        )
        results.update(sub_results)
    return results
