#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict
import json
import os
import shutil
from typing import List, Dict

from uie.extraction.record_schema import RecordSchema, merge_schema


def annonote_graph(entities: List[Dict]=[],
                   relations: List[Dict]=[],
                   events: List[Dict]=[]):
    """Convert Entity Relation Event to Spot-Assocation Graph

    Args:
        tokens (List[str]): Token List
        entities (List[Entity], optional): Entity List. Defaults to [].
        relations (List[Relation], optional): Relation List. Defaults to [].
        events (List[Event], optional): Event List. Defaults to [].

    Returns:
        set: Set of Spot
        set: Set of Asoc
    """
    spot_dict = dict()
    asoc_dict = defaultdict(list)

    def add_spot(spot):
        spot_key = (tuple(spot['offset']), spot['type'])
        spot_dict[spot_key] = spot

    def add_asoc(spot, asoc, tail):
        spot_key = (tuple(spot['offset']), spot['type'])
        asoc_dict[spot_key] += [(tuple(tail['offset']), tail['text'], asoc)]

    for entity in entities:
        add_spot(spot=entity)

    for relation in relations:
        add_spot(spot=relation['args'][0])
        add_asoc(
            spot=relation['args'][0],
            asoc=relation['type'],
            tail=relation['args'][1])

    for event in events:
        add_spot(spot=event)
        for argument in event['args']:
            add_asoc(spot=event, asoc=argument['type'], tail=argument)

    spot_asoc_instance = list()
    for spot_key in sorted(spot_dict.keys()):
        offset, label = spot_key

        if len(spot_dict[spot_key]['offset']) == 0:
            continue

        spot_instance = {
            'span': spot_dict[spot_key]['text'],
            'label': label,
            'asoc': list(),
        }
        for tail_offset, tail_text, asoc in sorted(asoc_dict.get(spot_key, [])):
            if len(tail_offset) == 0:
                continue

            spot_instance['asoc'] += [(asoc, tail_text)]
        spot_asoc_instance += [spot_instance]

    spot_labels = set([label for _, label in spot_dict.keys()])
    asoc_labels = set()
    for _, asoc_list in asoc_dict.items():
        for _, _, asoc in asoc_list:
            asoc_labels.add(asoc)
    return spot_labels, asoc_labels, spot_asoc_instance


def add_spot_asoc_to_multitask_file(filename, task_schema_dict):
    instances = [json.loads(line) for line in open(filename)]
    print(f'Add spot asoc to Multi {filename} ...')
    with open(filename, 'w') as output:
        for instance in instances:
            try:
                _, _, spot_asoc_instance = annonote_graph(
                    entities=instance['entity'],
                    relations=instance['relation'],
                    events=instance['event'], )
            except:
                print(instance)
                exit(1)
            # 将信息结构转换成 Spot Asoc 形式
            instance['spot_asoc'] = spot_asoc_instance
            # 添加任务中所有的 Spot 类别
            instance['spot'] = task_schema_dict[instance['schema']].type_list
            # 添加任务中所有的 Asoc 类别
            instance['asoc'] = task_schema_dict[instance['schema']].role_list
            output.write(json.dumps(instance, ensure_ascii=False) + '\n')


def add_spot_asoc_to_single_file(filename):
    instances = [json.loads(line) for line in open(filename)]
    print(f'Add spot asoc to {filename} ...')
    with open(filename, 'w') as output:
        for instance in instances:
            try:
                spots, asocs, spot_asoc_instance = annonote_graph(
                    entities=instance['entity'],
                    relations=instance['relation'],
                    events=instance['event'], )
            except:
                print(instance)
            # 将信息结构转换成 Spot Asoc 形式
            instance['spot_asoc'] = spot_asoc_instance
            # 添加该实例中存在的 Spot 类别
            instance['spot'] = list(spots)
            # 添加该实例中存在的 Asoc 类别
            instance['asoc'] = list(asocs)
            output.write(json.dumps(instance, ensure_ascii=False) + '\n')


def convert_duuie_to_spotasoc(data_folder):
    task_schema_dict = dict()

    for task_folder in os.listdir(data_folder):
        if not os.path.isdir(os.path.join(data_folder, task_folder)):
            continue
        print(f'Add spot asoc to {task_folder} ...')
        # 读取单任务的 Schema
        task_schema_file = os.path.join(data_folder, task_folder,
                                        'record.schema')

        # 向单任务数据中添加 Spot Asoc 标注
        add_spot_asoc_to_single_file(
            os.path.join(data_folder, task_folder, 'train.json'))
        add_spot_asoc_to_single_file(
            os.path.join(data_folder, task_folder, 'val.json'))
        record_schema = RecordSchema.read_from_file(task_schema_file)
        task_schema_dict[task_folder] = record_schema

    # 融合不同任务的 Schema
    multi_schema = merge_schema(task_schema_dict.values())
    train_file = os.path.join(data_folder, 'train.json')
    val_file = os.path.join(data_folder, 'val.json')

    # 向多任务数据中添加 Spot Asoc 标注
    print(f'Add spot asoc to Multi Task ...')
    add_spot_asoc_to_multitask_file(train_file, task_schema_dict)
    add_spot_asoc_to_multitask_file(val_file, task_schema_dict)
    multi_schema.write_to_file(os.path.join(data_folder, 'record.schema'))


if __name__ == "__main__":
    shutil.copytree('data/duuie_entity', 'data/duuie_entity_pre')
    convert_duuie_to_spotasoc('data/duuie_entity_pre')
