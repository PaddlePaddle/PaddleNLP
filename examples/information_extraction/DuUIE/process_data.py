#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import copy
from typing import List, Dict
from collections import defaultdict
import yaml
import json
import os
from uie.evaluation.sel2record import RecordSchema, merge_schema


def load_definition_schema_file(filename):
    """Load schema file in Yaml
    读取 YAML 定义的 Schema 文件
    """
    return yaml.load(open(filename, encoding='utf8'), Loader=yaml.FullLoader)


def load_jsonlines_file(filename):
    """Load Data file in JSONLINE
    读取 JSONLINE 文件
    """
    return [json.loads(line) for line in open(filename, encoding='utf8')]


def convert_entity_schema(entity_schema):
    """ Convert entity schmea to record schema
    """
    spots = list()
    asocs = list()
    spot_asoc_map = dict()
    for entity in entity_schema:
        spots += [entity]
        spot_asoc_map[entity] = list()
    return spots, asocs, spot_asoc_map


def convert_entity_relation_schema(entity_schema, relation_schema):
    """ Convert entity and relation chmea to record schema
    """
    spots = list()
    asocs = list()
    spot_asoc_map = dict()
    for entity in entity_schema:
        spots += [entity]
        spot_asoc_map[entity] = list()
    for relation in relation_schema:
        asocs += [relation]
        arg1_type = relation_schema[relation]['主体']
        if arg1_type not in spots:
            spots += [arg1_type]
            spot_asoc_map[arg1_type] = list()
        spot_asoc_map[arg1_type] += [relation]
    return spots, asocs, spot_asoc_map


def convert_event_schema(schema):
    """ Convert event schmea to record schema
    """
    spots = list()
    asocs = set()
    spot_asoc_map = dict()
    for event_type, definition in schema.items():
        spots += [event_type]
        spot_asoc_map[event_type] = list()
        for arg in definition['参数']:
            asocs.add(arg)
            spot_asoc_map[event_type] += [arg]
    return spots, list(asocs), spot_asoc_map


def dump_schema(output_folder, schema_dict):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for schema_name, schema in schema_dict.items():
        schema_file = f"{output_folder}/{schema_name}.schema"
        with open(schema_file, 'w', encoding='utf8') as output:
            for element in schema:
                output.write(json.dumps(element, ensure_ascii=False) + '\n')


def main_entity_relation(schema_file, schema_name, instances, output_folder):
    schema = yaml.load(open(schema_file, encoding='utf8'),
                       Loader=yaml.FullLoader)
    entity_schema = convert_entity_schema(schema.get('实体', {}))
    relation_schema = convert_entity_relation_schema(schema.get('实体', {}),
                                                     schema.get('关系', {}))
    event_schema = convert_event_schema({})
    dump_schema(output_folder=output_folder,
                schema_dict={
                    'entity': entity_schema,
                    'relation': relation_schema,
                    'event': event_schema,
                    'record': relation_schema,
                })

    with open(f"{output_folder}/test.json", 'w', encoding='utf8') as output:
        for instance in instances:
            if instance['schema'] == schema_name:
                output.write(json.dumps(instance, ensure_ascii=False) + '\n')
    return schema_name


def main_event(schema_file, schema_name, instances, output_folder):
    schema = yaml.load(open(schema_file, encoding='utf8'),
                       Loader=yaml.FullLoader)
    event_schema = convert_event_schema(schema.get('事件', {}))
    dump_schema(output_folder=output_folder,
                schema_dict={
                    'entity': [[], [], {}],
                    'relation': [[], [], {}],
                    'event': event_schema,
                    'record': event_schema,
                })

    with open(f"{output_folder}/test.json", 'w', encoding='utf8') as output:
        for instance in instances:
            if instance['schema'] == schema_name:
                output.write(json.dumps(instance, ensure_ascii=False) + '\n')
    return schema_name


def main_seprate_event(schema_file, schema_name, instances, output_folder):
    """ Prediction tasks are separated by event types
    按照事件类别分离预测任务生成抽取的 Schema
    """
    valid_instances = list()

    for instance in instances:
        if schema_name == instance['schema']:
            valid_instances += [instance]

    schema = yaml.load(open(schema_file, encoding='utf8'),
                       Loader=yaml.FullLoader)
    _, _, event_map = convert_event_schema(schema.get('事件', {}))

    for event in event_map:
        subevent_output_folder = f"{output_folder}_{event}"
        dump_schema(output_folder=subevent_output_folder,
                    schema_dict={
                        'entity': [[], [], {}],
                        'relation': [[], [], {}],
                        'event': [[event], event_map[event], {
                            event: event_map[event]
                        }],
                        'record': [[event], event_map[event], {
                            event: event_map[event]
                        }],
                    })

        with open(f"{subevent_output_folder}/test.json", 'w',
                  encoding='utf8') as output:
            for instance in valid_instances:
                output.write(json.dumps(instance, ensure_ascii=False) + '\n')
    return event_map.keys()


# 将关系抽取结果转换到提交格式
def convert_relation(relation):
    return {
        'type':
        relation[0],
        'args': [
            {
                'type': relation[1],
                'text': relation[2]
            },
            {
                'type': relation[3],
                'text': relation[4]
            },
        ]
    }


# 将实体抽取结果转换到提交格式
def convert_entity(entity):
    return {
        'type': entity[0],
        'text': entity[1],
    }


def convert_event(event):
    return {
        'type':
        event['type'],
        'text':
        event['trigger'],
        'args': [{
            'type': role_type,
            'text': arg
        } for role_type, arg in event['roles']]
    }


def merge_pred_text_file(text_filename, pred_filename):
    """ Merge extracted result
    基于实例编号合并抽取结果
    """

    # 读取原始文件中的数据，用于获取 ID
    test_instances = load_jsonlines_file(text_filename)
    # 读取抽取结果的预测文件
    pred_instances = load_jsonlines_file(pred_filename)

    assert len(test_instances) == len(pred_instances)

    to_sumbit_instances = dict()
    for test_instance, pred_instance in zip(test_instances, pred_instances):

        # 获取抽取结果中的字符串结果
        entity_list = pred_instance['entity'].get('string', [])
        relation_list = pred_instance['relation'].get('string', [])
        event_list = pred_instance['event'].get('string', [])

        # 将抽取结果转换为提交的数据格式
        to_sumbit_instance = {
            'id': test_instance['id'],
            'entity': [convert_entity(entity) for entity in entity_list],
            'relation':
            [convert_relation(relation) for relation in relation_list],
            'event': [convert_event(event) for event in event_list]
        }
        to_sumbit_instances[test_instance['id']] = to_sumbit_instance

    return to_sumbit_instances


def split_test(options):
    test_file = options.data_file
    schema_folder = options.schema_folder
    output_folder = options.output_folder
    instances = [json.loads(line) for line in open(test_file, encoding='utf8')]
    main_entity_relation(os.path.join(schema_folder, "人生信息.yaml"), "人生信息",
                         instances, os.path.join(output_folder, "人生信息"))
    main_entity_relation(os.path.join(schema_folder, "机构信息.yaml"), "机构信息",
                         instances, os.path.join(output_folder, "机构信息"))
    main_entity_relation(os.path.join(schema_folder, "影视情感.yaml"), "影视情感",
                         instances, os.path.join(output_folder, "影视情感"))
    main_event(os.path.join(schema_folder, "灾害意外.yaml"), "灾害意外", instances,
               os.path.join(output_folder, "灾害意外"))
    main_event(os.path.join(schema_folder, "体育竞赛.yaml"), "体育竞赛", instances,
               os.path.join(output_folder, "体育竞赛"))
    main_seprate_event(os.path.join(schema_folder, "金融信息.yaml"), "金融信息",
                       instances, os.path.join(output_folder, "金融信息"))


def merge_test(options):
    """ Merge predicted result from trained model
    将预测文件夹中的预测结果进行合并
    """
    output_folder = options.pred_folder
    submit_filename = options.submit

    to_sumbit_instances = dict()
    for schema in os.listdir(output_folder):
        test_filename = os.path.join(output_folder, schema, "test.json")
        pred_filename = os.path.join(output_folder, schema, "pred.json")
        sub_to_sumbit_instances = merge_pred_text_file(
            text_filename=test_filename,
            pred_filename=pred_filename,
        )
        print(
            f"Merge {schema} with {len(sub_to_sumbit_instances)} instances ...")
        for instance_id, instance in sub_to_sumbit_instances.items():
            if instance_id in to_sumbit_instances:
                to_sumbit_instances[instance_id]['entity'] += instance.get(
                    'entity', [])
                to_sumbit_instances[instance_id]['relation'] += instance.get(
                    'relation', [])
                to_sumbit_instances[instance_id]['event'] += instance.get(
                    'event', [])
            else:
                to_sumbit_instances[instance_id] = instance

    print(f"To submit instances number: {len(to_sumbit_instances)}")
    with open(submit_filename, 'w', encoding='utf8') as output:
        for instance in to_sumbit_instances.values():
            output.write(json.dumps(instance, ensure_ascii=False) + '\n')


def annonote_graph(entities: List[Dict] = [],
                   relations: List[Dict] = [],
                   events: List[Dict] = []):
    """Convert Entity Relation Event to Spot-Assocation Graph
    将实体、关系和事件的标注信息转换成需要生成的 Spot-Assocation 结构

    Args:
        tokens (List[str]): Token List
        entities (List[Entity], optional): Entity List. Defaults to [].
        relations (List[Relation], optional): Relation List. Defaults to [].
        events (List[Event], optional): Event List. Defaults to [].

    Returns:
        set: Set of Spot
        set: Set of Asoc
        list: Instance of Spot-Asoc
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
        add_asoc(spot=relation['args'][0],
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


def add_spot_asoc_to_single_file(filename):
    instances = [json.loads(line) for line in open(filename, encoding='utf8')]
    print(f'Add spot asoc to {filename} ...')
    with open(filename, 'w', encoding='utf8') as output:
        for instance in instances:
            spots, asocs, spot_asoc_instance = annonote_graph(
                entities=instance['entity'],
                relations=instance['relation'],
                events=instance['event'],
            )
            # 将信息结构转换成 Spot Asoc 形式
            instance['spot_asoc'] = spot_asoc_instance
            # 添加该实例中存在的 Spot 类别
            instance['spot'] = list(spots)
            # 添加该实例中存在的 Asoc 类别
            instance['asoc'] = list(asocs)
            output.write(json.dumps(instance, ensure_ascii=False) + '\n')


def convert_duuie_to_spotasoc(data_folder, ignore_datasets):

    schema_list = list()

    for task_folder in os.listdir(data_folder):
        if task_folder in ignore_datasets:
            continue
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

        schema_list += [record_schema]

        for line in open(os.path.join(data_folder, task_folder, 'train.json'),
                         encoding='utf8'):
            new_instance = json.loads(line)
            # 添加任务中所有的 Spot 类别
            new_instance['spot'] = record_schema.type_list
            # 添加任务中所有的 Asoc 类别
            new_instance['asoc'] = record_schema.role_list

        for line in open(os.path.join(data_folder, task_folder, 'val.json'),
                         encoding='utf8'):
            new_instance = json.loads(line)
            # 添加任务中所有的 Spot 类别
            new_instance['spot'] = record_schema.type_list
            # 添加任务中所有的 Asoc 类别
            new_instance['asoc'] = record_schema.role_list

    # 融合不同任务的 Schema
    multi_schema = merge_schema(schema_list)
    multi_schema.write_to_file(os.path.join(data_folder, 'record.schema'))


def dump_instances(instances, output_filename):
    with open(output_filename, 'w', encoding='utf8') as output:
        for instance in instances:
            output.write(json.dumps(instance, ensure_ascii=False) + '\n')


def dump_event_schema(event_map, output_folder):
    role_list = list()
    for roles in event_map.values():
        role_list += roles['参数']
    rols_list = list(set(role_list))
    type_list = list(event_map.keys())
    type_role_map = {
        event_type: list(event_map[event_type]['参数'].keys())
        for event_type in event_map
    }
    dump_schema(output_folder=output_folder,
                schema_dict={
                    'entity': [[], [], {}],
                    'relation': [[], [], {}],
                    'event': [type_list, rols_list, type_role_map],
                    'record': [type_list, rols_list, type_role_map],
                })


def filter_event_in_instance(instances, required_event_types):
    """Filter events in the instance, keep event mentions with `required_event_types`
    过滤实例中的事件，只保留需要的事件类别的事件标注
    """
    import copy
    new_instances = list()
    for instance in instances:
        new_instance = copy.deepcopy(instance)
        new_instance['event'] = list(
            filter(lambda x: x['type'] in required_event_types,
                   new_instance['event']))
        new_instances += [new_instance]
    return new_instances


def filter_event(data_folder, event_types, output_folder):
    """ Keep event with `event_types` in `data_folder` save to `output_folder`
    过滤 `data_folder` 中的事件，只保留 `event_types` 类型事件保存到 `output_folder` """
    dump_event_schema(event_types, output_folder)
    for split in ['train', 'val']:
        filename = os.path.join(data_folder, f"{split}.json")
        instances = [
            json.loads(line.strip()) for line in open(filename, encoding='utf8')
        ]
        new_instances = filter_event_in_instance(
            instances, required_event_types=event_types)
        dump_instances(new_instances,
                       os.path.join(output_folder, f"{split}.json"))


def preprocess_event(data_folder, schema_folder):
    """ Preprocessing event dataset for CCKS 2022
    针对 CCKS 2022 竞赛数据进行预处理
    """
    # Filter event annotation in raw data, only keep the required event in CCKS 2022
    # 对事件数据进行预处理，过滤除 `灾害意外` 和 `体育竞赛` 外的事件标注
    for schema in ['灾害意外', '体育竞赛']:
        print(f'Building {schema} dataset ...')
        duee_folder = os.path.join(data_folder, 'DUEE')
        schema_file = os.path.join(schema_folder, f'{schema}.yaml')
        output_folder = os.path.join(data_folder, schema)
        schema = load_definition_schema_file(schema_file)
        filter_event(
            data_folder=duee_folder,
            event_types=schema['事件'],
            output_folder=output_folder,
        )

    for schema in ['金融信息']:
        print(f'Building {schema} dataset ...')
        duee_fin_folder = os.path.join(data_folder, 'DUEE_FIN_LITE')
        schema_file = os.path.join(schema_folder, f'{schema}.yaml')
        output_folder = os.path.join(data_folder, schema)
        schema = load_definition_schema_file(schema_file)
        # 依据不同事件类别将多事件抽取分割成多个单事件类型抽取
        # Separate multi-type extraction to multiple single-type extraction
        for event_type in schema['事件']:
            filter_event(
                data_folder=duee_fin_folder,
                event_types={event_type: schema['事件'][event_type]},
                output_folder=output_folder + '_' + event_type,
            )


def merge_instance(instance_list):
    """Merge instances with same text but different annotation
    合并文本相同标记不同的实例
    """

    def all_equal(_x):
        for __x in _x:
            if __x != _x[0]:
                return False
        return True

    def entity_key(_x):
        return (tuple(_x['offset']), _x['type'])

    def relation_key(_x):
        return (
            tuple(_x['type']),
            tuple(_x['args'][0]['offset']),
            _x['args'][0]['type'],
            tuple(_x['args'][1]['offset']),
            _x['args'][1]['type'],
        )

    def event_key(_x):
        return (tuple(_x['offset']), _x['type'])

    assert all_equal([x['text'] for x in instance_list])

    element_dict = {
        'entity': dict(),
        'relation': dict(),
        'event': dict(),
    }
    instance_id_list = list()
    for x in instance_list:
        instance_id_list += [x['id']]
        for entity in x.get('entity', list()):
            element_dict['entity'][entity_key(entity)] = entity
        for relation in x.get('relation', list()):
            element_dict['relation'][relation_key(relation)] = relation
        for event in x.get('event', list()):
            element_dict['event'][event_key(event)] = event

    return {
        'id': '-'.join(instance_id_list),
        'text': instance_list[0]['text'],
        'tokens': instance_list[0]['tokens'],
        'entity': list(element_dict['entity'].values()),
        'relation': list(element_dict['relation'].values()),
        'event': list(element_dict['event'].values())
    }


def preprocess_duie(data_folder):
    life_folder = os.path.join(data_folder, 'DUIE_LIFE_SPO')
    org_folder = os.path.join(data_folder, 'DUIE_ORG_SPO')
    life_train_instances = load_jsonlines_file(f"{life_folder}/train.json")
    org_train_instances = load_jsonlines_file(f"{org_folder}/train.json")
    life_relation = RecordSchema.read_from_file(
        f"{life_folder}/record.schema").role_list
    org_relation = RecordSchema.read_from_file(
        f"{org_folder}/record.schema").role_list

    instance_dict = defaultdict(list)
    for instance in life_train_instances + org_train_instances:
        instance_dict[instance['text']] += [instance]

    for text in instance_dict:
        instance_dict[text] = merge_instance(instance_dict[text])

    with open(f"{life_folder}/train.json", 'w') as output:
        for instance in instance_dict.values():
            new_instance = copy.deepcopy(instance)
            new_instance['relation'] = list(
                filter(lambda x: x['type'] in life_relation,
                       instance['relation']))
            output.write(json.dumps(new_instance) + '\n')

    with open(f"{org_folder}/train.json", 'w') as output:
        for instance in instance_dict.values():
            new_instance = copy.deepcopy(instance)
            new_instance['relation'] = list(
                filter(lambda x: x['type'] in org_relation,
                       instance['relation']))
            output.write(json.dumps(new_instance) + '\n')


def preprocess(options):
    """ Preprocessing event dataset for CCKS 2022
    针对 CCKS 2022 竞赛数据进行预处理
    """
    import shutil
    shutil.rmtree(options.output_folder) if os.path.exists(
        options.output_folder) else None
    shutil.copytree(options.train_data, options.output_folder)

    preprocess_duie(data_folder=options.output_folder)
    preprocess_event(data_folder=options.output_folder,
                     schema_folder=options.schema_folder)
    convert_duuie_to_spotasoc(data_folder=options.output_folder,
                              ignore_datasets=options.ignore_datasets)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        help='Data preprocessing scripts for CCKS 2022')

    parser_t = subparsers.add_parser('preprocess', help='Data preprocessing')
    parser_t.add_argument('--train_data',
                          default='data/duuie',
                          help='Path for DuUIE data folder')
    parser_t.add_argument('--output_folder',
                          default='data/duuie_pre',
                          help='Path for Preprocessed DuUIE data folder')
    parser_t.add_argument('--ignore_datasets',
                          default=['DUEE', 'DUEE_FIN_LITE'],
                          nargs='+',
                          help='Ignore dataset in `output_folder` for training')
    parser_t.add_argument('--schema_folder',
                          default='data/seen_schema',
                          help='Path for seen schema folder')
    parser_t.set_defaults(func=preprocess)

    parser_a = subparsers.add_parser(
        'split-test', help='Split test file with schema for prediction')
    parser_a.add_argument('--data_file',
                          default='data/duuie_test_a.json',
                          help='Path for DuUIE data file')
    parser_a.add_argument('--output_folder',
                          default='data/duuie_test_a',
                          help='Path for DuUIE predicted folder')
    parser_a.add_argument('--schema_folder',
                          default='data/seen_schema',
                          help='Path for seen schema folder')
    parser_a.set_defaults(func=split_test)

    parser_b = subparsers.add_parser(
        'merge-test', help='Merge predicted result for submission')
    parser_b.add_argument('--data_file',
                          default='data/duuie_test_a.json',
                          help='Path for DuUIE data file')
    parser_b.add_argument('--pred_folder',
                          default='data/duuie_test_a',
                          help='Path for DuUIE predicted folder')
    parser_b.add_argument('--submit',
                          default='submit.txt',
                          help='Path for output submission file')
    parser_b.set_defaults(func=merge_test)

    options = parser.parse_args()
    options.func(options)
