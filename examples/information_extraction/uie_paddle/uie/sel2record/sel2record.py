#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict, OrderedDict
import os
from uie.extraction.record_schema import RecordSchema
from uie.extraction.predict_parser import get_predict_parser
from uie.sel2record.record import EntityRecord, MapConfig, RelationRecord, EventRecord
import logging

logger = logging.getLogger("__main__")

task_record_map = {
    'entity': EntityRecord,
    'relation': RelationRecord,
    'event': EventRecord,
}


def proprocessing_graph_record(graph, schema_dict):
    """将抽取的Spot-Asoc结构，根据不同的 Schema 转换成 Entity/Relation/Event 结果

    Args:
        graph ([type]): [description]
        schema_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    records = {
        'entity': list(),
        'relation': list(),
        'event': list(),
    }

    entity_dict = OrderedDict()

    # 根据不同任务的 Schema 将不同的 Spot 对应到不同抽取结果： Entity/Event
    for record in graph['pred_record']:

        if record['type'] in schema_dict['entity'].type_list:
            records['entity'] += [{
                'trigger': record['trigger'],
                'type': record['type']
            }]
            entity_dict[record['trigger']] = record['type']

        elif record['type'] in schema_dict['event'].type_list:
            records['event'] += [record]

        else:
            print("Type `%s` invalid." % record['type'])

    # 根据不同任务的 Schema 将不同的 Asoc 对应到不同抽取结果： Relation/Argument
    for record in graph['pred_record']:
        if record['type'] in schema_dict['entity'].type_list:
            for role in record['roles']:
                records['relation'] += [{
                    'type': role[0],
                    'roles': [
                        (record['type'], record['trigger']),
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


class SEL2Record:
    def __init__(self,
                 schema_dict,
                 decoding_schema,
                 map_config: MapConfig,
                 tokenizer=None) -> None:
        self._schema_dict = schema_dict
        self._predict_parser = get_predict_parser(
            decoding_schema=decoding_schema,
            label_constraint=schema_dict['record'],
            tokenizer=tokenizer, )
        self._map_config = map_config

    def __repr__(self) -> str:
        return f"## {self._map_config}"

    def sel2record(self, pred, text, tokens):
        # 将生成的结构表达式解析成 String 级别的 Record
        well_formed_list, counter = self._predict_parser.decode(
            gold_list=[],
            pred_list=[pred],
            text_list=[text], )

        # 将抽取的 Spot-Asoc Record 结构
        # 根据不同的 Schema 转换成 Entity/Relation/Event 结果
        pred_records = proprocessing_graph_record(well_formed_list[0],
                                                  self._schema_dict)

        pred = defaultdict(dict)
        # 将 String 级别的 Record 回标成 Offset 级别的 Record
        for task in task_record_map:
            record_map = task_record_map[task](map_config=self._map_config, )

            # 依据不同的 closest_role 规则，对 Offset 进行去重
            pred[task]['offset'] = record_map.to_offset(
                instance=pred_records.get(task, []),
                tokens=tokens, )

            pred[task]['string'] = record_map.to_string(
                pred_records.get(task, []), )
        return pred

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
