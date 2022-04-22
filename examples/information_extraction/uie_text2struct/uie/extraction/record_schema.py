#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from collections import defaultdict
from typing import List


class RecordSchema:
    def __init__(self, type_list, role_list, type_role_dict):
        self.type_list = type_list
        self.role_list = role_list
        self.type_role_dict = type_role_dict

    def __repr__(self) -> str:
        return f"Type: {self.type_list}\n" \
            f"Role: {self.role_list}\n" \
            f"Map: {self.type_role_dict}"

    @staticmethod
    def get_empty_schema():
        return RecordSchema(
            type_list=list(), role_list=list(), type_role_dict=dict())

    @staticmethod
    def read_from_file(filename):
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        return RecordSchema(type_list, role_list, type_role_dict)

    def write_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write(json.dumps(self.type_list) + '\n')
            output.write(json.dumps(self.role_list) + '\n')
            output.write(json.dumps(self.type_role_dict) + '\n')


def merge_schema(schema_list: List[RecordSchema]):
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
        type_role_dict=type_role_dict, )
