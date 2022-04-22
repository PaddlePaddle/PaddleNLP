#!/usr/bin/env python
# -*- coding:utf-8 -*-
from uie.extraction.record_schema import RecordSchema
from uie.extraction.constants import spot_prompt, asoc_prompt, text_start


class TaskConfig:
    def __init__(self, task_dict) -> None:
        self.dataset_name = task_dict.get('name', '')
        self.task_name = task_dict.get('task', '')
        self.data_path = task_dict.get('path', '')
        self.decoding_format = task_dict.get('decoding_format', '')
        self.weight = int(task_dict.get('weight', 0))
        self.sel2record = task_dict.get('sel2record', '')
        self.metrics = task_dict.get('metrics', [])
        self.eval_match_mode = task_dict.get('eval_match_mode', 'normal')
        self.schema = RecordSchema.read_from_file(
            f"{self.data_path}/{self.task_name}.schema")

    def __repr__(self) -> str:
        return f"dataset: {self.dataset_name}\n" \
               f"task   : {self.task_name}\n" \
               f"format : {self.decoding_format}\n" \
               f"path   : {self.data_path}\n" \
               f"schema : {self.schema}\n" \
               f"metrics: {self.metrics}\n" \
               f"eval_match_mode : {self.eval_match_mode}"

    @staticmethod
    def load_list_from_yaml(task_config):
        import yaml
        configs = yaml.load(open(task_config), Loader=yaml.FullLoader)
        task_configs = filter(lambda x: x.startswith('T'), configs)
        for task_config in task_configs:
            yield TaskConfig(configs[task_config])


class PrefixGenerator:
    def __init__(self, prefix_dict) -> None:
        self.type_list = prefix_dict.get('type', 'task dataset').split()
        self.position = prefix_dict.get('position', 'encoder')

    def __repr__(self) -> str:
        return f"Type.   : {self.type_list}\n" \
               f"Position: {self.position}\n"

    @staticmethod
    def load_from_yaml(dataset_config):
        import yaml
        configs = yaml.load(open(dataset_config), Loader=yaml.FullLoader)
        return PrefixGenerator(configs['Prefix'])

    @staticmethod
    def get_schema_prefix(schema: RecordSchema, add_split=True):
        prefix_list = list()
        for spot_label in sorted(schema.type_list):
            prefix_list += [spot_prompt, spot_label]
        for asoc_label in sorted(schema.role_list):
            prefix_list += [asoc_prompt, asoc_label]
        prefix = ' '.join(prefix_list)
        if add_split:
            return prefix + f' {text_start} '
        else:
            return prefix

    @staticmethod
    def get_dataset_name_prefix(dataset: TaskConfig, add_split=True):
        if add_split:
            return dataset.dataset_name + f' {text_start}'
        else:
            return dataset.dataset_name

    @staticmethod
    def get_task_name_prefix(dataset: TaskConfig, add_split=True):
        if add_split:
            return dataset.task_name + f' {text_start}'
        else:
            return dataset.task_name

    def get_prefix_by_dataset(self, dataset: TaskConfig):
        prefix_list = list()
        for prefix_type in self.type_list:
            if prefix_type == 'task':
                prefix = self.get_task_name_prefix(dataset, add_split=False)
            elif prefix_type == 'dataset':
                prefix = self.get_dataset_name_prefix(dataset, add_split=False)
            elif prefix_type == 'schema':
                prefix = self.get_schema_prefix(dataset.schema, add_split=False)
            elif prefix_type == 'meta':
                # Meta 使用 Schema 的 Prefix
                prefix = self.get_schema_prefix(dataset.schema, add_split=False)
            else:
                raise NotImplementedError("Prefix Type %s is not supported" %
                                          prefix_type)
            prefix_list += [prefix]
        return ' '.join(prefix_list) + f' {text_start}'
