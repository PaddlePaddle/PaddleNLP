#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json


def load_jsonlines_file(filename):
    return [json.loads(line) for line in open(filename)]


# 将关系抽取结果转换到提交格式
def convert_relation(relation):
    return {
        'type': relation[0],
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
        'type': event['type'],
        'text': event['trigger'],
        'args': [{
            'type': role_type,
            'text': arg
        } for role_type, arg in event['roles']]
    }


# 合并抽取结果和实体编号
def merge_pred_text_file(text_filename, pred_filename):

    # 读取原始文件中的数据，用于获取 ID
    test_instances = load_jsonlines_file(text_filename)
    # 读取抽取结果的预测文件
    pred_instances = load_jsonlines_file(pred_filename)

    assert len(test_instances) == len(pred_instances)

    to_sumbit_instances = list()
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
        to_sumbit_instances += [to_sumbit_instance]

    return to_sumbit_instances


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/duuie')
    parser.add_argument('--model', required=True)
    parser.add_argument('--submit', default='submit.txt')
    options = parser.parse_args()
    data_folder = options.data
    model_folder = options.model
    submit_filename = options.submit
    schema_list = [
        "company_info_event", "competition_info", "disaster",
        "finance_gongsishangshi", "finance_gudongzengchi",
        "finance_jiechuzhijia", "finance_qiyepochan", "finance_qiyeshougou",
        "finance_zhijia", "life_event", "msra", "company_info_relation",
        "conv_asa", "finance_gaoguanbiandong", "finance_gudongjianchi",
        "finance_gufenhuigou", "finance_kuisun", "finance_qiyerongzi",
        "finance_yuetan", "finance_zhongbiao", "life_relation", "peoples_daily"
    ]

    to_sumbit_instances = list()
    for schema in schema_list:
        print(f"Merge {schema} ...")
        test_filename = f"{data_folder}/{schema}/val.json"
        pred_filename = f"{model_folder}/valid-duuie_{schema}-preds_record.txt"
        sub_to_sumbit_instances = merge_pred_text_file(
            text_filename=test_filename,
            pred_filename=pred_filename, )
        to_sumbit_instances += sub_to_sumbit_instances

    with open(submit_filename, 'w') as output:
        for instance in to_sumbit_instances:
            output.write(json.dumps(instance, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
