#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import sys
import numpy as np
from pprint import pprint
from uie.extraction.scorer import EntityScorer, RelationScorer, EventScorer


def read_file(file_name):
    return [line for line in open(file_name).readlines()]


def write_to_file(result, output_filename, prefix=None):
    with open(output_filename, 'w') as output:
        for key, value in result.items():
            if prefix:
                key = '%s_%s' % (prefix, key)
            output.write("%s=%s\n" % (key, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', dest='gold_folder', help="Golden Dataset folder")
    parser.add_argument(
        '-p', dest='pred_folder', nargs='+', help="Predicted model folder")
    parser.add_argument(
        '-v',
        dest='verbose',
        action='store_true',
        help='Show more information during running')
    parser.add_argument(
        '-w',
        dest='write_to_file',
        action='store_true',
        help="Write evaluation results to predicted folder")
    parser.add_argument(
        '-m',
        dest='match_mode',
        default='normal',
        choices=['set', 'normal', 'multimatch'])
    parser.add_argument(
        '-case', dest='case', action='store_true', help='Show case study')
    options = parser.parse_args()

    data_dict = {
        'eval': ['eval_preds_record.txt', 'val.json'],
        'test': ['test_preds_record.txt', 'test.json'],
    }

    task_dict = {
        'entity': EntityScorer,
        'relation': RelationScorer,
        'event': EventScorer,
    }

    result_list = {'eval': list(), 'test': list()}
    for pred_folder in options.pred_folder:
        gold_folder = options.gold_folder

        for data_key, (generation, gold_file) in data_dict.items():

            gold_filename = os.path.join(gold_folder, gold_file)
            pred_filename = os.path.join(pred_folder, generation)

            if not os.path.exists(pred_filename):
                sys.stderr.write("%s not found.\n" % pred_filename)
                continue

            print("pred:", pred_filename)
            print("gold:", gold_filename)

            if options.case:
                for pred_line, gold_line in zip(
                        read_file(pred_filename), read_file(gold_filename)):
                    gold_instance = json.loads(gold_line)
                    pred_instance = json.loads(pred_line)
                    print('=========================')
                    print(gold_instance['text'])
                    for task in task_dict:
                        scorer = task_dict[task]
                        gold = scorer.load_gold_list([gold_instance[task]])[0]
                        pred = scorer.load_pred_list([pred_instance[task]])[0]
                        min_length = max(
                            len(gold['string']),
                            len(pred['string']),
                            len(gold.get('string_trigger', [])),
                            len(pred.get('string_trigger', [])),
                            len(gold.get('string_role', [])),
                            len(pred.get('string_role', [])), )
                        if min_length == 0:
                            continue
                        if task == 'entity':
                            print("Entity Gold:", sorted(gold['string']))
                            print("Entity Pred:", sorted(pred['string']))
                        if task == 'relation':
                            print("Relation Gold:", sorted(gold['string']))
                            print("Relation Pred:", sorted(pred['string']))
                        if task == 'event':
                            print("Event Gold Trigger:",
                                  sorted(gold['string_trigger']))
                            print("Event Pred Trigger:",
                                  sorted(pred['string_trigger']))
                            print("Event Gold Role   :",
                                  sorted(gold['string_role']))
                            print("Event Pred Role   :",
                                  sorted(pred['string_role']))

            results = dict()
            for task in task_dict:
                if task not in json.loads(read_file(pred_filename)[0]):
                    continue
                key = task
                scorer = task_dict[task]
                gold_list = [
                    json.loads(line)[key] for line in read_file(gold_filename)
                ]
                pred_list = [
                    json.loads(line)[task] for line in read_file(pred_filename)
                ]

                assert len(pred_list) == len(gold_list)
                gold_instance_list = scorer.load_gold_list(gold_list)
                pred_instance_list = scorer.load_pred_list(pred_list)
                assert len(pred_instance_list) == len(gold_instance_list)
                sub_results = scorer.eval_instance_list(
                    gold_instance_list=gold_instance_list,
                    pred_instance_list=pred_instance_list,
                    verbose=options.verbose,
                    match_mode=options.match_mode, )
                results.update(sub_results)

            pprint(results)
            result_list[data_key] += [results]

            if options.write_to_file:
                output_filename = "%s/%s_results.txt" % (pred_folder, data_key)
                write_to_file(
                    result=results,
                    output_filename=output_filename,
                    prefix=data_key, )

    print("===========> AVG <===========")

    for data_key in data_dict:
        if len(result_list[data_key]) < 1:
            continue
        for key in result_list[data_key][0]:
            ave = np.mean([result[key] for result in result_list[data_key]])
            print(data_key, key, ave)


if __name__ == "__main__":
    main()
