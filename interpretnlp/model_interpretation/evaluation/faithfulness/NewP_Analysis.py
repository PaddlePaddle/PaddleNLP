"""
 This script includes code to calculating NewP score for results form
 sentiment analysis, textual similarity, and MRC task
"""
import argparse
import os
import json
import numpy as np

def get_args():
    """
    get args
    """
    parser = argparse.ArgumentParser('NewP eval')
    
    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--golden_path', required=True)

    args = parser.parse_args()
    return args


def dataLoad(args):
    """
    load result data from file
    """
    pred_path = args.pred_path
    golden_path = args.golden_path

    with open(pred_path, 'r') as f_text:
        pred_list = []
        for line in f_text.readlines():
            line_dict = json.loads(line)
            pred_list.append(line_dict)

    with open(golden_path, 'r') as f_text:
        gold_list = {}
        for line in f_text.readlines():
            line_dict = json.loads(line)
            gold_list[line_dict['sent_id']] = line_dict
    return pred_list, gold_list


def analysis(args, instance, gold_list):
    """
    analysis result according to result data
    """
    New_P_list = []
    for ins in instance:
        golden_label = ins['pred_label']
        text_correct = 1 if ins['rationale_pred'] == golden_label else 0
        text_exclusive_correct = 1 if ins['no_rationale_pred'] == golden_label else 0 
        New_P_correct = 1 if (text_correct == 1 and text_exclusive_correct == 0) else 0
        New_P_list.append(New_P_correct)
    
    total_New_P = np.sum(New_P_list) / len(gold_list) if len(gold_list) else 0

    print('total\t%d\t%.1f' % (len(New_P_list), 100 * total_New_P))


if __name__ == '__main__':
    args = get_args()
    pred_list, gold_list = dataLoad(args)
    analysis(args, pred_list, gold_list)