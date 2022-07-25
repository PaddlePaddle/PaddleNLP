# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import psutil

import paddle
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset

from predictor import CLSPredictor

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
parser.add_argument("--model_name_or_path", default="ernie-health-chinese", type=str, help="The directory or name of model.")
parser.add_argument("--dataset", default="KUAKE-QIC", type=str, help="Dataset for text classfication.")
parser.add_argument("--data_file", default=None, type=str, help="The data to predict with one sample per line.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--use_fp16", action='store_true', help="Whether to use fp16 inference, only takes effect when deploying on gpu.")
parser.add_argument("--batch_size", default=200, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--num_threads", default=psutil.cpu_count(logical=False), type=int, help="num_threads for cpu.")
parser.add_argument("--device", choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--device_id", default=0, help="Select which gpu device to train model.")
args = parser.parse_args()
# yapf: enable

LABEL_LIST = {
    'kuake-qic': [
        '病情诊断', '治疗方案', '病因分析', '指标解读', '就医建议', '疾病表述', '后果表述', '注意事项', '功效作用',
        '医疗费用', '其他'
    ],
    'kuake-qtr': ['完全不匹配', '很少匹配，有一些参考价值', '部分匹配', '完全匹配'],
    'kuake-qqr': [
        'B为A的语义父集，B指代范围大于A； 或者A与B语义毫无关联。', 'B为A的语义子集，B指代范围小于A。',
        '表示A与B等价，表述完全一致。'
    ],
    'chip-ctc': [
        '成瘾行为', '居住情况', '年龄', '酒精使用', '过敏耐受', '睡眠', '献血', '能力', '依存性', '知情同意',
        '数据可及性', '设备', '诊断', '饮食', '残疾群体', '疾病', '教育情况', '病例来源', '参与其它试验',
        '伦理审查', '种族', '锻炼', '性别', '健康群体', '实验室检查', '预期寿命', '读写能力', '含有多类别的语句',
        '肿瘤进展', '疾病分期', '护理', '口腔相关', '器官组织状态', '药物', '怀孕相关', '受体状态', '研究者决定',
        '风险评估', '性取向', '体征(医生检测）', ' 吸烟状况', '特殊病人特征', '症状(患者感受)', '治疗或手术'
    ],
    'chip-sts': ['语义不同', '语义相同'],
    'chip-cdn-2c': ['否', '是'],
}

TEXT = {
    'kuake-qic': ["心肌缺血如何治疗与调养呢？", "什么叫痔核脱出？什么叫外痔？"],
    'kuake-qtr': [["儿童远视眼怎么恢复视力", "远视眼该如何保养才能恢复一些视力"],
                  ["抗生素的药有哪些", "抗生素类的药物都有哪些？"]],
    'kuake-qqr': [["茴香是发物吗", "茴香怎么吃？"], ["气的胃疼是怎么回事", "气到胃痛是什么原因"]],
    'chip-ctc': ["(1)前牙结构发育不良：釉质发育不全、氟斑牙、四环素牙等；", "怀疑或确有酒精或药物滥用史；"],
    'chip-sts': [["糖尿病能吃减肥药吗？能治愈吗？", "糖尿病为什么不能吃减肥药"],
                 ["H型高血压的定义", "WHO对高血压的最新分类定义标准数值"]],
    'chip-cdn-2c': [["1型糖尿病性植物神经病变", " 1型糖尿病肾病IV期"], ["髂腰肌囊性占位", "髂肌囊肿"]]
}

METRIC = {
    'kuake-qic': 'acc',
    'kuake-qtr': 'acc',
    'kuake-qqr': 'acc',
    'chip-ctc': 'macro',
    'chip-sts': 'macro',
    'chip-cdn-2c': 'macro'
}

if __name__ == "__main__":
    for arg_name, arg_value in vars(args).items():
        logger.info("{:20}: {}".format(arg_name, arg_value))

    args.dataset = args.dataset.lower()
    label_list = LABEL_LIST[args.dataset]
    if args.data_file is not None:
        with open(args.data_file, 'r') as fp:
            input_data = [x.strip().split('\t') for x in fp.readlines()]
            input_data = [x[0] if len(x) == 1 else x for x in input_data]
    else:
        input_data = TEXT[args.dataset]

    predictor = CLSPredictor(args, label_list)
    predictor.predict(input_data)
