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

from predictor import SPOPredictor

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
parser.add_argument("--model_name_or_path", default="ernie-health-chinese", type=str, help="The directory or name of model.")
parser.add_argument("--dataset", default="CMeIE", type=str, help="Dataset for named entity recognition.")
parser.add_argument("--data_file", default=None, type=str, help="The data to predict with one sample per line.")
parser.add_argument("--max_seq_length", default=300, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--use_fp16", action='store_true', help="Whether to use fp16 inference, only takes effect when deploying on gpu.")
parser.add_argument("--num_threads", default=psutil.cpu_count(logical=False), type=int, help="num_threads for cpu.")
parser.add_argument("--batch_size", default=20, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--device", choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--device_id", default=0, help="Select which gpu device to train model.")
args = parser.parse_args()
# yapf: enable

LABEL_LIST = {
    'cmeie': [
        '预防', '阶段', '就诊科室', '辅助治疗', '化疗', '放射治疗', '手术治疗', '实验室检查', '影像学检查',
        '辅助检查', '组织学检查', '内窥镜检查', '筛查', '多发群体', '发病率', '发病年龄', '多发地区', '发病性别倾向',
        '死亡率', '多发季节', '传播途径', '并发症', '病理分型', '相关（导致）', '鉴别诊断', '相关（转化）',
        '相关（症状）', '临床表现', '治疗后症状', '侵及周围组织转移的症状', '病因', '高危因素', '风险评估因素', '病史',
        '遗传因素', '发病机制', '病理生理', '药物治疗', '发病部位', '转移部位', '外侵部位', '预后状况', '预后生存率',
        '同义词'
    ]
}

TEXT = {
    'cmeie':
    ["骶髂关节炎是明确诊断JAS的关键条件。若有肋椎关节病变会使胸部扩张度减小。", "稳定型缺血性心脏疾病@肥胖与缺乏活动也导致高血压增多。"]
}

if __name__ == "__main__":
    for arg_name, arg_value in vars(args).items():
        logger.info("{:20}: {}".format(arg_name, arg_value))

    dataset = args.dataset.lower()
    label_list = LABEL_LIST[dataset]
    if args.data_file is not None:
        with open(args.data_file, 'r') as fp:
            input_data = [x.strip() for x in fp.readlines()]
    else:
        input_data = TEXT[dataset]

    predictor = SPOPredictor(args, label_list)
    predictor.predict(input_data)
