# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import argparse
from multiprocessing import cpu_count
from ernie_health_predictor import ErnieHealthPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--task_name",
        default='QIC',
        type=str,
        help="The name of the task to perform predict, selected in: QIC, QTR, QQR, CTC, STS, CDN, CMeEE, CMeIE"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="ernie-health-chinese",
        type=str,
        help="The directory or name of model.", )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--use_quantize",
        action='store_true',
        help="Whether to use quantization for acceleration.", )
    parser.add_argument(
        "--num_threads",
        default=cpu_count(),
        type=int,
        help="num_threads for cpu.", )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.task_name = args.task_name.lower()
    args.device = 'cpu'
    predictor = ErnieHealthPredictor(args)

    if args.task_name == 'qic':
        text = ["心肌缺血如何治疗与调养呢？", "什么叫痔核脱出？什么叫外痔？"]
    elif args.task_name == 'qtr':
        text = [["儿童远视眼怎么恢复视力", "远视眼该如何保养才能恢复一些视力"],
                ["抗生素的药有哪些", "抗生素类的药物都有哪些？"]]
    elif args.task_name == 'qqr':
        text = [["茴香是发物吗", "茴香怎么吃？"], ["气的胃疼是怎么回事", "气到胃痛是什么原因"]]
    elif args.task_name == 'ctc':
        text = ["(1)前牙结构发育不良：釉质发育不全、氟斑牙、四环素牙等；", "怀疑或确有酒精或药物滥用史；"]
    elif args.task_name == 'sts':
        text = [["糖尿病能吃减肥药吗？能治愈吗？", "糖尿病为什么不能吃减肥药"],
                ["H型高血压的定义", "WHO对高血压的最新分类定义标准数值"]]
    elif args.task_name == 'cdn':
        text = [["1型糖尿病性植物神经病变", " 1型糖尿病肾病IV期"], ["髂腰肌囊性占位", "髂肌囊肿"]]
    elif args.task_name == 'cmeee':
        text = [
            "研究证实，细胞减少与肺内病变程度及肺内炎性病变吸收程度密切相关。",
            "可为不规则发热、稽留热或弛张热，但以不规则发热为多，可能与患儿应用退热药物导致热型不规律有关。"
        ]
    elif args.task_name == 'cmeie':
        text = [
            "骶髂关节炎是明确诊断JAS的关键条件。若有肋椎关节病变会使胸部扩张度减小。",
            "稳定型缺血性心脏疾病@肥胖与缺乏活动也导致高血压增多。"
        ]
    else:
        print(args.task_name, "is not supported!")

    outputs = predictor.predict(text)


if __name__ == "__main__":
    main()
