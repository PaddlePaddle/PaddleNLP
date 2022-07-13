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

import numpy as np

import paddle
import paddle.nn.functional as F
from paddlenlp.data import Tuple, Pad
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--params_path",
                    default="./checkpoint/",
                    type=str,
                    help="The path to model parameters to be loaded.")
parser.add_argument("--dataset_dir",
                    default=None,
                    type=str,
                    help="Local dataset directory should"
                    " include data.txt and label.txt")
parser.add_argument("--max_seq_length",
                    default=512,
                    type=int,
                    help="The maximum total input sequence length "
                    "after tokenization. Sequences longer than this"
                    "will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument('--device',
                    choices=['cpu', 'gpu', 'xpu', 'npu'],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


@paddle.no_grad()
def predict(data, label_list):
    """
    Predicts the data labels.
    Args:

        data (obj:`List`): The processed data whose each element is one sequence.
        label_map(obj:`List`): The label id (key) to label str (value) map.
 
    """
    paddle.set_device(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
    tokenizer = AutoTokenizer.from_pretrained(args.params_path)

    examples = []
    for text in data:
        result = tokenizer(text=text, max_seq_len=args.max_seq_length)
        examples.append((result['input_ids'], result['token_type_ids']))

    # Seperates data into some batches.
    batches = [
        examples[i:i + args.batch_size]
        for i in range(0, len(examples), args.batch_size)
    ]

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): fn(samples)

    results = []
    model.eval()
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.sigmoid(logits).numpy()
        confidence = []
        for prob in probs:
            labels = []
            for i, p in enumerate(prob):
                if p > 0.5:
                    labels.append(i)
            results.append(labels)

    for idx, text in enumerate(data):
        label_name = [label_list[r] for r in results[idx]]
        print("input data:", text)

        label = []
        for r in results[idx]:
            label.append(label_list[r])
        print('label: {}'.format(', '.join(label)))
        print('---------------------------------')
    return


if __name__ == "__main__":
    if args.dataset_dir is not None:
        data_dir = os.path.join(args.dataset_dir, "data.txt")
        label_dir = os.path.join(args.dataset_dir, "label.txt")
        label_list = []
        data = []
        with open(label_dir, 'r', encoding='utf-8') as f:
            for line in f:
                label_list.append(line.strip())
        f.close()
        with open(data_dir, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())
        f.close()
    else:
        data = [
            "经审理查明，2012年4月5日19时许，被告人王某在杭州市下城区朝晖路农贸市场门口贩卖盗版光碟、淫秽光碟时被民警当场抓获，并当场查获其贩卖的各类光碟5515张，其中5280张某属非法出版物、235张某属淫秽物品。上述事实，被告人王某在庭审中亦无异议，且有经庭审举证、质证的扣押物品清单、赃物照片、公安行政处罚决定书、抓获经过及户籍证明等书证；证人胡某、徐某的证言；出版物鉴定书、淫秽物品审查鉴定书及检查笔录等证据证实，足以认定。",
            "榆林市榆阳区人民检察院指控：2015年11月22日2时许，被告人王某某在自己经营的榆阳区长城福源招待所内，介绍并容留杨某向刘某某、白某向乔某某提供性服务各一次。",
            "静乐县人民检察院指控，2014年8月30日15时许，静乐县苏坊村村民张某某因占地问题去苏坊村半切沟静静铁路第五标施工地点阻拦施工时，遭被告人王某某阻止，张某某打电话叫来儿子李某某，李某某看到张某某躺在地上，打了王某某一耳光。于是王某某指使工人殴打李某某，致李某某受伤。经忻州市公安司法鉴定中心鉴定，李某某的损伤评定为轻伤一级。李某某被打伤后，被告人王某某为逃避法律追究，找到任某某，指使任某某作实施××的伪证，并承诺每月给1万元。同时王某某指使工人王某甲、韩某某去丰润派出所作由任某某打伤李某某的伪证，导致任某某被静乐县公安局以涉嫌××罪刑事拘留。公诉机关认为，被告人王某某的行为触犯了《中华人民共和国刑法》××、《中华人民共和国刑法》××××之规定，应以××罪和××罪追究其刑事责任，数罪并罚。"
        ]
        label_list = [
            '故意伤害', '盗窃', '危险驾驶', '非法[持有、私藏][枪支、弹药]', '交通肇事', '寻衅滋事', '[窝藏、包庇]',
            '放火', '故意毁坏财物', '绑架', '赌博', '妨害公务', '合同诈骗', '[走私、贩卖、运输、制造]毒品', '抢劫',
            '非法拘禁', '诬告陷害', '非法采矿', '容留他人吸毒', '强奸', '[伪造、变造、买卖]国家机关[公文、证件、印章]',
            '故意杀人', '诈骗', '聚众斗殴', '[掩饰、隐瞒][犯罪所得、犯罪所得收益]', '敲诈勒索',
            '[组织、强迫、引诱、容留、介绍]卖淫', '[引诱、容留、介绍]卖淫', '开设赌场', '重大责任事故', '抢夺',
            '破坏电力设备', '[制造、贩卖、传播]淫秽物品', '传播淫秽物品', '虐待', '非法[采伐、毁坏]国家重点保护植物',
            '非法[制造、买卖、运输、邮寄、储存][枪支、弹药、爆炸物]', '受贿', '脱逃', '行贿',
            '破坏[广播电视设施、公用电信设施]', '[伪造、变造]居民身份证', '拐卖[妇女、儿童]', '强迫交易',
            '拒不支付劳动报酬', '帮助[毁灭、伪造]证据', '爆炸', '污染环境', '非法持有毒品', '破坏易燃易爆设备',
            '妨害信用卡管理', '[引诱、教唆、欺骗]他人吸毒', '非法处置[查封、扣押、冻结]的财产', '贪污', '职务侵占',
            '帮助犯罪分子逃避处罚', '盗伐林木', '挪用资金', '重婚', '侵占', '[窝藏、转移、收购、销售]赃物', '妨害作证',
            '挪用公款', '伪造[公司、企业、事业单位、人民团体]印章', '[窝藏、转移、隐瞒][毒品、毒赃]',
            '[虚开增值税专用发票、用于骗取出口退税、抵扣税款发票]', '非法侵入住宅', '信用卡诈骗', '非法获取公民个人信息',
            '滥伐林木', '非法经营', '招摇撞骗', '以危险方法危害公共安全', '[盗窃、侮辱]尸体', '过失致人死亡',
            '[持有、使用]假币', '传授犯罪方法', '猥亵儿童', '逃税', '非法吸收公众存款', '非法[转让、倒卖]土地使用权',
            '骗取[贷款、票据承兑、金融票证]', '破坏生产经营', '高利转贷', '[盗窃、抢夺][枪支、弹药、爆炸物]',
            '[盗窃、抢夺][枪支、弹药、爆炸物、危险物质]', '假冒注册商标', '[伪造、变造]金融票证', '强迫卖淫',
            '扰乱无线电通讯管理秩序', '虚开发票', '非法占用农用地', '[组织、领导、参加]黑社会性质组织',
            '[隐匿、故意销毁][会计凭证、会计帐簿、财务会计报告]', '保险诈骗', '强制[猥亵、侮辱]妇女', '非国家工作人员受贿',
            '伪造货币', '拒不执行[判决、裁定]', '[生产、销售]伪劣产品', '非法[收购、运输][盗伐、滥伐]的林木',
            '冒充军人招摇撞骗', '组织卖淫', '持有伪造的发票', '[生产、销售][有毒、有害]食品',
            '非法[制造、出售]非法制造的发票', '[伪造、变造、买卖]武装部队[公文、证件、印章]', '[组织、领导]传销活动',
            '强迫劳动', '走私', '贷款诈骗', '串通投标', '虚报注册资本', '侮辱', '伪证', '聚众扰乱社会秩序',
            '聚众扰乱[公共场所秩序、交通秩序]', '劫持[船只、汽车]', '集资诈骗', '盗掘[古文化遗址、古墓葬]', '失火',
            '票据诈骗', '经济犯', '单位行贿', '投放危险物质', '过失致人重伤', '破坏交通设施', '聚众哄抢',
            '走私普通[货物、物品]', '收买被拐卖的[妇女、儿童]', '非法狩猎', '销售假冒注册商标的商品', '破坏监管秩序',
            '拐骗儿童', '非法行医', '协助组织卖淫', '打击报复证人', '强迫他人吸毒',
            '非法[收购、运输、加工、出售][国家重点保护植物、国家重点保护植物制品]', '[生产、销售]不符合安全标准的食品',
            '非法买卖制毒物品', '滥用职权', '聚众冲击国家机关', '[出售、购买、运输]假币', '对非国家工作人员行贿',
            '[编造、故意传播]虚假恐怖信息', '玩忽职守', '私分国有资产', '非法携带[枪支、弹药、管制刀具、危险物品]危及公共安全',
            '过失以危险方法危害公共安全', '走私国家禁止进出口的[货物、物品]', '违法发放贷款', '徇私枉法',
            '非法[买卖、运输、携带、持有]毒品原植物[种子、幼苗]', '动植物检疫徇私舞弊', '重大劳动安全事故', '走私[武器、弹药]',
            '破坏计算机信息系统', '[制作、复制、出版、贩卖、传播]淫秽物品牟利', '单位受贿',
            '[生产、销售]伪劣[农药、兽药、化肥、种子]', '过失损坏[武器装备、军事设施、军事通信]', '破坏交通工具',
            '包庇毒品犯罪分子', '[生产、销售]假药', '非法种植毒品原植物', '诽谤', '传播性病', '介绍贿赂',
            '金融凭证诈骗', '非法[猎捕、杀害][珍贵、濒危]野生动物', '徇私舞弊不移交刑事案件', '巨额财产来源不明',
            '过失损坏[广播电视设施、公用电信设施]', '挪用特定款物', '[窃取、收买、非法提供]信用卡信息', '非法组织卖血',
            '利用影响力受贿', '非法捕捞水产品', '对单位行贿', '遗弃', '徇私舞弊[不征、少征]税款',
            '提供[侵入、非法控制计算机信息系统][程序、工具]', '非法进行节育手术', '危险物品肇事',
            '非法[制造、买卖、运输、储存]危险物质', '非法[制造、销售]非法制造的注册商标标识', '侵犯著作权', '倒卖[车票、船票]',
            '过失投放危险物质', '走私废物', '非法出售发票', '走私[珍贵动物、珍贵动物制品]', '[伪造、倒卖]伪造的有价票证',
            '招收[公务员、学生]徇私舞弊', '非法[生产、销售]间谍专用器材', '倒卖文物', '虐待被监管人', '洗钱',
            '非法[生产、买卖]警用装备', '非法获取国家秘密', '非法[收购、运输、出售][珍贵、濒危野生动物、珍贵、濒危野生动物]制品'
        ]
    predict(data, label_list)
