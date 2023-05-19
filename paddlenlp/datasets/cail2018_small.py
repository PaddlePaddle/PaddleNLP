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

import collections
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url

from ..utils.env import DATA_HOME
from .dataset import DatasetBuilder

__all__ = ["CAIL2018Small"]


class CAIL2018Small(DatasetBuilder):
    """
    CAIL2018-Small 196,000 criminal cases，which are collected from http://wenshu.court.gov.cn/
    published by the Supreme People’s Court of China. Each case in CAIL2018 consists of two parts,
    i.e., fact description and corresponding judgment result. The judgment result of each case is
    refined into 3 representative ones, including relevant law articles, charges, and prison terms.

    charges: predict the charges from referee result with regular expressions.

    law_articles: predict the relevant law articles from referee result with regular expressions.

    prison_term: predict the prison terms from referee result with regular expressions.

    Find more dataset dertails in https://github.com/thunlp/CAIL
    """

    lazy = False
    URL = "https://paddlenlp.bj.bcebos.com/datasets/cail2018_small.tar.gz"
    MD5 = "963401d107150e250580d115dd2d43fc"
    META_INFO = collections.namedtuple("META_INFO", ("file", "md5"))
    SPLITS = {
        "train": META_INFO(os.path.join("cail2018_small", "train.json"), "e11fc099cc7709a8d128e9fe9f029621"),
        "dev": META_INFO(os.path.join("cail2018_small", "dev.json"), "ee13108aee6a08a94490fadeb400debb"),
        "test": META_INFO(os.path.join("cail2018_small", "test.json"), "27cea977fff2f85b5c32a8e0f708b093"),
    }

    def _get_data(self, mode, **kwargs):
        """Check and download Dataset"""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and not md5file(fullname) == data_hash):

            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, *args):

        with open(filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                sentence = line["fact"]
                if self.name == "charges":
                    label = line["meta"]["accusation"]
                    yield {"sentence": sentence, "label": label}
                elif self.name == "law_articles":
                    label = line["meta"]["relevant_articles"]
                    yield {"sentence": sentence, "label": label}
                elif self.name == "prison_term":
                    if line["meta"]["term_of_imprisonment"]["life_imprisonment"]:
                        lp = -1
                    elif line["meta"]["term_of_imprisonment"]["death_penalty"]:
                        lp = -2
                    else:
                        lp = line["meta"]["term_of_imprisonment"]["imprisonment"]
                    yield {"sentence": sentence, "label": lp}
                else:
                    assert "Dataset name {} does not exist".format(self.name)
        f.close()

    def get_labels(self):
        """
        Return labels of the CAIL2018-Small.
        """
        if self.name == "charges":
            return [
                "故意伤害",
                "盗窃",
                "危险驾驶",
                "非法[持有、私藏][枪支、弹药]",
                "交通肇事",
                "寻衅滋事",
                "[窝藏、包庇]",
                "放火",
                "故意毁坏财物",
                "绑架",
                "赌博",
                "妨害公务",
                "合同诈骗",
                "[走私、贩卖、运输、制造]毒品",
                "抢劫",
                "非法拘禁",
                "诬告陷害",
                "非法采矿",
                "容留他人吸毒",
                "强奸",
                "[伪造、变造、买卖]国家机关[公文、证件、印章]",
                "故意杀人",
                "诈骗",
                "聚众斗殴",
                "[掩饰、隐瞒][犯罪所得、犯罪所得收益]",
                "敲诈勒索",
                "[组织、强迫、引诱、容留、介绍]卖淫",
                "[引诱、容留、介绍]卖淫",
                "开设赌场",
                "重大责任事故",
                "抢夺",
                "破坏电力设备",
                "[制造、贩卖、传播]淫秽物品",
                "传播淫秽物品",
                "虐待",
                "非法[采伐、毁坏]国家重点保护植物",
                "非法[制造、买卖、运输、邮寄、储存][枪支、弹药、爆炸物]",
                "受贿",
                "脱逃",
                "行贿",
                "破坏[广播电视设施、公用电信设施]",
                "[伪造、变造]居民身份证",
                "拐卖[妇女、儿童]",
                "强迫交易",
                "拒不支付劳动报酬",
                "帮助[毁灭、伪造]证据",
                "爆炸",
                "污染环境",
                "非法持有毒品",
                "破坏易燃易爆设备",
                "妨害信用卡管理",
                "[引诱、教唆、欺骗]他人吸毒",
                "非法处置[查封、扣押、冻结]的财产",
                "贪污",
                "职务侵占",
                "帮助犯罪分子逃避处罚",
                "盗伐林木",
                "挪用资金",
                "重婚",
                "侵占",
                "[窝藏、转移、收购、销售]赃物",
                "妨害作证",
                "挪用公款",
                "伪造[公司、企业、事业单位、人民团体]印章",
                "[窝藏、转移、隐瞒][毒品、毒赃]",
                "[虚开增值税专用发票、用于骗取出口退税、抵扣税款发票]",
                "非法侵入住宅",
                "信用卡诈骗",
                "非法获取公民个人信息",
                "滥伐林木",
                "非法经营",
                "招摇撞骗",
                "以危险方法危害公共安全",
                "[盗窃、侮辱]尸体",
                "过失致人死亡",
                "[持有、使用]假币",
                "传授犯罪方法",
                "猥亵儿童",
                "逃税",
                "非法吸收公众存款",
                "非法[转让、倒卖]土地使用权",
                "骗取[贷款、票据承兑、金融票证]",
                "破坏生产经营",
                "高利转贷",
                "[盗窃、抢夺][枪支、弹药、爆炸物]",
                "[盗窃、抢夺][枪支、弹药、爆炸物、危险物质]",
                "假冒注册商标",
                "[伪造、变造]金融票证",
                "强迫卖淫",
                "扰乱无线电通讯管理秩序",
                "虚开发票",
                "非法占用农用地",
                "[组织、领导、参加]黑社会性质组织",
                "[隐匿、故意销毁][会计凭证、会计帐簿、财务会计报告]",
                "保险诈骗",
                "强制[猥亵、侮辱]妇女",
                "非国家工作人员受贿",
                "伪造货币",
                "拒不执行[判决、裁定]",
                "[生产、销售]伪劣产品",
                "非法[收购、运输][盗伐、滥伐]的林木",
                "冒充军人招摇撞骗",
                "组织卖淫",
                "持有伪造的发票",
                "[生产、销售][有毒、有害]食品",
                "非法[制造、出售]非法制造的发票",
                "[伪造、变造、买卖]武装部队[公文、证件、印章]",
                "[组织、领导]传销活动",
                "强迫劳动",
                "走私",
                "贷款诈骗",
                "串通投标",
                "虚报注册资本",
                "侮辱",
                "伪证",
                "聚众扰乱社会秩序",
                "聚众扰乱[公共场所秩序、交通秩序]",
                "劫持[船只、汽车]",
                "集资诈骗",
                "盗掘[古文化遗址、古墓葬]",
                "失火",
                "票据诈骗",
                "经济犯",
                "单位行贿",
                "投放危险物质",
                "过失致人重伤",
                "破坏交通设施",
                "聚众哄抢",
                "走私普通[货物、物品]",
                "收买被拐卖的[妇女、儿童]",
                "非法狩猎",
                "销售假冒注册商标的商品",
                "破坏监管秩序",
                "拐骗儿童",
                "非法行医",
                "协助组织卖淫",
                "打击报复证人",
                "强迫他人吸毒",
                "非法[收购、运输、加工、出售][国家重点保护植物、国家重点保护植物制品]",
                "[生产、销售]不符合安全标准的食品",
                "非法买卖制毒物品",
                "滥用职权",
                "聚众冲击国家机关",
                "[出售、购买、运输]假币",
                "对非国家工作人员行贿",
                "[编造、故意传播]虚假恐怖信息",
                "玩忽职守",
                "私分国有资产",
                "非法携带[枪支、弹药、管制刀具、危险物品]危及公共安全",
                "过失以危险方法危害公共安全",
                "走私国家禁止进出口的[货物、物品]",
                "违法发放贷款",
                "徇私枉法",
                "非法[买卖、运输、携带、持有]毒品原植物[种子、幼苗]",
                "动植物检疫徇私舞弊",
                "重大劳动安全事故",
                "走私[武器、弹药]",
                "破坏计算机信息系统",
                "[制作、复制、出版、贩卖、传播]淫秽物品牟利",
                "单位受贿",
                "[生产、销售]伪劣[农药、兽药、化肥、种子]",
                "过失损坏[武器装备、军事设施、军事通信]",
                "破坏交通工具",
                "包庇毒品犯罪分子",
                "[生产、销售]假药",
                "非法种植毒品原植物",
                "诽谤",
                "传播性病",
                "介绍贿赂",
                "金融凭证诈骗",
                "非法[猎捕、杀害][珍贵、濒危]野生动物",
                "徇私舞弊不移交刑事案件",
                "巨额财产来源不明",
                "过失损坏[广播电视设施、公用电信设施]",
                "挪用特定款物",
                "[窃取、收买、非法提供]信用卡信息",
                "非法组织卖血",
                "利用影响力受贿",
                "非法捕捞水产品",
                "对单位行贿",
                "遗弃",
                "徇私舞弊[不征、少征]税款",
                "提供[侵入、非法控制计算机信息系统][程序、工具]",
                "非法进行节育手术",
                "危险物品肇事",
                "非法[制造、买卖、运输、储存]危险物质",
                "非法[制造、销售]非法制造的注册商标标识",
                "侵犯著作权",
                "倒卖[车票、船票]",
                "过失投放危险物质",
                "走私废物",
                "非法出售发票",
                "走私[珍贵动物、珍贵动物制品]",
                "[伪造、倒卖]伪造的有价票证",
                "招收[公务员、学生]徇私舞弊",
                "非法[生产、销售]间谍专用器材",
                "倒卖文物",
                "虐待被监管人",
                "洗钱",
                "非法[生产、买卖]警用装备",
                "非法获取国家秘密",
                "非法[收购、运输、出售][珍贵、濒危野生动物、珍贵、濒危野生动物]制品",
            ]
        elif self.name == "law_articles":
            return [
                114,
                115,
                116,
                117,
                118,
                119,
                122,
                124,
                125,
                127,
                128,
                130,
                132,
                133,
                134,
                135,
                136,
                140,
                141,
                143,
                144,
                147,
                149,
                150,
                151,
                152,
                153,
                155,
                156,
                158,
                159,
                161,
                162,
                163,
                164,
                168,
                170,
                171,
                172,
                175,
                176,
                177,
                184,
                185,
                186,
                191,
                192,
                193,
                194,
                196,
                198,
                199,
                200,
                201,
                205,
                209,
                210,
                211,
                212,
                213,
                214,
                215,
                217,
                220,
                223,
                224,
                225,
                226,
                227,
                228,
                231,
                232,
                233,
                234,
                235,
                236,
                237,
                238,
                239,
                240,
                241,
                243,
                244,
                245,
                246,
                248,
                253,
                258,
                260,
                261,
                262,
                263,
                264,
                266,
                267,
                268,
                269,
                270,
                271,
                272,
                273,
                274,
                275,
                276,
                277,
                279,
                280,
                281,
                282,
                283,
                285,
                286,
                288,
                290,
                291,
                292,
                293,
                294,
                295,
                302,
                303,
                305,
                307,
                308,
                310,
                312,
                313,
                314,
                315,
                316,
                326,
                328,
                333,
                336,
                338,
                340,
                341,
                342,
                343,
                344,
                345,
                346,
                347,
                348,
                349,
                350,
                351,
                352,
                353,
                354,
                356,
                357,
                358,
                359,
                360,
                361,
                363,
                364,
                367,
                369,
                372,
                375,
                382,
                383,
                384,
                385,
                386,
                387,
                388,
                389,
                390,
                391,
                392,
                393,
                395,
                396,
                397,
                399,
                402,
                404,
                413,
                417,
                418,
            ]
        elif self.name == "prison_term":
            return None
        else:
            assert "Dataset name {} does not exist".format(self.name)
