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

tnews_labels = {
    '109': '科技',
    '102': '娱乐',
    '107': '汽车',
    '112': '旅游',
    '104': '财经',
    '108': '教育',
    '113': '国际',
    '106': '房产',
    '116': '电竞',
    '110': '军事',
    '100': '故事',
    '101': '文化',
    '103': '体育',
    '115': '农业',
    '114': '股票'
}

tnews_label_descriptions = {
    key: "下面报道一条" + value + "新闻"
    for key, value in tnews_labels.items()
}

eprstmt_labels = {'Negative': '不满意', 'Positive': '满意'}
eprstmt_label_descriptions = {
    key: "这表达了" + value + "的情感"
    for key, value in eprstmt_labels.items()
}

csldcp_labels = {
    "材料科学与工程": "材料",
    "作物学": "作物",
    "口腔医学": "口腔",
    "药学": "药学",
    "教育学": "教育",
    "水利工程": "水利",
    "理论经济学": "经济",
    "食品科学与工程": "食品",
    "畜牧学/兽医学": "兽医",
    "体育学": "体育",
    "核科学与技术": "核能",
    "力学": "力学",
    "园艺学": "园艺",
    "水产": "水产",
    "法学": "法学",
    "地质学/地质资源与地质工程": "地质",
    "石油与天然气工程": "能源",
    "农林经济管理": "农林",
    "信息与通信工程": "通信",
    "图书馆、情报与档案管理": "情报",
    "政治学": "政治",
    "电气工程": "电气",
    "海洋科学": "海洋",
    "民族学": "民族",
    "航空宇航科学与技术": "航空",
    "化学/化学工程与技术": "化工",
    "哲学": "哲学",
    "公共卫生与预防医学": "卫生",
    "艺术学": "艺术",
    "农业工程": "农业",
    "船舶与海洋工程": "船舶",
    "计算机科学与技术": "计科",
    "冶金工程": "冶金",
    "交通运输工程": "交通",
    "动力工程及工程热物理": "动力",
    "纺织科学与工程": "纺织",
    "建筑学": "建筑",
    "环境科学与工程": "环境",
    "公共管理": "管理",
    "数学": "数学",
    "物理学": "物理",
    "林学/林业工程": "林业",
    "心理学": "心理",
    "历史学": "历史",
    "工商管理": "工商",
    "应用经济学": "经济",
    "中医学/中药学": "中医",
    "天文学": "天文",
    "机械工程": "机械",
    "土木工程": "土木",
    "光学工程": "光学",
    "地理学": "地理",
    "农业资源利用": "农业",
    "生物学/生物科学与工程": "生物",
    "兵器科学与技术": "兵器",
    "矿业工程": "矿业",
    "大气科学": "大气",
    "基础医学/临床医学": "医学",
    "电子科学与技术": "电子",
    "测绘科学与技术": "测绘",
    "控制科学与工程": "控制",
    "军事学": "军事",
    "中国语言文学": "语言",
    "新闻传播学": "新闻",
    "社会学": "社会",
    "地球物理学": "地球",
    "植物保护": "植物",
}

csldcp_label_description = {
    key: "这篇论文阐述了" + key
    for key, value in csldcp_labels.items()
}

iflytek_labels = {
    "0": "打车",
    "100": "美颜",
    "101": "影像剪辑",
    "102": "摄影修图",
    "103": "相机",
    "104": "绘画",
    "105": "二手",
    "106": "电商",
    "107": "团购",
    "108": "外卖",
    "109": "电影票务",
    "10": "社区服务",
    "110": "社区超市",
    "111": "购物咨询",
    "112": "笔记",
    "113": "办公",
    "114": "日程管理",
    "115": "女性",
    "116": "经营",
    "117": "收款",
    "118": "其他",
    "11": "薅羊毛",
    "12": "魔幻",
    "13": "仙侠",
    "14": "卡牌",
    "15": "飞行空战",
    "16": "射击游戏",
    "17": "休闲益智",
    "18": "动作类",
    "19": "体育竞技",
    "1": "地图导航",
    "20": "棋牌中心",
    "21": "经营养成",
    "22": "策略",
    "23": "MOBA",
    "24": "辅助工具",
    "25": "约会社交",
    "26": "即时通讯",
    "27": "工作社交",
    "28": "论坛圈子",
    "29": "婚恋社交",
    "2": "免费WIFI",
    "30": "情侣社交",
    "31": "社交工具",
    "32": "生活社交",
    "33": "微博博客",
    "34": "新闻",
    "35": "漫画",
    "36": "小说",
    "37": "技术",
    "38": "教辅",
    "39": "问答交流",
    "3": "租车",
    "40": "搞笑",
    "41": "杂志",
    "42": "百科",
    "43": "影视娱乐",
    "44": "求职",
    "45": "兼职",
    "46": "视频",
    "47": "短视频",
    "48": "音乐",
    "49": "直播",
    "4": "同城服务",
    "50": "电台",
    "51": "K歌",
    "52": "成人",
    "53": "中小学",
    "54": "职考",
    "55": "公务员",
    "56": "英语",
    "57": "视频教育",
    "58": "高等教育",
    "59": "成人教育",
    "5": "快递物流",
    "60": "艺术",
    "61": "语言(非英语)",
    "62": "旅游资讯",
    "63": "综合预定",
    "64": "民航",
    "65": "铁路",
    "66": "酒店",
    "67": "行程管理",
    "68": "民宿短租",
    "69": "出国",
    "6": "婚庆",
    "70": "工具",
    "71": "亲子儿童",
    "72": "母婴",
    "73": "驾校",
    "74": "违章",
    "75": "汽车咨询",
    "76": "汽车交易",
    "77": "日常养车",
    "78": "行车辅助",
    "79": "租房",
    "7": "家政",
    "80": "买房",
    "81": "装修家居",
    "82": "电子产品",
    "83": "问诊挂号",
    "84": "养生保健",
    "85": "医疗服务",
    "86": "减肥瘦身",
    "87": "美妆美业",
    "88": "菜谱",
    "89": "餐饮店",
    "8": "公共交通",
    "90": "体育咨讯",
    "91": "运动健身",
    "92": "支付",
    "93": "保险",
    "94": "股票",
    "95": "借贷",
    "96": "理财",
    "97": "彩票",
    "98": "记账",
    "99": "银行",
    "9": "政务",
}

iflytek_label_description = {
    key: "这段文本的应用描述主题是" + value
    for key, value in iflytek_labels.items()
}

bustm_label_description = {"1": "entail", "0": "not_entail"}

ocnli_label_description = {
    "contradiction": "不",
    "entailment": "很",
    "neutral": "无"
}

# Note: chid_label_description has no effect for model performance
# just for get candidate number from chid_label_description
chid_label_description = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
}

csl_label_description = {"1": "entail", "0": "not_entail"}

cluewsc_label_description = {"true": "entail", "false": "not_entail"}

TASK_LABELS_DESC = {
    "tnews": tnews_label_descriptions,
    "eprstmt": eprstmt_label_descriptions,
    "csldcp": csldcp_label_description,
    "iflytek": iflytek_label_description,
    "bustm": bustm_label_description,
    "ocnli": ocnli_label_description,
    "chid": chid_label_description,
    "csl": csl_label_description,
    "cluewsc": cluewsc_label_description,
}
