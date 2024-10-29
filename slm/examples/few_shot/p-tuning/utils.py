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

import json
import os
import pathlib

import numpy as np
import paddle

from paddlenlp.datasets import load_dataset

LABEL_TO_STANDARD = {
    "tnews": {
        "news_story": "100",
        "news_culture": "101",
        "news_entertainment": "102",
        "news_sports": "103",
        "news_finance": "104",
        "news_house": "106",
        "news_car": "107",
        "news_edu": "108",
        "news_tech": "109",
        "news_military": "110",
        "news_travel": "112",
        "news_world": "113",
        "news_stock": "114",
        "news_agriculture": "115",
        "news_game": "116",
    },
    "iflytek": {
        "打车": 0,
        "美颜": 100,
        "影像剪辑": 101,
        "摄影修图": 102,
        "相机": 103,
        "绘画": 104,
        "二手": 105,
        "电商": 106,
        "团购": 107,
        "外卖": 108,
        "电影票务": 109,
        "社区服务": 10,
        "社区超市": 110,
        "购物咨询": 111,
        "笔记": 112,
        "办公": 113,
        "日程管理": 114,
        "女性": 115,
        "经营": 116,
        "收款": 117,
        "其他": 118,
        "薅羊毛": 11,
        "魔幻": 12,
        "仙侠": 13,
        "卡牌": 14,
        "飞行空战": 15,
        "射击游戏": 16,
        "休闲益智": 17,
        "动作类": 18,
        "体育竞技": 19,
        "地图导航": 1,
        "棋牌中心": 20,
        "经营养成": 21,
        "策略": 22,
        "MOBA": 23,
        "辅助工具": 24,
        "约会社交": 25,
        "即时通讯": 26,
        "工作社交": 27,
        "论坛圈子": 28,
        "婚恋社交": 29,
        "免费WIFI": 2,
        "情侣社交": 30,
        "社交工具": 31,
        "生活社交": 32,
        "微博博客": 33,
        "新闻": 34,
        "漫画": 35,
        "小说": 36,
        "技术": 37,
        "教辅": 38,
        "问答交流": 39,
        "租车": 3,
        "搞笑": 40,
        "杂志": 41,
        "百科": 42,
        "影视娱乐": 43,
        "求职": 44,
        "兼职": 45,
        "视频": 46,
        "短视频": 47,
        "音乐": 48,
        "直播": 49,
        "同城服务": 4,
        "电台": 50,
        "K歌": 51,
        "成人": 52,
        "中小学": 53,
        "职考": 54,
        "公务员": 55,
        "英语": 56,
        "视频教育": 57,
        "高等教育": 58,
        "成人教育": 59,
        "快递物流": 5,
        "艺术": 60,
        "语言(非英语)": 61,
        "旅游资讯": 62,
        "综合预定": 63,
        "民航": 64,
        "铁路": 65,
        "酒店": 66,
        "行程管理": 67,
        "民宿短租": 68,
        "出国": 69,
        "婚庆": 6,
        "工具": 70,
        "亲子儿童": 71,
        "母婴": 72,
        "驾校": 73,
        "违章": 74,
        "汽车咨询": 75,
        "汽车交易": 76,
        "日常养车": 77,
        "行车辅助": 78,
        "租房": 79,
        "家政": 7,
        "买房": 80,
        "装修家居": 81,
        "电子产品": 82,
        "问诊挂号": 83,
        "养生保健": 84,
        "医疗服务": 85,
        "减肥瘦身": 86,
        "美妆美业": 87,
        "菜谱": 88,
        "餐饮店": 89,
        "公共交通": 8,
        "体育咨讯": 90,
        "运动健身": 91,
        "支付": 92,
        "保险": 93,
        "股票": 94,
        "借贷": 95,
        "理财": 96,
        "彩票": 97,
        "记账": 98,
        "银行": 99,
        "政务": 9,
    },
}


def load_prompt_arguments(args):
    """
    Load prompt and label words according to prompt index.
    """
    with open(args.prompt_path, "r", encoding="utf-8") as fp:
        configs = json.load(fp)
        assert len(configs["verbalizer"]) == len(configs["template"])
        assert configs["verbalizer"][0] is not None
        verbalizer = [configs["verbalizer"][0]]
        last_verb_index = 0
        for index, verb in enumerate(configs["verbalizer"][1:]):
            if verb is None or len(verb) == 0:
                verbalizer.append(configs["verbalizer"][last_verb_index])
            else:
                verbalizer.append(verb)
                last_verb_index = index + 1
        configs["verbalizer"] = verbalizer
        args.prompt = configs["template"][args.prompt_index]["text"]
        label_words = configs["verbalizer"][args.prompt_index]
        if isinstance(label_words, list):
            label_words = {k: k for k in label_words}
        args.label_words = label_words
        return args


def save_pseudo_data(save_path, task_name, label_preds, verbalizer, labels):
    """
    Combine unsupervised data and corresponding predicted labels and
    save one example per line.
    """
    if task_name == "cluewsc":
        return None

    data_ds = load_dataset("fewclue", name=task_name, splits="unlabeled")
    preds = paddle.to_tensor(label_preds.predictions)
    preds = verbalizer.aggregate_multiple_mask(preds)
    preds = paddle.nn.functional.softmax(preds, axis=1).numpy()
    label_preds = np.argmax(preds, axis=1)
    label_probs = np.max(preds, axis=1)
    pseudo_data = []
    for index, example in enumerate(data_ds):
        example["labels"] = labels[label_preds[index]]
        example["prob"] = str(label_probs[index])
        pseudo_data.append(example)
    save_data(pseudo_data, save_path)


def save_fewclue_prediction(save_path, task_name, label_preds, verbalizer, labels):
    """
    Extract predicted labels and save as the format required by FewCLUE.
    """
    preds = paddle.to_tensor(label_preds.predictions)
    preds = verbalizer.aggregate_multiple_mask(preds)
    if task_name == "chid":
        batch_size = preds.shape[0]
        preds = paddle.nn.functional.softmax(preds, axis=1)[:, 1]
        preds = preds.reshape([batch_size // 7, 7])
    preds = paddle.nn.functional.softmax(preds, axis=1).numpy()
    preds = np.argmax(preds, axis=1)
    test_ds = load_dataset("fewclue", name=task_name, splits="test")

    ret_list = []
    maps = LABEL_TO_STANDARD.get(task_name, None)
    for idx, example in enumerate(test_ds):
        uid = example.get("id", idx)
        if task_name in ["bustm", "csl"]:
            ret_list.append({"id": uid, "label": str(preds[idx])})
        elif task_name == "chid":
            ret_list.append({"id": uid, "answer": preds[idx]})
        elif task_name in ["cluewsc", "eprstmt", "ocnli", "csldcp"]:
            ret_list.append({"id": uid, "label": labels[preds[idx]]})
        elif task_name in ["iflytek", "tnews"]:
            ret_list.append({"id": uid, "label": str(maps[labels[preds[idx]]])})
    save_file = task_name if task_name in ["bustm", "csldcp", "eprstmt"] else task_name + "f"
    save_data(ret_list, save_path, save_file + "_predict.json")


def save_data(data, save_path, save_file=None):
    if save_file is not None:
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_path, save_file)
    with open(save_path, "w") as fp:
        for example in data:
            fp.write(json.dumps(example, ensure_ascii=False) + "\n")
