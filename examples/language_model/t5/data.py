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

import collections
import os
from functools import partial

from paddle.io import BatchSampler, DataLoader
from utils import load_pickle, save_pickle

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset

CLUE_PROCESSED = collections.OrderedDict(
    [
        ("afqmc", (["afqmc sentence1: ", "afqmc sentence2: "], ["不同", "类似"])),
        (
            "tnews",
            (
                ["tnews sentence: "],
                ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "股票", "农业", "电竞"],
            ),
        ),
        (
            "iflytek",
            (
                ["iflytek sentence: "],
                [
                    "打车",
                    "地图导航",
                    "免费WIFI",
                    "租车",
                    "同城服务",
                    "快递物流",
                    "婚庆",
                    "家政",
                    "公共交通",
                    "政务",
                    "社区服务",
                    "薅羊毛",
                    "魔幻",
                    "仙侠",
                    "卡牌",
                    "飞行空战",
                    "射击游戏",
                    "休闲益智",
                    "动作类",
                    "体育竞技",
                    "棋牌中心",
                    "经营养成",
                    "策略",
                    "MOBA",
                    "辅助工具",
                    "约会社交",
                    "即时通讯",
                    "工作社交",
                    "论坛圈子",
                    "婚恋社交",
                    "情侣社交",
                    "社交工具",
                    "生活社交",
                    "微博博客",
                    "新闻",
                    "漫画",
                    "小说",
                    "技术",
                    "教辅",
                    "问答交流",
                    "搞笑",
                    "杂志",
                    "百科",
                    "影视娱乐",
                    "求职",
                    "兼职",
                    "视频",
                    "短视频",
                    "音乐",
                    "直播",
                    "电台",
                    "K歌",
                    "成人",
                    "中小学",
                    "职考",
                    "公务员",
                    "英语",
                    "视频教育",
                    "高等教育",
                    "成人教育",
                    "艺术",
                    "语言(非英语)",
                    "旅游资讯",
                    "综合预定",
                    "民航",
                    "铁路",
                    "酒店",
                    "行程管理",
                    "民宿短租",
                    "出国",
                    "工具",
                    "亲子儿童",
                    "母婴",
                    "驾校",
                    "违章",
                    "汽车咨询",
                    "汽车交易",
                    "日常养车",
                    "行车辅助",
                    "租房",
                    "买房",
                    "装修家居",
                    "电子产品",
                    "问诊挂号",
                    "养生保健",
                    "医疗服务",
                    "减肥瘦身",
                    "美妆美业",
                    "菜谱",
                    "餐饮店",
                    "体育咨讯",
                    "运动健身",
                    "支付",
                    "保险",
                    "股票",
                    "借贷",
                    "理财",
                    "彩票",
                    "记账",
                    "银行",
                    "美颜",
                    "影像剪辑",
                    "摄影修图",
                    "相机",
                    "绘画",
                    "二手",
                    "电商",
                    "团购",
                    "外卖",
                    "电影票务",
                    "社区超市",
                    "购物咨询",
                    "笔记",
                    "办公",
                    "日程管理",
                    "女性",
                    "经营",
                    "收款",
                    "其他",
                ],
            ),
        ),
        ("cmnli", (["cmnli sentence1: ", "cmnli sentence2: "], ["矛盾", "中立", "蕴涵"])),
        ("ocnli", (["ocnli sentence1: ", "ocnli sentence2: "], ["蕴涵", "矛盾", "中立"])),
        ("cluewsc2020", (["cluewsc2020 sentence: "], ["同义", "歧义"])),
        ("csl", ((["csl sentence1: ", "csl sentence2: "], ["伪造", "真实"]))),
    ]
)
GLUE_PROCESSED = collections.OrderedDict(
    [
        ("cola", (["cola sentence: "], ["not_acceptable", "acceptable"])),
        ("sst-2", (["sst2 sentence: "], ["negative", "positive"])),
        (
            "mrpc",
            (["mrpc sentence1: ", " sentence2: "], ["not_equivalent", "equivalent"]),
        ),
        ("sts-b", (["stsb sentence1: ", " sentence2: "], None)),
        ("qqp", (["qqp question1: ", " question2: "], ["not_duplicate", "duplicate"])),
        (
            "mnli",
            (
                ["mnli hypothesis: ", " premise: "],
                ["contradiction", "entailment", "neutral"],
            ),
        ),
        (
            "qnli",
            (["qnli question: ", " sentence: "], ["entailment", "not_entailment"]),
        ),
        (
            "rte",
            (["rte sentence1: ", " rte sentence2: "], ["entailment", "not_entailment"]),
        ),
    ]
)

GLUE_1_1_PROCESSED = collections.OrderedDict(
    [
        ("cola", (["cola sentence: "], ["outrageous", "acceptable"])),
        ("sst-2", (["sst2 sentence: "], ["negative", "positive"])),
        (
            "mrpc",
            (["mrpc sentence1: ", " sentence2: "], ["nonidentical", "equivalent"]),
        ),
        ("sts-b", (["stsb sentence1: ", " sentence2: "], None)),
        ("qqp", (["qqp question1: ", " question2: "], ["inequable", "duplicate"])),
        (
            "mnli",
            (
                ["mnli hypothesis: ", " premise: "],
                ["contradiction", "entailment", "neutral"],
            ),
        ),
        (
            "qnli",
            (["qnli question: ", " sentence: "], ["entailment", "contradiction"]),
        ),
        (
            "rte",
            (["rte sentence1: ", " rte sentence2: "], ["entailment", "contradiction"]),
        ),
    ]
)


def trans_func(example, tokenizer, args):
    task_name = args.task_name
    processed, label = GLUE_PROCESSED[task_name]
    if label:
        id2label = dict(zip(range(len(label)), label))
    else:
        id2label = None

    if not args.is_test:
        if id2label:
            label_text = id2label[example["labels"]]
        else:
            label_text = str(example["labels"])
        target = tokenizer(label_text, return_token_type_ids=False, return_attention_mask=True)

    if len(processed) == 1:
        text = processed[0] + example["sentence"]
    else:
        text = processed[0] + example["sentence1"] + processed[1] + example["sentence2"]

    source = tokenizer(
        text,
        max_seq_len=args.max_seq_length,
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    if not args.is_test:
        return (
            source["input_ids"],
            source["attention_mask"],
            target["input_ids"],
            target["attention_mask"],
        )
    else:
        return source["input_ids"], source["attention_mask"]


def get_train_dataloader(tokenizer, args):
    filename = os.path.join("caches", args.task_name + "_train" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="train")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=True)

    # batchify_fn = lambda samples, fn=Tuple(
    #     Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
    #     Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # attention_mask
    #     Pad(axis=0, pad_val=-100, dtype="int64"),  # lm_labels
    #     Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # decoder_attention_mask
    # ): fn(samples)
    def batchify_fn(
        samples,
        fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # attention_mask
            Pad(axis=0, pad_val=-100, dtype="int64"),  # lm_labels
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # decoder_attention_mask
        ),
    ):
        return fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_dev_dataloader(tokenizer, args):
    filename = os.path.join("caches", args.task_name + "_dev" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="dev")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=False)

    def batchify_fn(
        samples,
        fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # attention_mask
            Pad(axis=0, pad_val=-100, dtype="int64"),  # lm_labels
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # decoder_attention_mask
        ),
    ):
        return fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_mnli_dev_dataloader(tokenizer, args, matched=True):
    if matched:
        split = "dev_matched"
    else:
        split = "dev_mismatched"
    filename = os.path.join("caches", args.task_name + f"_{split}" + ".pkl")
    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits=split)
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=False)

    def batchify_fn(
        samples,
        fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # attention_mask
            Pad(axis=0, pad_val=-100, dtype="int64"),  # lm_labels
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # decoder_attention_mask
        ),
    ):
        return fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader
