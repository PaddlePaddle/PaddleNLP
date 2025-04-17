# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 Langboat Authors. All Rights Reserved.
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
"""
https://github.com/Langboat/mengzi-zero-shot
"""

from collections import Counter

import paddle

from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer


def task_type_map(task_type):
    task_map = {
        "sentiment_classifier": sentiment_cls,
        "news_classifier": news_cls,
        "medical_domain_intent_classifier": domain_cls,
        "entity_extraction": entity_extr,
        "text_similarity": text_sim,
        "financial_relationship_extraction": finance_extr,
        "ad_generation": ad_gen,
        "comment_object_extraction": com_obj_extr,
    }

    return task_map[task_type]


def create_input_with_prompt(task_type, input_string, input_string2=None, entity1=None, entity2=None):
    prompt_map = task_type_map(task_type)

    if task_type == "text_similarity":
        return prompt_map(input_string, input_string2)
    elif task_type == "financial_relationship_extraction":
        return prompt_map(input_string, entity1, entity2)
    return prompt_map(input_string)


def entity_extr(
    s,
):
    """
    dataset: CLUENER
    task: 实体抽取
    output:
    """
    prompts = [f"“{s}”找出上述句子中的实体和他们对应的类别"]
    return prompts


def text_sim(s1, s2):
    """
    dataset:
    task: 语义相似度
    output:
    """
    prompts = [f"“{s1}”和“{s2}”这两句话是在说同一件事吗?"]
    return prompts


def finance_extr(s, e1, e2):
    """
    dataset:
    task: 金融关系抽取
    output:
    """
    prompts = [f"“{s}”中的“{e1}”和“{e2}”是什么关系？答:"]
    return prompts


def ad_gen(s):
    """
    dataset:
    task: 广告文案生成
    output:
    """
    prompts = [f"请根据以下产品信息设计广告文案。商品信息:{s}"]
    return prompts


def domain_cls(s):
    """
    dataset:
    task: 医学领域意图分类
    output:
    """
    # dataset: quake-qic
    prompts = [f"问题:“{s}”。此问题的医学意图是什么？选项：病情诊断，病因分析，治疗方案，就医建议，指标解读，疾病描述，后果表述，注意事项，功效作用，医疗费用。"]
    return prompts


def sentiment_cls(s):
    """
    dataset: eprstmt
    task: 评论情感分类
    output: 消极/积极
    """
    prompts = [f"评论:{s}。请判断该条评论所属类别(积极或消极)并填至空格处。回答："]
    #    f'"{s}"。 如果这个评论的作者是客观的，那么请问这个评论的内容是什么态度的回答？答：',
    #    f'现有机器人能判断句子是消极评论还是积极评论。已知句子：“{s}”。这个机器人将给出的答案是：'
    return prompts


def com_obj_extr(s):
    """
    dataset:
    task: 评论对象抽取
    output:
    """
    prompts = [f"评论:{s}.这条评论的评价对象是谁？"]
    return prompts


def news_cls(s):
    """
    dataset: tnews
    task: 新闻分类
    output:
    """
    label_list = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "股票", "农业", "电竞"]

    prompts = [
        f'“{s}”是什么新闻频道写的？选项：{"，".join(label_list)}。答：',
    ]
    #    f'这条新闻是关于什么主题的？新闻：{s}。选项：{"，".join(label_list)}。答：',
    #    f'这是关于“{"，".join(label_list)}”中哪个选项的文章？文章：{s}。 答：']
    return prompts


class Demo:
    def __init__(self, model_name_or_path="Langboat/mengzi-t5-base-mt", max_predict_len=512):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        print("Loading the model parameters, please wait...")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.eval()
        self.max_predict_len = max_predict_len
        print("Model loaded.")

    def token_decode(self, s):
        return self.tokenizer.decode(s, skip_special_tokens=True)

    def pick_most_common(self, x):
        return Counter(x).most_common(1)[0][0]

    @paddle.no_grad()
    def generate(self, task_type, input_string, input_string2=None, entity1=None, entity2=None, max_predict_len=None):
        max_predict_len = max_predict_len if max_predict_len is not None else self.max_predict_len

        input_text = create_input_with_prompt(task_type, input_string, input_string2, entity1, entity2)
        # tokenize
        encodings = self.tokenizer(input_text, max_seq_len=512)
        encodings = {k: paddle.to_tensor(v) for k, v in encodings.items()}
        outputs = self.model.generate(**encodings, max_length=max_predict_len)[0]
        dec_out = list(map(self.token_decode, outputs))
        output = self.pick_most_common(dec_out)
        print("input_text:", input_text[0])
        print("output:", output)
        print("=" * 50)
        return output


if __name__ == "__main__":

    demo = Demo(model_name_or_path="Langboat/mengzi-t5-base-mt")
    # (1) 实体抽取
    demo.generate(task_type="entity_extraction", input_string="导致泗水的砭石受到追捧，价格突然上涨。而泗水县文化市场综合执法局颜鲲表示，根据监控")
    # 泗水：地址，泗水县文化市场综合执法局：政府，颜鲲：姓名

    # (2) 语义相似度
    demo.generate(task_type="text_similarity", input_string="你好，我还款银行怎么更换", input_string2="怎么更换绑定还款的卡")
    # 是

    # (3) 金融关系抽取
    demo.generate(
        task_type="financial_relationship_extraction",
        input_string="为打消市场顾虑,工行两位洋股东——美国运通和安联集团昨晚做出承诺,近期不会减持工行H股。",
        entity1="工行",
        entity2="美国运通",
    )
    # 被持股

    # (4) 广告文案生成
    demo.generate(task_type="ad_generation", input_string="类型-裤，版型-宽松，风格-潮，风格-复古，风格-文艺，图案-复古，裤型-直筒裤，裤腰型-高腰，裤口-毛边")
    # 这款牛仔裤采用高腰直筒的版型设计,搭配宽松的裤型,穿着舒适又显潮流感。而裤脚的毛边设计,增添几分复古文艺的气息。

    # (5) 医学领域意图分类
    demo.generate(task_type="medical_domain_intent_classifier", input_string="呼气试验阳性什么意思")
    # 指标解读

    # (6) 情感分类
    demo.generate(task_type="sentiment_classifier", input_string="房间很一般，小，且让人感觉脏，隔音效果差，能听到走廊的人讲话，走廊光线昏暗，旁边没有什么可吃")
    # 消极

    # (7) 评论对象抽取
    demo.generate(task_type="comment_object_extraction", input_string="灵水的水质清澈，建议带个浮潜装备，可以看清湖里的小鱼。")
    # 灵水

    # (8) 新闻分类
    demo.generate(task_type="news_classifier", input_string="懒人适合种的果树：长得多、好打理，果子多得都得送邻居吃")
    # 农业
