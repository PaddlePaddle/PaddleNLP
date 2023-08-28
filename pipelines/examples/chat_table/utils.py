# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import time

from create_index import chat_table, get_text_information

from pipelines.nodes import ErnieBot

all_titles = [
    "主营收入",
    "公司主营业务",
    "主要资产堆积",
    "历年存货堆积",
    "历年主要负债堆积",
    "应收账款及减值准备情况",
    "历年现金流量",
    "日行情K线数据",
    "历年净现比数据",
    "历年营业总收入、归母净利润、扣非净利润数据",
    "历年加权净资产收益率",
    "历年毛利率和净利率数据",
    "资产负债率数据",
    "十大股东数据",
    "股档户数和持股情况数据",
    "历年期间费用数据",
    "营业收入的主营产品构成情况",
    "人均薪酬、创利、创收情况",
    "流动速动比率",
    "分产品毛利率",
    "五大客户、供应商集中度情况",
    "历年研发投入",
    "投入资本回报率",
    "营运能力各周转率",
    "自由现金流",
    "近年来市盈率(TTM)、市净率(LF)、市销率(TTM)变化情况",
    "近年来市盈率(TTM)、市净率(LF)、市销率(TTM)历史分位",
    "前五大客户和供应商占比",
]


def get_answer(api_key=None, secret_key=None, query="", maxlen=11200):
    """
    Get the answer to the query
    """
    prompt = (
        """
    给你年报主要内容的28个标题，具体为
    主营收入；公司主营业务；主要资产堆积；历年存货堆积；历年主要负债堆积；应收账款及减值准备情况；
    历年现金流量；日行情K线数据；历年净现比数据；历年营业总收入、归母净利润、扣非净利润数据；历年加权净资产收益率；历年毛利率和净利率数据；
    资产负债率数据；十大股东数据；股档户数和持股情况数据；历年期间费用数据；营业收入的主营产品构成情况；
    人均薪酬、创利、创收情况；流动速动比率；分产品毛利率；五大客户、供应商集中度情况；历年研发投入；
    投入资本回报率；营运能力各周转率；自由现金流；近年来市盈率(TTM)、市净率(LF)、市销率(TTM)变化情况；
    近年来市盈率(TTM)、市净率(LF)、市销率(TTM)历史分位；前五大客户和供应商占比。
    现在给你一个问题，你需要理解这个问题，并给出这个问题涉及到公司年报哪些标题，
    请你返回一个json格式的字符串，键值包含"公司名"和"涉及标题"，涉及标题的value是一个列表，
    列表包含这个问题涉及到公司年报哪些标题，
    请你保证你输出的涉及标题不要把27个标题都输出，只输出你认为与问题相关性高的标题。
    如果问题关于公司发展情况，你只要输出['主营收入','公司主营业务','历年营业总收入、归母净利润、扣非净利润数据','营业收入的主营产品构成情况','历年毛利率和净利率数据']这些标题，就可以解决问题。
    如果问题关于公司风险点，你只要输入['历年营业总收入、归母净利润、扣非净利润数据','公司主营业务','五大客户、供应商集中度情况','前五大客户和供应商占比','近年来市盈率(TTM)、市净率(LF)、市销率(TTM)历史分位','营运能力各周转率','应收账款及减值准备情况','历年存货堆积']这些标题，就可以解决问题。
    如果问题关于公司股票、债券表现情况，你只要输入['公司主营业务','日行情K线数据','近年来市盈率(TTM)、市净率(LF)、市销率(TTM)变化情况','近年来市盈率(TTM)、市净率(LF)、市销率(TTM)历史分位','人均薪酬、创利、创收情况','历年加权净资产收益率','历年研发投入','历年净现比数据','历年主要负债堆积']这些标题，就可以解决问题。
    现在让我们开始
    输入问题："""
        + query
        + "\n请你记住你的输出是一个json格式的字符串,键值有两个,请你保证你输出的标题是与问题相关的标题"
    )
    ernie_bot = ErnieBot(api_key=api_key, secret_key=secret_key)
    query_keys = []
    company = ""
    for i in range(3):
        try:
            result = ernie_bot.run(prompt)[0]
            result = result["result"]
            result = result[result.find("{") :]
            result = result[: result.find("}")] + "}"
            result = json.loads(result)
            company = result.get("公司名", "")
            query_keys.extend([i.strip() for i in result["涉及标题"] if i.strip() in all_titles])
        except:
            time.sleep(3)
            continue
    query_keys = list(set(query_keys))
    if query_keys == []:
        return "我无法解答这个问题"
    else:
        text = ""
        for key in query_keys:
            if key in ["主营收入", "公司主营业务"]:
                query_text = "请提取" + company + key + "的信息,来回答" + query + "这个问题"
                text += "从" + key + "角度：\n" + get_text_information(query_text) + "\n"
            elif key != "日行情K线数据":
                query_text = "请提取" + company + key + "的信息来解答" + query + "这个问题"
                text += "从" + key + "角度：\n" + chat_table(query_text, api_key, secret_key, key) + "\n"
    ernie_bot = ErnieBot(api_key, secret_key)
    try:
        prompt = "你现在是金融助手，请你根据背景信息回答问题。请你记住，你的回答要基于背景信息，不要胡编乱造。背景信息{information},请回答相关问题{query}"
        information = text[: maxlen - 1 - len(query) - len(prompt)]
        prompt = prompt.format(information=information, query=query)
        text = ernie_bot.run(prompt)[0]["result"] + "\n" + text
        return text
    except:
        return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default="", type=str, help="The API Key.")
    parser.add_argument("--secret_key", default="", type=str, help="The secret key.")
    parser.add_argument("--query", default="宁德时代的股票可以买吗？", type=str, help="the query")
    args = parser.parse_args()
    result = get_answer(query=args.query, api_key=args.api_key, secret_key=args.secret_key)
    print(result)
    result = get_text_information(query="宁德时代的公司主营业务")
    print(result)
