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

from create_index import columns_titles, index_name

from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import DensePassageRetriever, ErnieBot, ErnieRanker
from pipelines.pipelines import Pipeline

all_titles = [value for key, value in columns_titles.items()]


def text_retrieval(query, key="", index="text"):
    """
    Obtain the  matching text information
    """
    document_store = FAISSDocumentStore.load(index_name)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="moka-ai/m3e-base",
        passage_embedding_model="moka-ai/m3e-base",
        output_emb_size=None,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        batch_size=16,
        use_gpu=True,
        embed_title=False,
        pooling_mode="mean_tokens",
    )
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    prediction = query_pipeline.run(query=query, params={"Retriever": {"top_k": 5, "index": str(index)}})
    if key == "":
        return prediction["documents"][0].content
    else:
        document = ""
        i = 0
        while i < 5:
            if key in prediction["documents"][i].content:
                document = prediction["documents"][i].content
                break
            i += 1
        return document


def chat_table(query, api_key=None, secret_key=None, key="", maxlen=11200, index="table"):
    """
    Obtain the  matching table information and  conduct Table Q&A
    """
    document_store = FAISSDocumentStore.load(index_name)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="moka-ai/m3e-base",
        passage_embedding_model="moka-ai/m3e-base",
        output_emb_size=None,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        batch_size=16,
        use_gpu=True,
        embed_title=False,
        pooling_mode="mean_tokens",
    )
    ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=True)
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
    prediction = query_pipeline.run(
        query=query, params={"Retriever": {"top_k": 15, "index": str(index)}, "Ranker": {"top_k": 5}}
    )
    documents = ""
    if key != "":
        i = 0
        while i < 5:
            if key in prediction["documents"][i].content:
                documents = prediction["documents"][i].meta["info"]
                break
            i += 1
    else:
        documents = prediction["documents"][0].meta["info"]
    if documents == "":
        return ""
    else:
        ernie_bot = ErnieBot(api_key=api_key, secret_key=secret_key)
        prompt = "你是一个金融助手，你的任务是是一个抽取任务，你要抽取表格中信息来回答问题。请你记住，你输出的信息只能是表格中的内容，你只是抽取相关内容，不能生成无关的数据，不能对数据进行运算。现给你表格信息，请你抽取表格相关内容回答输入问题，输入表格信息：{documents}，输入问题：{query}"
        documents = documents[: maxlen - 1 - len(query) - len(prompt)]
        prompt = prompt.format(documents=documents, query=query)
        for _ in range(2):
            try:
                result = ernie_bot.run(prompt)[0]
                return result["result"]
            except:
                time.sleep(3)
                continue
        return "我暂时不能回答这个方面的问题"


def parsing_QA(api_key=None, secret_key=None, query="", maxlen=11200):
    """
    FistrParsing query to obtain keywords,
    then,text retrieval, and table Q&A
    """
    prompt = "给你年报主要内容的28个标题，具体为\n{}。".format(";".join(all_titles))
    prompt += """现在给你一个问题，你需要理解这个问题，并给出这个问题涉及到公司年报哪些标题，
    请你返回一个json格式的字符串，键值包含"公司名"和"涉及标题"，涉及标题的value是一个列表，
    列表包含这个问题涉及到公司年报哪些标题，
    请你保证你输出的涉及标题不要把27个标题都输出，只输出你认为与问题相关性高的标题。
    下面给你以下示例：
    示例1:
    输入问题：宁德时代近三年发展的怎么样
    输出：{"公司名":宁德时代,"涉及标题":['主营收入','公司主营业务','历年营业总收入、归母净利润、扣非净利润数据','营业收入的主营产品构成情况','历年毛利率和净利率数据']}
    示例2:
    输入问题：宁德时代的股票可以买吗？
    输出：{"公司名":宁德时代,"涉及标题":['公司主营业务','日行情K线数据','近年来市盈率(TTM)、市净率(LF)、市销率(TTM)变化情况','近年来市盈率(TTM)、市净率(LF)、市销率(TTM)历史分位','人均薪酬、创利、创收情况','历年加权净资产收益率','历年研发投入','历年净现比数据','历年主要负债堆积']}
    示例3:
    输入问题：寒武纪的风险点有哪些？
    输出：{"公司名":寒武纪,"涉及标题":['历年营业总收入、归母净利润、扣非净利润数据','公司主营业务','五大客户、供应商集中度情况','前五大客户和供应商占比','近年来市盈率(TTM)、市净率(LF)、市销率(TTM)历史分位','营运能力各周转率','应收账款及减值准备情况','历年存货堆积']}
    现在让我们开始
    输入问题："""
    prompt += query + "\n请你记住你的输出是一个json格式的字符串,键值有两个,请你保证你输出的标题是与问题相关的标题"
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
        index = 0
        for key in query_keys:
            index += 1
            query_text = "请提取" + company + key + "的信息来解答" + query + "这个问题"
            if key in ["主营收入", "公司主营业务"]:
                text += str(index) + "." + text_retrieval(query_text, key) + "\n\n"
            elif key != "日行情K线数据":
                text += str(index) + "." + chat_table(query_text, api_key, secret_key, key) + "\n\n"
    ernie_bot = ErnieBot(api_key, secret_key)
    try:
        prompt = "你现在是金融助手，请你根据背景信息回答问题。请你记住，你的回答要基于背景信息，不要胡编乱造。背景信息{information},请回答相关问题{query}"
        information = text[: maxlen - 1 - len(query) - len(prompt)]
        prompt = prompt.format(information=information, query=query)
        text = "以下是我为这个问题提供的相关信息：\n" + text + "\n\n" + ernie_bot.run(prompt)[0]["result"]
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
    # Single Table Q&A
    result = chat_table(query="比亚迪2022年固定资产是多少", api_key=args.api_key, secret_key=args.secret_key)
    print(result)
    # Single text retrieval
    result = text_retrieval(query="宁德时代的公司主营业务")
    print(result)
    # Multiple Q&A
    result = parsing_QA(query=args.query, api_key=args.api_key, secret_key=args.secret_key)
    print(result)
