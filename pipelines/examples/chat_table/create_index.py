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

import glob
import json
import logging
import os
import re
import shutil
import time

from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import DensePassageRetriever, ErnieBot, ErnieRanker
from pipelines.pipelines import Pipeline

logging.getLogger().setLevel(logging.INFO)
index_name = "cropus_base"
from collections import defaultdict

all_tables_dict = defaultdict(list)
columns_titles = {
    "maincwdata": "主营收入",
    "mainbusiness": "公司主营业务",
    "zcchartdata": "主要资产堆积",
    "chchartdata": "历年存货堆积",
    "fzchartdata": "历年主要负债堆积",
    "zlchartdata": "应收账款及减值准备情况",
    "cashchartdata": "历年现金流量",
    "hqchartdata": "日行情K线数据",
    "jxbchartdata": "历年净现比数据",
    "ysjlchartdata": "历年营业总收入、归母净利润、扣非净利润数据",
    "roechartdata": "历年加权净资产收益率",
    "mjlvchartdata": "历年毛利率和净利率数据",
    "fzlvchartdata": "资产负债率数据",
    "tenchartdata": "十大股东数据",
    "choumachartdata": "股档户数和持股情况数据",
    "qijianchartdata": "历年期间费用数据",
    "zycpchartdata": "营业收入的主营产品构成情况",
    "xcchartdata": "人均薪酬、创利、创收情况",
    "liudongdata": "流动速动比率",
    "fcpsrchartdata": "分产品毛利率",
    "top5chartdata": "五大客户、供应商集中度情况",
    "yftrchartdata": "历年研发投入",
    "roicchartdata": "投入资本回报率",
    "zzlchartdata": "营运能力各周转率",
    "freecashchartdata": "自由现金流",
    "sylchartdata": "近年来市盈率(TTM)、市净率(LF)、市销率(TTM)变化情况",
    "hisperdata": "近年来市盈率(TTM)、市净率(LF)、市销率(TTM)历史分位",
    "top5zbchartdata": "前五大客户和供应商占比",
}


def get_json(path):
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    keys = data.keys()
    if "comname" in data:
        comname = data["comname"]
    else:
        comname = path.split("/")[-1].replace(".json", "")
    for keys, value in data.items():
        if keys == "maincwdata":
            maincw = ""
            maincw += "".join([v for k, v in value.items()])
            all_tables_dict["主营收入"].append({"content": maincw, "meta": {}})
        elif keys == "mainbusiness":
            mainbusiness = comname + columns_titles.get(keys, "") + ":" + value
            all_tables_dict["公司主营业务"].append({"content": mainbusiness, "meta": {}})
        elif keys in columns_titles and value != "":
            try:
                meta = {}
                value_str = str(value)
                column_list = re.findall(r"legend\d*.*?]", value_str)
                assert len(column_list) > 0
                column_name = ""
                for item in column_list:
                    item = re.sub(r"legend\d*", "", item)
                    column_name += re.sub(r"[\[|\]\:|\']", "", item)
                title_list = re.findall(r"\'title\':\s*\'.*?\'", value_str)
                title_list = [re.sub(r"[\'title\':\s* |\']", "", item) for item in title_list]
                text_list = re.findall(r"\'text\':\s*\'.*?\'", value_str)
                text_list = [re.sub(r"[\'text\':\s* |\']", "", item) for item in text_list]
                meta = {"info": value_str}
                title = "".join(title_list)
                text = "".join(text_list)
                all_tables_dict["表格"].append(
                    {"content": comname + columns_titles.get(keys, "") + text + title + column_name, "meta": meta}
                )
            except:
                continue
        else:
            continue


def get_all_tables(paths):
    for path in paths:
        get_json(path)


def create_table_db(index_name, tables_dict):
    if os.path.exists("faiss_cropus_store_all.db"):
        os.remove("faiss_cropus_store_all.db")
    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    document_store = FAISSDocumentStore(
        embedding_dim=768,
        faiss_index_factory_str="Flat",
        duplicate_documents="skip",
        sql_url="sqlite:///faiss_cropus_store_all.db",
    )
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
    for key, value in tables_dict.items():
        print(key)
        value = retriever.run_indexing(value)[0]["documents"]
        document_store.write_documents(value, index=str(key))
    document_store.save(index_name)


def get_notabular_information(query, index):
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
    prediction = query_pipeline.run(query=query, params={"Retriever": {"top_k": 1, "index": str(index)}})
    return prediction["documents"][0].content


def chat_table(query, api_key=None, secret_key=None, key="", maxlen=11200):
    index = "表格"
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
            else:
                continue
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
                print(result["result"])
                return result["result"]
            except:
                time.sleep(3)
                continue
        return "我暂时不能回答这个方面的问题"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default="", type=str, help="The API Key.")
    parser.add_argument("--secret_key", default="", type=str, help="The secret key.")
    parser.add_argument("--dirname", default="./", type=str, help="The dirname of json files")
    parser.add_argument("--query", default="比亚迪近五年的无形资产分别是多少", type=str, help="the query")
    args = parser.parse_args()
    files = glob.glob(args.dirname + "/*.json", recursive=True)
    # 建库
    get_all_tables(files)
    create_table_db(index_name, all_tables_dict)
