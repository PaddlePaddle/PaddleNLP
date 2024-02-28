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

from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import DensePassageRetriever

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


def preprocess(path):
    """
    Preprocessing json file
    """
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    keys = data.keys()
    if "comname" in data:
        comname = data["comname"]
    else:
        comname = path.split("/")[-1].replace(".json", "")
    for keys, value in data.items():
        if keys == "maincwdata":
            maincw = comname + columns_titles.get(keys, "") + ":"
            maincw += "".join([v for k, v in value.items()])
            all_tables_dict["text"].append({"content": maincw, "meta": {}})
        elif keys == "mainbusiness":
            mainbusiness = comname + columns_titles.get(keys, "") + ":" + value
            all_tables_dict["text"].append({"content": mainbusiness, "meta": {}})
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
                all_tables_dict["table"].append(
                    {"content": comname + columns_titles.get(keys, "") + text + title + column_name, "meta": meta}
                )
            except:
                continue
        else:
            continue


def get_all_tables(paths):
    """
    Process json files to obtain text and table information
    """
    for path in paths:
        preprocess(path)


def create_index(index_name, tables_dict):
    """
    Creating indexes
    """
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
        value = retriever.run_indexing(value)[0]["documents"]
        document_store.write_documents(value, index=str(key))
    document_store.save(index_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", default="./", type=str, help="The dirname of json files")
    args = parser.parse_args()
    files = glob.glob(args.dirname + "/*.json", recursive=True)
    # 建库
    get_all_tables(files)
    create_index(index_name, all_tables_dict)
