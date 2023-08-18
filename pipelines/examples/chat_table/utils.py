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
from pipelines.nodes import (
    DensePassageRetriever,
    ErnieBot,
    PromptTemplate,
    TruncatedConversationHistory,
)
from pipelines.pipelines import Pipeline
from pipelines.schema import Document

logging.getLogger().setLevel(logging.INFO)
index_name = "cropus_base"


def get_json(path):
    data_all = []
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    for keys, value in data.items():
        try:
            assert keys != "hqchartdata"
            content = path.split("/")[-1].replace(".json", "")
            meta = {}
            value_str = str(value)
            column_list = re.findall(r"legend\d*.*?]", value_str)
            assert len(column_list) > 0
            column_name = ""
            for item in column_list:
                item = re.sub(r"legend\d*", "", item)
                column_name += re.sub(r"[\[|\]\:|\']", "", item)
            meta = {"info": value_str}
            data_all.append({"content": content + column_name + value["text"], "meta": meta})
        except:
            continue
    return data_all


def get_all_tables(paths):
    all_tables = []
    for path in paths:
        company_tables = get_json(path)
        all_tables.extend(company_tables)
    return all_tables


# create_index
def create_table_db(index_name, all_tables):
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
    all_tables = retriever.run_indexing(all_tables)[0]["documents"]
    document_store.write_documents(all_tables)
    document_store.save(index_name)


def chat_table(query, history=[], api_key=None, secret_key=None, index=None):
    if index is None:
        index = "document"
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
    indexes = query_pipeline.run(query=query, params={"Retriever": {"top_k": 3, "index": str(index)}})
    content = ""
    for item in indexes["documents"]:
        content = content + item.meta["info"]
    content = Document(content=content, content_type="text", meta={})
    prompt = PromptTemplate(
        "给你一些表格信息，请你根据表格回答我提出的有关信息。如果你无法回答，请输出我无法回答这个问题，请记住你的任务是基于我给出的表格信息给出回答，下面是我给出的表格信息{documents}，我的问题是{query}"
    ).run(query, [content])
    prompt = prompt[0]["query"]
    history = TruncatedConversationHistory(max_length=1000).run(prompt, history)
    ernie_bot = ErnieBot(api_key=api_key, secret_key=secret_key)
    prediction = ernie_bot.run(history[0])
    return prediction[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default="", type=str, help="The API Key.")
    parser.add_argument("--secret_key", default="", type=str, help="The secret key.")
    parser.add_argument("--dirname", default="./", type=str, help="The dirname of json files")
    parser.add_argument("--query", default="比亚迪近五年的无形资产分别是多少", type=str, help="the query")
    args = parser.parse_args()
    files = glob.glob(args.dirname + "/*.json", recursive=True)
    # create_index
    all_tables = get_all_tables(files)
    print(len(all_tables))
    create_table_db(index_name, all_tables)
    # query
    query = "贵州茅台2022年货币资金是多少"
    result = chat_table(query, api_key=args.api_key, secret_key=args.secret_key)
    print(result["result"])
