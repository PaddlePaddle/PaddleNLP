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

import argparse

search_response = {
    "type": "object",
    "description": "检索结果，内容为论文摘要, 标题以及关键词",
    "properties": {
        "documents": {
            "type": "array",
            "description": "检索结果，内容为论文摘要, 标题以及关键词",
            "items": {
                "type": "object",
                "properties": {
                    "document": {"type": "string", "description": "论文摘要"},
                    "title": {"type": "string", "description": "论文标题"},
                    "key_words": {"type": "string", "description": "论文关键词"},
                },
            },
        }
    },
    "required": ["documents"],
}

functions = [
    {
        "name": "search_multi_paper",
        "description": "根据query, 在论文库内检索最相关的论文",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "论文检索的查询语句"},
                "top_k": {"type": "integer", "description": "论文检索的数量，默认值为3"},
            },
            "required": ["query"],
        },
        "responses": search_response,
        "examples": [
            {"role": "user", "content": "你好，我想了解一下半监督学习这反面的最新的进展。请给我推荐6篇论文。"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "search_multi_paper",
                    "thoughts": "这是一个多篇论文搜索请求。我需要设置query为'半监督学习',检索数量为6",
                    "arguments": '{ "query": "半监督学习", "top_k": 6}',
                },
            },
        ],
    },
    {
        "name": "search_single_paper",
        "description": "根据论文标题定位具体论文, 再通过query检索该篇论文中最相关的内容片段",
        "parameters": {
            "type": "object",
            "description": "根据论文标题定位具体论文, 再通过query检索该篇论文中最相关的内容片段",
            "properties": {
                "query": {"type": "string", "description": "根据输入的query在用户指定的单篇论文上检索最相关的信息"},
                "title": {"type": "string", "description": "论文标题，用于定位单篇论文"},
            },
            "required": ["query", "title"],
        },
        "responses": search_response,
        "examples": [
            {"role": "user", "content": "计算机视觉与神经网络相结合在自动驾驶系统中的应用,这篇文章的主要创新点是什么？"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "search_single_paper",
                    "thoughts": "这是一个单篇论文搜索请求。我需要设置title为'计算机视觉与神经网络相结合在自动驾驶系统中的应用', query为'创新点'",
                    "arguments": '{"query":"创新点", "title":"计算机视觉与神经网络相结合在自动驾驶系统中的应用"}',
                },
            },
        ],
    },
]


def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type", type=str, default="qianfan")
    parser.add_argument("--api_key", type=str, default="", help="The API Key.")
    parser.add_argument("--secret_key", type=str, default="", help="The secret key.")
    parser.add_argument("--bos_ak", type=str, default="", help="The Access Token for uploading files to bos")
    parser.add_argument("--abstract_index_name", default="weipu_abstract_v2", type=str, help="The ann index name")
    parser.add_argument("--full_text_index_name", default="weipu_full_text_v2", type=str, help="The ann index name")
    parser.add_argument("--username", type=str, default="", help="Username of ANN search engine")
    parser.add_argument("--password", type=str, default="", help="Password of ANN search engine")
    parser.add_argument(
        "--retriever_top_k", default=3, type=int, help="Number of element to retrieve from embedding search"
    )
    parser.add_argument(
        "--retriever_batch_size",
        default=16,
        type=int,
        help="The batch size of retriever to extract passage embedding for building ANN index.",
    )
    parser.add_argument("--embedding_dim", default=312, type=int, help="The embedding_dim of index")
    parser.add_argument("--host", type=str, default="localhost", help="host ip of ANN search engine")
    parser.add_argument("--port", type=str, default="9200", help="port of ANN search engine")
    parser.add_argument("--embedding_api_key", default=None, type=str, help="The Embedding API Key.")
    parser.add_argument("--embedding_secret_key", default=None, type=str, help="The Embedding secret key.")
    parser.add_argument("--embed_title", default=False, type=bool, help="The title to be  embedded into embedding")
    parser.add_argument("--serving_name", default="0.0.0.0", help="Serving ip.")
    parser.add_argument("--serving_port", default=8099, type=int, help="Serving port.")
    args = parser.parse_args()
    # yapf: enable
    return args
