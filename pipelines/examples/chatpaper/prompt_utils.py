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
                    "uuid": {"type": "string", "description": "uuid"},
                },
            },
        }
    },
    "required": ["documents"],
}

multi_texts = """
根据您的需求，以下是两篇人工智能和机械设计相关的论文：
              标题：大型矿用挖掘机驾驶室工效设计方法及技术研究
              摘要：驾驶室工效设计是大型矿用挖掘机研制的重要环节之一。良好的驾驶室工效是顺利完成挖掘任务，发挥大型矿用挖掘机产品性能的关键。研究大型矿用挖掘机驾驶室工效问题，对提升大型矿用挖掘机的安全可靠性和系统工效，降低驾驶员肌肉骨骼疾患，提高我国大型矿用设备自主设计研发水平以及改善恶劣环境中劳动者健康状况，创建舒适工作空间具有重要意义。
              该论文的UUID是924877fa5e835475d9dc5604ef5aba77，关键词包括人机工程、大型矿用挖掘机、驾驶室工效、布局设计、人机界面等。
              标题：基于AFM的纳米加工深度模型及跨尺度结构加工工艺研究
              摘要：由于具有低成本、操作简单、高精度以及较低环境要求等优势，这种基于AFM纳米机械加工的方法目前被认为是一种简单、可行的纳米级加工技术。但是这种方法还处于初步研究阶段，基于AFM探针刻划的材料去除机理及加工跨尺度纳米结构的工艺方法还有待深入研究，导致目前很难加工出深度可控、大范围的纳米结构。
              该论文的UUID是e41e40149bde487c4c3fb578182f77eb，关键词包括机械加工、纳米结构、加工深度、工艺参数、分子动力学模型、原子力显微镜等。
"""

single_text = """
根据您提供的论文检索工具，以下是两篇与人工智能相关的论文及其摘要：
              论文标题：计算机视觉与神经网络相结合在自动驾驶系统中的应用
              摘要：本文主要研究了计算机视觉与神经网络相结合在自动驾驶系统中的应用。我们针对车外环境感知的不同任务，包括目标检测识别、行人骨架线识别、图像语义分割等，提出了基于深度学习神经网络的方法。
              论文标题：人工智能在医疗诊断中的应用研究
              摘要：近年来，人工智能技术在医疗诊断领域得到了广泛应用。通过对大量医学数据的分析，人工智能能够辅助医生进行更准确、更快速的疾病诊断和治疗方案制定。本文详细介绍了人工智能在医疗诊断中的应用，并探讨了其未来的发展趋势和挑战。
"""

functions = [
    {
        "name": "search_multi_paper",
        "description": "根据query在海量论文库内检索最相关的论文的标题, 内容, uuid以及关键词",
        "parameters": {
            "type": "object",
            "description": "根据输入的query在海量论文库内检索最相关的论文摘要",
            "properties": {"query": {"type": "string", "description": "根据输入的query在海量论文库内检索最相关的论文摘要"}},
            "required": ["query"],
        },
        "responses": search_response,
        "examples": [
            {"role": "user", "content": "半监督论文有哪些?"},
            {
                "role": "assistant",
                "content": "null",
                "function_call": {"name": "search_multi_paper", "arguments": '{ "query": "半监督论文有哪些？"}'},
            },
            {"role": "function", "name": "search_multi_paper", "content": multi_texts},
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
                "content": "null",
                "function_call": {
                    "name": "search_single_paper",
                    "arguments": '{"query":"创新点是什么？", "title":"计算机视觉与神经网络相结合在自动驾驶系统中的应用"}',
                },
            },
            {"role": "function", "name": "search_single_paper", "content": single_text},
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
