# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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
import logging
import os
import socket
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from pipelines.document_stores import ElasticsearchDocumentStore, MilvusDocumentStore
from pipelines.nodes import DensePassageRetriever
from pipelines.utils import convert_files_to_dicts, launch_es

API_ENDPOINT = os.getenv("API_ENDPOINT")
STATUS = "initialized"
HS_VERSION = "hs_version"
DOC_REQUEST = "query"
DOC_REQUEST_CHATFILE = "chatfile_query"
FILE_REQUEST = "query_images"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"
DOC_UPLOAD_SPLITTER = "file-upload-splitter"
DOC_PARSE = "files"
IMAGE_REQUEST = "query_text_to_images"
QA_PAIR_REQUEST = "query_qa_pairs"
FILE_UPLOAD_QA_GENERATE = "file-upload-qa-generate"


def pipelines_is_ready():
    """
    Used to show the "pipelines is loading..." message
    """
    url = f"{API_ENDPOINT}/{STATUS}"
    try:
        if requests.get(url).status_code < 400:
            return True
    except Exception as e:
        logging.exception(e)
        sleep(1)  # To avoid spamming a non-existing endpoint at startup
    return False


@st.cache
def pipelines_version():
    """
    Get the pipelines version from the REST API
    """
    url = f"{API_ENDPOINT}/{HS_VERSION}"
    return requests.get(url, timeout=0.1).json()["hs_version"]


def pipelines_files(file_name):
    """
    Get the pipelines files from the REST API
    # http://server_ip:server_port/files?file_name=8f6435d7ff1f1913dbcd74feb47e2fdb_0.png
    """
    server_ip = socket.gethostbyname(socket.gethostname())
    server_port = API_ENDPOINT.split(":")[-1]
    url = f"http://{server_ip}:{server_port}/files?file_name={file_name}"
    return url


def query(
    query, filters={}, top_k_reader=5, top_k_ranker=5, top_k_retriever=5
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {
        "filters": filters,
        "Retriever": {"top_k": top_k_retriever},
        "Ranker": {"top_k": top_k_ranker},
        "Reader": {"top_k": top_k_reader},
    }
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    # Format response
    results = []
    answers = response["answers"]
    for answer in answers:
        if answer.get("answer", None):
            results.append(
                {
                    "context": "..." + answer["context"] + "...",
                    "answer": answer.get("answer", None),
                    "source": answer["meta"]["name"],
                    "relevance": round(answer["score"] * 100, 2),
                    "document": [doc for doc in response["documents"] if doc["id"] == answer["document_id"]][0],
                    "offset_start_in_doc": answer["offsets_in_document"][0]["start"],
                    "_raw": answer,
                }
            )
        else:
            results.append(
                {
                    "context": None,
                    "answer": None,
                    "document": None,
                    "relevance": round(answer["score"] * 100, 2),
                    "_raw": answer,
                }
            )
    return results, response


def multi_recall_semantic_search(
    query, filters={}, top_k_ranker=5, top_k_bm25_retriever=5, top_k_dpr_retriever=5
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {
        "filters": filters,
        "DenseRetriever": {"top_k": top_k_dpr_retriever},
        "BMRetriever": {"top_k": top_k_bm25_retriever},
        "Ranker": {"top_k": top_k_ranker},
    }
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    # Format response
    results = []
    answers = response["documents"]
    for answer in answers:
        results.append(
            {
                "context": answer["content"],
                "source": answer["meta"]["name"],
                "answer": answer["meta"]["answer"] if "answer" in answer["meta"].keys() else "",
                "relevance": round(answer["score"] * 100, 2),
                "images": answer["meta"]["images"] if "images" in answer["meta"] else [],
            }
        )
    return results, response


def semantic_search(
    query, filters={}, top_k_reader=5, top_k_retriever=5
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {"filters": filters, "Retriever": {"top_k": top_k_retriever}, "Ranker": {"top_k": top_k_reader}}
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    # Format response
    results = []
    answers = response["documents"]
    for answer in answers:
        results.append(
            {
                "context": answer["content"],
                "source": answer["meta"]["name"],
                "answer": answer["meta"]["answer"] if "answer" in answer["meta"].keys() else "",
                "relevance": round(answer["score"] * 100, 2),
                "images": answer["meta"]["images"] if "images" in answer["meta"] else [],
            }
        )
    return results, response


def ChatFile(
    query,
    filters={},
    top_k_reader=5,
    top_k_retriever=5,
    pooling_mode="mean_tokens",
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
):
    url = f"{API_ENDPOINT}/{DOC_REQUEST_CHATFILE}"
    if api_key is not None and api_key != " " and secret_key is not None and secret_key != " ":
        params = {
            "filters": filters,
            "Retriever": {
                "top_k": top_k_retriever,
                "pooling_mode": pooling_mode,
            },
            "Ranker": {"top_k": top_k_reader},
            "ErnieBot": {"api_key": api_key, "secret_key": secret_key},
        }
    else:
        params = {
            "filters": filters,
            "Retriever": {
                "top_k": top_k_retriever,
                "pooling_mode": pooling_mode,
            },
            "Ranker": {"top_k": top_k_reader},
        }
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))
    return response


def text_to_image_search(
    query, resolution="1024*1024", top_k_images=5, style="探索无限"
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a prompt text and corresponding parameters to the REST API
    """
    url = f"{API_ENDPOINT}/{IMAGE_REQUEST}"
    params = {
        "TextToImageGenerator": {
            "style": style,
            "topk": top_k_images,
            "resolution": resolution,
        }
    }
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))
    results = response["answers"]
    return results, response


def image_text_search(query, filters={}, top_k_retriever=5) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {"filters": filters, "Retriever": {"top_k": top_k_retriever}}
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    # Format response
    results = []
    answers = response["documents"]
    for answer in answers:
        results.append(
            {
                "context": answer["content"],
                "relevance": round(answer["meta"]["es_ann_score"] * 100, 2),
            }
        )
    return results, response


def image_to_text_search(file, filters={}, top_k_retriever=5) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"{API_ENDPOINT}/{FILE_REQUEST}"
    # {"Retriever": {"top_k": 2, "query_type":"image"}}
    params = {"filters": filters, "Retriever": {"top_k": top_k_retriever, "query_type": "image"}}
    req = {"meta": json.dumps(params)}
    files = [("files", file)]

    response = requests.post(url, files=files, data=req, verify=False).json()
    return response


def text_to_qa_pair_search(query, is_filter=True) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a prompt text and corresponding parameters to the REST API
    """
    url = f"{API_ENDPOINT}/{QA_PAIR_REQUEST}"
    params = {
        "QAFilter": {
            "is_filter": is_filter,
        },
    }

    req = {"meta": [query], "params": params}
    response_raw = requests.post(url, json=req)
    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))
    results = response["filtered_cqa_triples"]
    return results, response


def send_feedback(query, answer_obj, is_correct_answer, is_correct_document, document) -> None:
    """
    Send a feedback (label) to the REST API
    """
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    req = {
        "query": query,
        "document": document,
        "is_correct_answer": is_correct_answer,
        "is_correct_document": is_correct_document,
        "origin": "user-feedback",
        "answer": answer_obj,
    }
    response_raw = requests.post(url, json=req)
    if response_raw.status_code >= 400:
        raise ValueError(f"An error was returned [code {response_raw.status_code}]: {response_raw.json()}")


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response = requests.post(url, files=files).json()
    return response


def upload_chatfile(file, chunk_size: int = 300, separator: str = "\n", filters: list = ["\n"]):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD_SPLITTER}"
    params = {
        "DocxSplitter": {"filters": filters, "chunk_size": chunk_size},
        "MarkdownSplitter": {"filters": filters, "chunk_size": chunk_size},
        "TextSplitter": {"filters": filters, "chunk_size": chunk_size, "separator": separator},
        "PDFSplitter": {"filters": filters, "chunk_size": chunk_size, "separator": separator},
        "ImageSplitter": {"filters": filters, "chunk_size": chunk_size, "separator": separator},
    }

    files = [("files", file)]
    req = {"meta": json.dumps(params)}
    response = requests.post(url, data=req, files=files, verify=False).json()
    return response


def file_upload_qa_generate(file):
    url = f"{API_ENDPOINT}/{FILE_UPLOAD_QA_GENERATE}"
    files = [("files", file)]
    response = requests.post(url, files=files).json()
    return response


def get_backlink(result) -> Tuple[Optional[str], Optional[str]]:
    if result.get("document", None):
        doc = result["document"]
        if isinstance(doc, dict):
            if doc.get("meta", None):
                if isinstance(doc["meta"], dict):
                    if doc["meta"].get("url", None) and doc["meta"].get("title", None):
                        return doc["meta"]["url"], doc["meta"]["title"]
    return None, None


def offline_ann(
    index_name,
    doc_dir,
    search_engine="elastic",
    host="127.0.0.1",
    port="9200",
    query_embedding_model="rocketqa-zh-nano-query-encoder",
    passage_embedding_model="rocketqa-zh-nano-para-encoder",
    params_path="checkpoints/model_40/model_state.pdparams",
    embedding_dim=312,
    split_answers=True,
):
    if search_engine == "milvus":
        document_store = MilvusDocumentStore(
            embedding_dim=embedding_dim,
            host=host,
            index=index_name,
            port=port,
            index_param={"M": 16, "efConstruction": 50},
            index_type="HNSW",
        )
    else:
        launch_es()
        document_store = ElasticsearchDocumentStore(
            host=host, port=port, username="", password="", embedding_dim=embedding_dim, index=index_name
        )
    # 将每篇文档按照段落进行切分
    dicts = convert_files_to_dicts(
        dir_path=doc_dir, split_paragraphs=True, split_answers=split_answers, encoding="utf-8"
    )

    print(dicts[:3])

    # 文档数据写入数据库
    document_store.write_documents(dicts)

    # 语义索引模型
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=query_embedding_model,
        passage_embedding_model=passage_embedding_model,
        params_path=params_path,
        output_emb_size=embedding_dim,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        batch_size=1,
        use_gpu=True,
        embed_title=False,
    )

    # 建立索引库
    document_store.update_embeddings(retriever)
