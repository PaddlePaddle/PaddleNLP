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

from typing import List, Dict, Any, Tuple, Optional

import os
import logging
import requests
from time import sleep
from uuid import uuid4
import streamlit as st

API_ENDPOINT = os.getenv("API_ENDPOINT")
STATUS = "initialized"
HS_VERSION = "hs_version"
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"


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


def query(query,
          filters={},
          top_k_reader=5,
          top_k_ranker=5,
          top_k_retriever=5) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {
        "filters": filters,
        "Retriever": {
            "top_k": top_k_retriever
        },
        "Ranker": {
            "top_k": top_k_ranker
        },
        "Reader": {
            "top_k": top_k_reader
        }
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
            results.append({
                "context":
                "..." + answer["context"] + "...",
                "answer":
                answer.get("answer", None),
                "source":
                answer["meta"]["name"],
                "relevance":
                round(answer["score"] * 100, 2),
                "document": [
                    doc for doc in response["documents"]
                    if doc["id"] == answer["document_id"]
                ][0],
                "offset_start_in_doc":
                answer["offsets_in_document"][0]["start"],
                "_raw":
                answer,
            })
        else:
            results.append({
                "context": None,
                "answer": None,
                "document": None,
                "relevance": round(answer["score"] * 100, 2),
                "_raw": answer,
            })
    return results, response


def semantic_search(
        query,
        filters={},
        top_k_reader=5,
        top_k_retriever=5) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {
        "filters": filters,
        "Retriever": {
            "top_k": top_k_retriever
        },
        "Ranker": {
            "top_k": top_k_reader
        }
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
        results.append({
            "context": answer["content"],
            "source": answer["meta"]["name"],
            "relevance": round(answer["score"] * 100, 2),
        })
    return results, response


def send_feedback(query, answer_obj, is_correct_answer, is_correct_document,
                  document) -> None:
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
        raise ValueError(
            f"An error was returned [code {response_raw.status_code}]: {response_raw.json()}"
        )


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response = requests.post(url, files=files).json()
    return response


def get_backlink(result) -> Tuple[Optional[str], Optional[str]]:
    if result.get("document", None):
        doc = result["document"]
        if isinstance(doc, dict):
            if doc.get("meta", None):
                if isinstance(doc["meta"], dict):
                    if doc["meta"].get("url", None) and doc["meta"].get(
                            "title", None):
                        return doc["meta"]["url"], doc["meta"]["title"]
    return None, None
