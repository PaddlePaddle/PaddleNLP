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
from typing import Optional

import fitz
import requests

from pipelines.document_stores import BaiduElasticsearchDocumentStore
from pipelines.nodes import EmbeddingRetriever, ErnieRanker
from pipelines.pipelines import Pipeline


def load_all_json_path(path):
    json_path = {}
    with open(path, encoding="utf-8", mode="r") as f:
        for line in f:
            try:
                json_id, json_name = line.strip().split()
                json_path[json_id] = json_name
            except:
                continue
    return json_path


def pdf2image(pdfPath, imgPath, zoom_x=10, zoom_y=10, rotation_angle=0):
    """
    Convert PDF to Image
    """
    pdf = fitz.open(pdfPath)
    image_path = []
    for pg in range(0, pdf.page_count):
        page = pdf[pg]
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotation_angle)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        pm._writeIMG(imgPath + "/" + str(pg) + ".png", format=1)
        image_path.append((imgPath + "/" + str(pg) + ".png", "page:" + str(pg)))
    pdf.close()
    return image_path


def _apply_token(api_key, secret_key):
    """
    Gererate an access token.
    """
    payload = ""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token_host = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    response = requests.request("POST", token_host, headers=headers, data=payload)
    if response:
        res = response.json()
    else:
        raise RuntimeError("Request access token error.")
    return res["access_token"]


def get_shown_context(context):
    """Get gradio chatbot."""
    shown_context = []
    for turn_idx in range(0, len(context), 2):
        shown_context.append([context[turn_idx]["content"], context[turn_idx + 1]["content"]])
    return shown_context


def tackle_history(history=[]):
    messages = []
    if len(history) < 3:
        return messages
    for turn_idx in range(2, len(history)):
        messages.extend(
            [{"role": "user", "content": history[turn_idx][0]}, {"role": "assistant", "content": history[turn_idx][1]}]
        )
    return messages


def retrieval(
    query: str,
    file_id: Optional[str] = None,
    es_host: str = "",
    es_port: int = "",
    es_username: str = "",
    es_password: str = "",
    es_index: str = "",
    es_chunk_size: int = 500,
    es_thread_count: int = 30,
    es_queue_size: int = 30,
    retriever_batch_size: int = 16,
    retriever_api_key: str = "",
    retriever_embed_title: bool = False,
    retriever_secret_key: str = "",
    retriever_topk: int = 30,
    rank_topk: int = 5,
):
    """
    :param query: The query
    :param file_id: The id of a file
    :param es_host: The host of es
    :param es_por: The host of es
    :param es_username: The username of es
    :param es_password: The password of es
    :param es_index: The index of es
    :param es_chunk_size: The chunk size of es
    :param es_thread_count: The thread count of es
    :param es_queue_size: The queue size of es
    :param retriever_batch_size: The batch_size of retriever
    :param retriever_api_key: The api_key of retriever
    :param retriever_embed_title: The embed_title of retriever
    :param retriever_secret_key: The secret_key of retriever
    :param retriever_topk: How many documents to return per query in  .
    :param rank_topk: The maximum number of documents to return in rank.
    """
    ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=True)
    document_store = BaiduElasticsearchDocumentStore(
        embedding_dim=384,
        duplicate_documents="skip",
        host=es_host,
        port=es_port,
        username=es_username,
        password=es_password,
        index=es_index,
        similarity="dot_prod",
        vector_type="bpack_vector",
        search_fields=["content", "meta"],
        chunk_size=es_chunk_size,
        thread_count=es_thread_count,
        queue_size=es_queue_size,
    )
    retriever = EmbeddingRetriever(
        document_store=document_store,
        retriever_batch_size=retriever_batch_size,
        api_key=retriever_api_key,
        embed_title=retriever_embed_title,
        secret_key=retriever_secret_key,
    )
    if not file_id:
        query_pipeline = Pipeline()
        query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
        prediction = query_pipeline.run(
            query=query, params={"Retriever": {"top_k": retriever_topk}, "Ranker": {"top_k": rank_topk}}
        )
        return prediction
    else:
        filters = {
            "$and": {
                "id": {"$eq": file_id},
            }
        }
        query_pipeline = Pipeline()
        query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
        prediction = query_pipeline.run(
            query=query,
            params={"Retriever": {"top_k": retriever_topk, "filters": filters}, "Ranker": {"top_k": rank_topk}},
        )
        return prediction
