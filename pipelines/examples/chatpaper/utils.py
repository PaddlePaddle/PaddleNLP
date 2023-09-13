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
import logging
import os
from typing import Optional

import fitz
import requests

from pipelines.document_stores import BaiduElasticsearchDocumentStore
from pipelines.nodes import EmbeddingRetriever, ErnieRanker
from pipelines.pipelines import Pipeline

logging.getLogger().setLevel(logging.INFO)
import time
from functools import partial
from multiprocessing import Pool

from pipelines.nodes import ErnieBot
from pipelines.nodes.combine_documents import (
    MapReduceDocuments,
    ReduceDocuments,
    StuffDocuments,
)
from pipelines.nodes.preprocessor.text_splitter import SpacyTextSplitter


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


def pdf2image_index(start, end, pdfPath, imgPath, zoom_x=10, zoom_y=10, rotation_angle=0):
    pdf = fitz.open(pdfPath)
    image_path = []
    for index in range(start, end):
        page = pdf[index]
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotation_angle)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        pm._writeIMG(imgPath + "/" + str(index) + ".png", format=1)
        image_path.append((imgPath + "/" + str(index) + ".png", "page:" + str(index)))
    return image_path


def pdf2image(pdfPath, imgPath, zoom_x=10, zoom_y=10, rotation_angle=0, number_process_page=5):
    """
    Convert PDF to Image
    """
    pdf = fitz.open(pdfPath)
    image_path = []
    if pdf.page_count % number_process_page == 0:
        number_process = pdf.page_count // number_process_page
    else:
        number_process = pdf.page_count // number_process_page + 1
    number_process = min(number_process, os.cpu_count())
    pool = Pool(processes=number_process)
    index_list = [i for i in range(0, pdf.page_count, number_process_page)]
    if index_list[-1] < pdf.page_count:
        index_list.append(pdf.page_count)
    print(number_process)
    func = partial(
        pdf2image_index, pdfPath=pdfPath, imgPath=imgPath, zoom_x=zoom_x, zoom_y=zoom_y, rotation_angle=rotation_angle
    )
    result = pool.starmap(func, [(start, end) for start, end in zip(index_list, index_list[1:])])
    pool.close()
    pool.join()
    pdf.close()
    for item in result:
        image_path.extend(item)
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


# Summary of a paper
def summarize_collapse(text_list, api_key, secret_key):
    document_prompt = "这是一篇论文摘要的第{index}部分的内容：{content}"
    llm_prompt = """
    我需要你的帮助来阅读和总结输入论文摘要的主要内容，请你使用中文输出。
    根据以下四点进行总结。(专有名词需要用英语标记）
    （1）：论文研究背景是什么？
    （2）：过去的方法是什么？他们有什么问题？这种方法是否有良好的动机？
    （3）：论文提出的研究方法是什么？
    （4）：在什么任务上，通过论文的方法实现了什么性能？表现能支持他们的目标吗？
    （5）意义是什么？
    （6）从创新点、绩效和工作量三个维度总结论文优势和劣势
    语句尽可能简洁和学术，不要有太多重复的信息，数值使用原始数字，请你使用中文输出
    输入论文摘要:{}
    总结输出:
    """
    combine_documents = StuffDocuments(
        api_key=api_key, secret_key=secret_key, llm_prompt=llm_prompt, document_prompt=document_prompt
    )
    reduce_documents = ReduceDocuments(combine_documents=combine_documents)
    MapReduce = MapReduceDocuments(
        api_key=api_key, secret_key=secret_key, llm_prompt=llm_prompt, reduce_documents=reduce_documents
    )
    summary = MapReduce.run(text_list)
    return summary[0]["result"]


def summarize_abstract(abstract, api_key, secret_key, chunk_size=300, max_token=4200):
    llm_prompt = """
    我需要你的帮助来阅读和总结输入论文摘要的主要内容，请你使用中文输出。
    根据以下四点进行总结。(专有名词需要用英语标记）
    （1）：论文研究背景是什么？
    （2）：过去的方法是什么？他们有什么问题？这种方法是否有良好的动机？
    （3）：论文提出的研究方法是什么？
    （4）：在什么任务上，通过论文的方法实现了什么性能？表现能支持他们的目标吗？
    （5）意义是什么？
    （6）从创新点、绩效和工作量三个维度总结论文优势和劣势
    语句尽可能简洁和学术，不要有太多重复的信息，数值使用原始数字，请你使用中文输出
    输入论文摘要:{}
    总结输出:
    """
    if len(llm_prompt.format(abstract)) > max_token:
        file_splitter_chinese = SpacyTextSplitter(chunk_size=chunk_size, separator="\n", chunk_overlap=0)
        txt_split = file_splitter_chinese.split_text(abstract)
        txt_list = []
        for split in txt_split:
            txt_list.append({"content": split, "meta": {}})
        summary = summarize_collapse(txt_list, api_key, secret_key)
    else:
        ernie_bot = ErnieBot(api_key=api_key, secret_key=secret_key, model_name="ERNIE-Bot")
        summary = ernie_bot.run(llm_prompt.format(abstract))[0]["result"]
    return summary.replace("\n\n", "\n")


# Summary of multiple papers
def merge_summary(text_list, api_key, secret_key):
    document_prompt = "输入的第{index}论文的内容：{content}"
    llm_prompt = """你需要完成多篇论文总结任务，不要分别进行单篇论文总结。
    我需要你的帮助来总结一下多篇论文在背景、研究方法、数据集、结论这四个方面的共同之处和不同之处。
    输入的多篇论文:{}
    总结输出:
    """
    sum_prompt = "总结输入的论文摘要，保留主要内容，论文摘要:{}"
    combine_documents = StuffDocuments(
        api_key=api_key, secret_key=secret_key, llm_prompt=llm_prompt, document_prompt=document_prompt
    )
    reduce_documents = ReduceDocuments(combine_documents=combine_documents)
    MapReduce = MapReduceDocuments(
        api_key=api_key, secret_key=secret_key, llm_prompt=sum_prompt, reduce_documents=reduce_documents
    )
    summary = MapReduce.run(text_list)
    return summary[0]["result"]


# translation
dict_l = {"中文": "英文", "英文": "中文"}


def ernie_bot_translation(prompt, api_key, secret_key, cycle_num=3, key="文本翻译"):
    ernie_bot = ErnieBot(api_key=api_key, secret_key=secret_key, model_name="ERNIE-Bot")
    i = 0
    while i < cycle_num:
        try:
            txt = ernie_bot.run(prompt)[0]["result"]
            return str(txt)
        except:
            i += 1
            time.sleep(0.5)
    return None


def translate_part(text, api_key, secret_key, task="翻译", max_length=10000, lang="中文", chunk_size=1000, cycle_num=3):
    if lang == "中文":
        file_splitter = SpacyTextSplitter(chunk_size=chunk_size, separator="\n", chunk_overlap=0)
    elif lang == "英文":
        file_splitter = SpacyTextSplitter(
            chunk_size=chunk_size, separator="\n", pipeline="en_core_web_sm", chunk_overlap=0
        )
    text = text.replace("\n\n", "\n")
    prompt_all = """
        你现在是一个翻译助手。你需要将输入的{l_s}内容翻译为{l_t}，你必须保证完成了将{l_s}内容翻译为{l_t}内容的任务。
        下面让我们正式开始：输入的{l_s}内容为：{content}
        你必须保证完成了将{l_s}内容翻译为{l_t}内容的任务，并输出翻译结果。
        翻译结果：
        """
    if len(prompt_all.format(content=text, l_s=lang, l_t=dict_l[lang])) > max_length:
        documents = file_splitter.split_text(text)
        txt = ""
        for split in documents:
            txt_split = ernie_bot_translation(
                prompt_all.format(content=split, l_s=lang, l_t=dict_l[lang]),
                api_key=api_key,
                secret_key=secret_key,
                cycle_num=cycle_num,
                key="文本翻译",
            )
            if txt_split:
                txt += txt_split
            else:
                txt += split
    else:
        txt = ernie_bot_translation(
            prompt_all.format(content=text, l_s=lang, l_t=dict_l[lang]),
            api_key=api_key,
            secret_key=secret_key,
            cycle_num=cycle_num,
            key="文本翻译",
        )
        if not txt:
            txt = text
    return txt


def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type", type=str, default="qianfan")
    parser.add_argument("--api_key", type=str, default="", help="The API Key.")
    parser.add_argument("--secret_key", type=str, default="", help="The secret key.")
    parser.add_argument("--bos_ak", type=str, default="", help="The Access Token for uploading files to bos")
    parser.add_argument("--bos_sk", type=str, default="", help="The Secret Token for uploading files to bos")
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.7,
        help="The range is between 0 and 1.The smaller the parameter, the more stable the generated result. When it is 0, randomness is minimized",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.95,
        help="The smaller the parameter, the more stable the generated result",
    )
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum number of generated tokens")
    parser.add_argument("--ernie_model", type=str, default="ernie-bot-3.5", help="Model type")
    parser.add_argument("--system_prompt", type=str, default="你是我的AI助理。", help="System settings for dialogue models")
    parser.add_argument("--es_host", type=str, default="", help="the host of es")
    parser.add_argument("--es_port", type=int, default=8309, help="the port of es")
    parser.add_argument("--es_username", type=str, default="", help="the username of es")
    parser.add_argument("--es_password", type=str, default="", help="the password of es")
    parser.add_argument("--es_index_abstract", type=str, default="", help="the index of abstracts")
    parser.add_argument("--es_index_full_text", type=str, default="", help="the index of all papers")
    parser.add_argument("--es_chunk_size", type=int, default=500, help="the size of chunk in es")
    parser.add_argument("--es_thread_count", type=int, default=30, help="the thread count in es")
    parser.add_argument("--es_queue_size", type=int, default=30, help="the size of queue in es")
    parser.add_argument("--retriever_batch_size", type=int, default=16, help="the batch size of retriever ")
    parser.add_argument("--retriever_api_key", type=str, default="", help="the api key of retriever")
    parser.add_argument("--retriever_secret_key", type=str, default="", help="the secret key of retriever")
    parser.add_argument(
        "--retriever_embed_title", type=bool, default=False, help="whether use embedding title in retriever"
    )
    parser.add_argument("--retriever_threshold", type=float, default=0.95, help="the threshold of retriever")
    parser.add_argument(
        "--txt_file", type=str, default="", help="the path of a txt file which includes all papers path"
    )
    parser.add_argument("--max_token", type=int, default=11200, help=" the max number of tokens of LLM")
    parser.add_argument("--translation_chunk_size", type=int, default=300, help="the chunk size of translation")
    parser.add_argument("--translation_cycle_num", type=int, default=3, help="fault tolerance times")
    parser.add_argument(
        "--translation_max_token", type=int, default=500, help="the max number of tokens of translation segment"
    )
    parser.add_argument("--serving_name", default="0.0.0.0", help="Serving ip.")
    parser.add_argument("--serving_port", default=8099, type=int, help="Serving port.")
    parser.add_argument(
        "--number_process_page", default=5, type=int, help="the number of PDF pages processed per process"
    )
    args = parser.parse_args()
    return args
