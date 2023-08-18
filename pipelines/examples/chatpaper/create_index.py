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
import logging
import multiprocessing
import os
import re
import shutil
from functools import partial
from multiprocessing import Manager, Pool

from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import (
    DensePassageRetriever,
    ErnieBot,
    ErnieRanker,
    PDFToTextConverter,
    PromptTemplate,
    TruncatedConversationHistory,
)
from pipelines.nodes.combine_documents import (
    MapReduceDocuments,
    ReduceDocuments,
    StuffDocuments,
)
from pipelines.nodes.preprocessor.text_splitter import (
    CharacterTextSplitter,
    SpacyTextSplitter,
)
from pipelines.pipelines import Pipeline

logging.getLogger().setLevel(logging.INFO)
manager = Manager()
all_data_result = manager.dict()
document_abs = manager.list()
index_name = "knowledge_base_all"


def summary_confine(text_list, api_key, secret_key):
    document_prompt = "这是一篇论文摘要的第{index}部分的内容：{content}"
    llm_prompt = "我需要你的帮助来阅读和总结以下问题{}\n1.标记论文关键词\n根据以下四点进行总结。(专有名词需要用英语标记）\n-（1）：论文研究背景是什么？\n-（2）：过去的方法是什么？他们有什么问题？这种方法是否有良好的动机？\n-（3）：论文提出的研究方法是什么？\n-（4）：在什么任务上，通过论文的方法实现了什么性能？表现能支持他们的目标吗？\n-（5）意义是什么？\n-（6）从创新点、绩效和工作量三个维度总结论文优势和劣势\n请遵循以下输出格式：\n1.关键词：xxx\n\n2.摘要：\n\n8.结论：\n\nxxx；\n创新点：xxx；业绩：xxx；工作量：xxx；\n语句尽可能简洁和学术，不要有太多重复的信息，数值使用原始数字，一定要严格遵循格式，将相应的内容输出到xxx，按照\n换行"
    sum_prompt = "总结输入的内容，保留主要内容，输入内容:{}"
    combine_documents = StuffDocuments(
        api_key=api_key, secret_key=secret_key, llm_prompt=llm_prompt, document_prompt=document_prompt
    )
    reduce_documents = ReduceDocuments(combine_documents=combine_documents)
    MapReduce = MapReduceDocuments(
        api_key=api_key, secret_key=secret_key, llm_prompt=sum_prompt, reduce_documents=reduce_documents
    )
    summary = MapReduce.run(text_list)
    return summary[0]["result"]


def clean(txt, filters):
    for special_character in filters:
        txt = txt.replace(special_character, "")
    return txt


def chatfile_base(indexes, query, api_key, secret_key, history=[]):
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
        duplicate_documents="skip",
    )
    ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=True)
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
    all_doc = []
    for index in indexes:
        doc = query_pipeline.run(
            query=query, params={"Retriever": {"top_k": 30, "index": index}, "Ranker": {"top_k": 2}}
        )
        all_doc.extend(doc["documents"])
    prompt = PromptTemplate("背景：{documents} 问题：{query}").run(query, all_doc)
    prompt = prompt[0]["query"]
    history = TruncatedConversationHistory(max_length=1000).run(prompt, history)
    ernie_bot = ErnieBot(api_key=api_key, secret_key=secret_key)
    prediction = ernie_bot.run(history[0])
    return prediction[0]


def get_summary(path, api_key, secret_key, filters=["\n"]):
    document_paper = []
    try:
        pdf_converter = PDFToTextConverter()
        content = pdf_converter.convert(path, meta=None, encoding="Latin1")[0]["content"].replace("\n", "")
        try:
            pdf_splitter = SpacyTextSplitter(separator="\n", filters="\n", chunk_size=500, chunk_overlap=0)
            content_split = pdf_splitter.split_text(content)
        except:
            pdf_splitter = CharacterTextSplitter(separator="\n", filters="\n", chunk_size=500, chunk_overlap=0)
            content_split = pdf_splitter.split_text(content)
        for item in content_split:
            item = clean(item, filters)
            document_paper.append({"content": item, "meta": {"name": path}})
        index1 = re.search("摘\s*?要", content, re.IGNORECASE).span()[0]
        index2 = re.search("ABSTRACT", content, flags=re.I).span()[0]
        assert index2 > index1
        content_abs = re.sub(r"摘\s*?要|\f|\r", "", content[index1:index2])
        try:
            pdf_splitter = SpacyTextSplitter(separator="\n", filters="\n", chunk_size=500, chunk_overlap=0)
            content_split_abs = pdf_splitter.split_text(content_abs)
        except:
            pdf_splitter = CharacterTextSplitter(separator="\n", filters="\n", chunk_size=500, chunk_overlap=0)
            content_split_abs = pdf_splitter.split_text(content_abs)
        paper_abs = []
        for item in content_split_abs:
            item = clean(item, filters)
            paper_abs.append({"content": item, "meta": {"name": path}})
        paper_sum = summary_confine(paper_abs, api_key, secret_key)
        document_abs.append({"content": paper_sum, "meta": {"name": path}})
    except:
        return None
    return document_paper, path


def mul_tackle(p_m, path_list, api_key, secret_key, filters=["\n"]):
    func = partial(get_summary, api_key=api_key, secret_key=secret_key, filters=filters)
    pool = Pool(processes=min(p_m, multiprocessing.cpu_count()))
    result = pool.map(func, path_list)
    pool.close()
    pool.join()
    return result


def bulid_base(paths, api_key, secret_key, filters=["\n"]):
    if os.path.exists("faiss_base_store_all.db"):
        os.remove("faiss_base_store_all.db")
    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    document_store = FAISSDocumentStore(
        embedding_dim=768,
        faiss_index_factory_str="Flat",
        sql_url="sqlite:///faiss_base_store_all.db",
        duplicate_documents="skip",
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
        duplicate_documents="skip",
    )
    results = mul_tackle(1, paths, api_key=api_key, secret_key=secret_key, filters=filters)
    results = [item for item in results if item is not None]
    for split, path in results:
        index = path.split("/")[-1].replace(".pdf", "")
        split = retriever.run_indexing(split)[0]["documents"]
        document_store.write_documents(split, index=str(index))
    document_abs_embed = retriever.run_indexing(document_abs)[0]["documents"]
    document_store.write_documents(document_abs_embed)
    document_store.save(index_name)


def chat_papers(query, api_key, secret_key, retriever_top=30, ranker_top=3, history=[]):
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
    prediction = query_pipeline.run(query=query, params={"Retriever": {"top_k": 30}, "Ranker": {"top_k": 3}})
    paths = [item.meta["name"] for item in prediction["documents"] if os.path.isfile(item.meta["name"])]
    indexes = [path.split("/")[-1].replace(".pdf", "") for path in paths]
    result = chatfile_base(indexes, query=query, api_key=api_key, secret_key=secret_key, history=history)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default="", type=str, help="The API Key.")
    parser.add_argument("--secret_key", default="", type=str, help="The secret key.")
    parser.add_argument("--dirname", default="./", type=str, help="The dirname of PDF files")
    args = parser.parse_args()
    dirname = args.dirname
    files = glob.glob(dirname + "/*/*.pdf", recursive=True)
    bulid_base(files, api_key=args.api_key, secret_key=args.secret_key)
    result = chat_papers(query="商业银行薪酬制度的政策效应", api_key=args.api_key, secret_key=args.secret_key)
