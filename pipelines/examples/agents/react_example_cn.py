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
import glob
import os

from pipelines.agents import Agent, Tool
from pipelines.agents.base import ToolsManager
from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import (
    CharacterTextSplitter,
    DensePassageRetriever,
    DocxToTextConverter,
    FileTypeClassifier,
    PDFToTextConverter,
    PromptNode,
    TextConverter,
    WebRetriever,
)
from pipelines.nodes.prompt.prompt_template import PromptTemplate
from pipelines.pipelines import Pipeline, WebQAPipeline
from pipelines.utils import fetch_archive_from_http

few_shot_prompt = """
你是一个乐于助人、知识渊博的人工智能助手。为了实现正确回答复杂问题的目标，您可以使用以下工具:
搜索: 当你需要用谷歌搜索问题时很有用。你应该问一些有针对性的问题，例如，谁是安东尼·迪雷尔的兄弟？
要回答问题，你需要经历多个步骤，包括逐步思考和选择合适的工具及其输入；工具将以观察作为回应。当您准备好接受最终答案时，回答"最终答案":
示例:
##
问题: 哈利波特的作者是谁？
思考: 让我们一步一步地思考。要回答这个问题，我们首先需要了解哈利波特是什么。
工具: 搜索
工具输入: 哈利波特是什么？
观察: 哈利波特是一系列非常受欢迎的魔幻小说，以及后来的电影和衍生作品。
思考: 我们了解到哈利波特是一系列魔幻小说。现在我们需要找到这些小说的作者是谁。
工具: 搜索
工具输入: 哈利波特的作者是谁？
观察: 哈利波特系列的作者是J.K.罗琳（J.K. Rowling）。
思考: 根据搜索结果，哈利波特系列的作者是J.K.罗琳。所以最终答案是J.K.罗琳。
最终答案: J.K.罗琳
##
问题: {query}
思考:{transcript}
"""

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--index_name", default='dureader_index', type=str, help="The ann index name of ANN.")
parser.add_argument("--search_engine", choices=['faiss', 'milvus'], default="faiss", help="The type of ANN search engine.")
parser.add_argument("--retriever", choices=['dense', 'SerperDev', 'SerpAPI'], default="dense", help="The type of Retriever.")
parser.add_argument("--max_seq_len_query", default=64, type=int, help="The maximum total length of query after tokenization.")
parser.add_argument("--max_seq_len_passage", default=256, type=int, help="The maximum total length of passage after tokenization.")
parser.add_argument("--retriever_batch_size", default=16, type=int, help="The batch size of retriever to extract passage embedding for building ANN index.")
parser.add_argument("--query_embedding_model", default="rocketqa-zh-base-query-encoder", type=str, help="The query_embedding_model path")
parser.add_argument("--passage_embedding_model", default="rocketqa-zh-base-query-encoder", type=str, help="The passage_embedding_model path")
parser.add_argument("--params_path", default="checkpoints/model_40/model_state.pdparams", type=str, help="The checkpoint path")
parser.add_argument("--embedding_dim", default=768, type=int, help="The embedding_dim of index")
parser.add_argument("--search_api_key", default=None, type=str, help="The Serper.dev or SerpAPI key.")
parser.add_argument('--embed_title', default=False, type=bool, help="The title to be  embedded into embedding")
parser.add_argument('--model_type', choices=['ernie_search', 'ernie', 'bert', 'neural_search'], default="ernie", help="the ernie model types")
parser.add_argument('--llm_name', choices=['ernie-bot', 'THUDM/chatglm-6b', "gpt-3.5-turbo", "gpt-4"], default="THUDM/chatglm-6b", help="The chatbot models ")
parser.add_argument("--api_key", default=None, type=str, help="The API Key.")
parser.add_argument("--secret_key", default=None, type=str, help="The secret key.")
args = parser.parse_args()
# yapf: enable


def indexing_files(retriever, document_store, filepaths, chunk_size):
    try:
        text_converter = TextConverter()
        pdf_converter = PDFToTextConverter()
        doc_converter = DocxToTextConverter()

        text_splitter = CharacterTextSplitter(separator="\f", chunk_size=chunk_size, chunk_overlap=0, filters=["\n"])
        pdf_splitter = CharacterTextSplitter(
            separator="\f",
            chunk_size=chunk_size,
            chunk_overlap=0,
            filters=['([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))'],
        )
        file_classifier = FileTypeClassifier()
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_node(component=file_classifier, name="file_classifier", inputs=["File"])
        indexing_pipeline.add_node(component=doc_converter, name="DocConverter", inputs=["file_classifier.output_4"])
        indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["file_classifier.output_1"])
        indexing_pipeline.add_node(component=pdf_converter, name="PDFConverter", inputs=["file_classifier.output_2"])

        indexing_pipeline.add_node(
            component=text_splitter, name="TextSplitter", inputs=["TextConverter", "DocConverter"]
        )
        indexing_pipeline.add_node(component=pdf_splitter, name="PDFSplitter", inputs=["PDFConverter"])
        indexing_pipeline.add_node(component=retriever, name="Retriever", inputs=["TextSplitter", "PDFSplitter"])
        indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])
        files = glob.glob(filepaths + "/*.*", recursive=True)
        indexing_pipeline.run(file_paths=files)
    except Exception as e:
        print(e)
        pass


def get_faiss_retriever(use_gpu):
    faiss_document_store = "faiss_document_store.db"
    if os.path.exists(args.index_name) and os.path.exists(faiss_document_store):
        # connect to existed FAISS Index
        document_store = FAISSDocumentStore.load(args.index_name)
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            passage_embedding_model=args.passage_embedding_model,
            params_path=args.params_path,
            output_emb_size=args.embedding_dim if args.model_type in ["ernie_search", "neural_search"] else None,
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            batch_size=args.retriever_batch_size,
            use_gpu=use_gpu,
            embed_title=args.embed_title,
        )
    else:
        dureader_data = "https://paddlenlp.bj.bcebos.com/applications/dureader_dev.zip"
        zip_dir = "data/dureader_dev"
        fetch_archive_from_http(url=dureader_data, output_dir=zip_dir)

        document_store = FAISSDocumentStore(embedding_dim=args.embedding_dim, faiss_index_factory_str="Flat")
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            passage_embedding_model=args.passage_embedding_model,
            params_path=args.params_path,
            output_emb_size=args.embedding_dim if args.model_type in ["ernie_search", "neural_search"] else None,
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            batch_size=args.retriever_batch_size,
            use_gpu=use_gpu,
            embed_title=args.embed_title,
            top_k=5,
        )
        filepaths = "data/dureader_dev/dureader_dev"
        indexing_files(retriever, document_store, filepaths, chunk_size=500)
        document_store.save(args.index_name)
    return retriever


def search_and_action_example(web_retriever):

    qa_template = PromptTemplate(
        name="文档问答",
        prompt_text="使用以下段落作为来源回答以下问题。"
        "答案应该简短，最多几个字。\n"
        "段落:\n{documents}\n"
        "问题: {query}\n\n"
        "说明: 考虑以上所有段落及其相应的分数，得出答案。 "
        "虽然一个段落可能得分很高， "
        "但重要的是要考虑同一候选答案的所有段落，以便准确回答。\n\n"
        "在考虑了所有的可能性之后，最终答案是:\n",
    )
    pn = PromptNode(
        args.llm_name,
        max_length=512,
        default_prompt_template=qa_template,
        api_key=args.api_key,
        secret_key=args.secret_key,
    )

    pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=pn)

    prompt_node = PromptNode(
        args.llm_name, max_length=512, api_key=args.api_key, secret_key=args.secret_key, stop_words=["观察: "]
    )

    web_qa_tool = Tool(
        name="搜索",
        pipeline_or_node=pipeline,
        description="当你需要用谷歌搜索问题时很有用。",
        output_variable="results",
    )
    few_shot_agent_template = PromptTemplate("few-shot-react", prompt_text=few_shot_prompt)
    # Time to initialize the Agent specifying the PromptNode to use and the Tools
    agent = Agent(
        prompt_node=prompt_node,
        prompt_template=few_shot_agent_template,
        tools_manager=ToolsManager(
            tools=[web_qa_tool],
            tool_pattern=r"工具:\s*(\w+)\s*工具输入:\s*(?:\"([\s\S]*?)\"|((?:.|\n)*))\s*",
            observation_prefix="观察: ",
            llm_prefix="思考: ",
        ),
        max_steps=8,
        final_answer_pattern=r"最终答案\s*:\s*(.*)",
        observation_prefix="观察: ",
        llm_prefix="思考: ",
    )
    hotpot_questions = ["范冰冰的身高是多少?", "武则天传位给了谁？"]
    for question in hotpot_questions:
        result = agent.run(query=question)
        print(f"\n{result['transcript']}")


if __name__ == "__main__":
    if args.retriever == "dense":
        use_gpu = True if args.device == "gpu" else False
        web_retriever = get_faiss_retriever(use_gpu)
    else:
        # https://serper.dev
        web_retriever = WebRetriever(api_key=args.search_api_key, engine="google", top_search_results=2)
    search_and_action_example(web_retriever)
