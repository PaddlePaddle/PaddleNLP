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

import os
import time

import arxiv
import gradio as gr

# from ulits import  single_paper_sum
from utils import merge_summary, pdf2image, single_paper_abs_sum, translation

from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import (
    DensePassageRetriever,
    ErnieBot,
    ErnieRanker,
    PromptTemplate,
    TruncatedConversationHistory,
)

os.environ["no_proxy"] = "localhost,10.9.189.4,::1"
from create_base import chat_papers

from pipelines.pipelines import Pipeline

paper_all = []
index_name = "dureader_index"
import multiprocessing
from multiprocessing import Manager, Pool

manager = Manager()
all_data_result = manager.dict()
papers_sum = manager.list()


def clear_session():
    global all_data_result
    global paper_all
    all_data_result = {}
    paper_all = []
    return None, "https://arxiv.org/abs/2303.08774"


def chat_file(
    query,
    history=None,
    index_paper=None,
    api_key="",
    secret_key="",
):
    if history is None:
        history = []
    if index_paper is None:
        index = "document"
    else:
        index = index_paper.split("/")[-1].replace(".pdf", "").replace(".", "_")
    document_store = FAISSDocumentStore.load(index_name)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="moka-ai/m3e-base",
        passage_embedding_model="moka-ai/m3e-base",
        params_path="checkpoints/model_40/model_state.pdparams",
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
    query_pipeline.add_node(component=PromptTemplate("背景：{documents} 问题：{query}"), name="Template", inputs=["Ranker"])
    query_pipeline.add_node(
        component=TruncatedConversationHistory(max_length=256), name="TruncateHistory", inputs=["Template"]
    )
    ernie_bot = ErnieBot(api_key=api_key, secret_key=secret_key)
    query_pipeline.add_node(component=ernie_bot, name="ErnieBot", inputs=["TruncateHistory"])
    prediction = query_pipeline.run(
        query=query, params={"Retriever": {"top_k": 30, "index": str(index)}, "Ranker": {"top_k": 3}}
    )
    history.append(["user: {}".format(query), "assistant: {}".format(prediction["result"])])
    return "", history, history


def tackle_paper(root_path, path, api_key, secret_key, lang="简体中文"):
    pdf_image = pdf2image(path, root_path)
    if lang == "English":
        translation_str, translation_file, sum_str, sum_file = translation(
            root_path, path, api_key=api_key, secret_key=secret_key
        )
        all_data_result[path.split("/")[-1].replace(".pdf", "").replace(".", "_")] = [
            pdf_image,
            path,
            translation_str,
            translation_file,
            sum_str,
            sum_file,
        ]
    else:
        data_split, sum_str, sum_file = single_paper_abs_sum(root_path, path, api_key=api_key, secret_key=secret_key)
        all_data_result[path.split("/")[-1].replace(".pdf", "").replace(".", "_")] = [
            pdf_image,
            path,
            "",
            None,
            sum_str,
            sum_file,
        ]
    papers_sum.append({"content": sum_str, "meta": {"name": path}})
    index = path.split("/")[-1].replace(".pdf", "").replace(".", "_")
    return index, data_split


def mul_tackle(
    p_m,
    root_path_list,
    path_list,
    api_key="",
    secret_key="",
    lang="简体中文",
):
    from functools import partial

    func = partial(tackle_paper, api_key=api_key, secret_key=secret_key, lang=lang)
    pool = Pool(processes=min(p_m, multiprocessing.cpu_count()))
    result = pool.starmap(func, [(root_path, path) for root_path, path in zip(root_path_list, path_list)])
    pool.close()
    pool.join()
    return result


def predict(
    file_upload,
    input1=None,
    lang="简体中文",
    api_key="",
    secret_key="",
):

    if os.path.exists("faiss_document_store.db"):
        os.remove("faiss_document_store.db")
    if os.path.exists(index_name):
        import shutil

        shutil.rmtree(index_name)
    if lang == "English":
        if file_upload:
            path_list = [path.name for path in file_upload]
            root_path_list = [os.path.dirname(path) for path in path_list]
        else:
            paths = input1.split(";")
            path_list = []
            root_path_list = ["./" for i in range(len(path_list))]
            root_path = root_path_list[0]
            for index, path_item in enumerate(paths):
                paper = next(arxiv.Search(id_list=[path_item.split("/")[-1]]).results())
                path_name = "{}.pdf".format(path_item.split("/")[-1])
                paper.download_pdf(dirpath=root_path, filename=path_name)
                path = os.path.join(root_path, path_name)
                path_list.append(path)
    else:
        path_list = [path.name for path in file_upload]
        root_path_list = [os.path.dirname(path) for path in path_list]
        document_store = FAISSDocumentStore(
            embedding_dim=768, faiss_index_factory_str="Flat", duplicate_documents="skip"
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
    # import pdb;pdb.set_trace()
    multi_result = mul_tackle(
        p_m=3, root_path_list=root_path_list, path_list=path_list, api_key=api_key, secret_key=secret_key, lang=lang
    )
    for index, split_text in multi_result:
        split_text = retriever.run_indexing(split_text)[0]["documents"]
        document_store.write_documents(split_text, index=str(index))
        # document_store.update_embeddings(retriever,index=str(index))
        document_store.write_documents(split_text)
    # document_store.update_embeddings(retriever)
    document_store.save(index_name)
    mul_sum = merge_summary(papers_sum, api_key=api_key, secret_key=secret_key)
    file_name_sum = root_path_list[0] + "/" + "mul_papers_sum.txt"
    with open(file_name_sum, "w", encoding="utf-8") as f:
        f.write(mul_sum)
    return (
        gr.Textbox.update(value="可以开始chatfile功能"),
        gr.Textbox.update(value="可以开始chatfile功能"),
        gr.Textbox.update(value="可以开始chatfile功能"),
        mul_sum,
        file_name_sum,
    )


def tr_result(file_name):
    global all_data_result
    file_name = file_name.split("/")[-1].replace(".pdf", "").replace(".", "_")
    while True:
        if file_name in all_data_result.keys():
            tr_result = all_data_result[file_name]
            break
        else:
            time.sleep(60)
    return tr_result[:4]


def sum_result(file_name):
    global all_data_result
    file_name = file_name.split("/")[-1].replace(".pdf", "").replace(".", "_")
    while True:
        if file_name in all_data_result.keys():
            sum_result = all_data_result[file_name]
            break
        else:
            time.sleep(60)
    return sum_result[:2] + sum_result[4:]


def retriever_papers(
    query,
    history=None,
    api_key="",
    secret_key="",
    retriever_top=30,
    ranker_top=3,
):
    if history is not None:
        history = []
    message = chat_papers(
        query, api_key=api_key, secret_key=secret_key, retriever_top=retriever_top, ranker_top=ranker_top
    )
    # import pdb;pdb.set_trace()
    history.append(["user: {}".format(query), "assistant: {}".format(message["result"])])
    return "", history, history


def Dropdown_list(papers, inputs):
    global paper_all
    if papers is not None:
        paper_list = [paper.name.split("/")[-1] for paper in papers]
    else:
        paper_list = inputs.split(";")
    paper_all += [i for i in paper_list if i not in paper_all]
    return gr.Dropdown.update(choices=paper_all, value=paper_all[0]), gr.Dropdown.update(
        choices=paper_all, value=paper_all[0]
    )


with gr.Blocks() as demo:
    with gr.Tab("论文翻译总结"):
        with gr.Accordion("论文翻译精读：输入区", open=True, elem_id="input-panel") as area_input_primary:
            with gr.Row():
                file_upload = gr.inputs.File(label="(输入方式1) 请上传论文PDF(仅支持PDF，支持多篇文章翻译)", file_count="multiple")
                with gr.Accordion("(输入方式2) 输入论文的arxiv链接（支持多篇文章翻译，链接直接用英文;隔开）"):
                    input1 = gr.Textbox(
                        label="", value="https://arxiv.org/abs/2303.08774", placeholder="", interactive=True
                    )
                    output1 = gr.Dropdown(choices=["简体中文", "English"], label="输出语言")
                    output2 = gr.Dropdown(choices=["简体中文", "English"], label="输入论文类别", value="简体中文")
        with gr.Row():
            clear = gr.Button(value="清空输入区")
            submit = gr.Button(value="开始翻译精读")

        with gr.Accordion("论文翻译总结：输出区", open=True, elem_id="input-panel") as area_input_primary:
            with gr.Tab("单文翻译"):  # 包含下载功能
                file_name = gr.Dropdown(choices=[""], max_choices=1, label="选择展示论文")
                with gr.Row():
                    with gr.Column():
                        gr.Dropdown(choices=["英文", "中文"], max_choices=1, label="论文原文-PDF插件-支持下载；此处为PDF占位符")
                        ori_paper = gr.Gallery(label="论文原文", show_label=False, elem_id="gallery").style(
                            columns=[2], rows=[2], object_fit="contain", height="auto"
                        )
                        ori_pdf = gr.File(label="原文下载链接")
                    with gr.Accordion("   "):
                        gr.Dropdown(choices=["英文", "中文"], max_choices=1, label="整体翻译-PDF插件-支持下载；此处为PDF占位符")
                        trans_paper = gr.Textbox(label="翻译结果", value="", max_lines=10)
                        trans_down = gr.File(label="翻译下载链接")
                        with gr.Group():
                            start_chatfile_tr = textbox = gr.Textbox(value="暂时无法问答", label="问答启动")
                            chatbot = gr.Chatbot(label="Chatbot")
                            state = gr.State()
                            with gr.Row():
                                textbox = gr.Textbox(
                                    container=False,
                                    show_label=False,
                                    placeholder="这篇论文最大的贡献是什么?",
                                    scale=10,
                                )
                                submit_button = gr.Button("🚀 提交", variant="primary", scale=2, min_width=0)
                                submit_button.click(
                                    chat_file, inputs=[textbox, state, file_name], outputs=[textbox, chatbot, state]
                                )
            with gr.Tab("单文总结"):  # 包含下载功能
                file_sum = gr.Dropdown(choices=[""], max_choices=1, label="选择论文")
                with gr.Row():
                    with gr.Column():
                        gr.Dropdown(choices=["英文", "中文"], max_choices=1, label="论文原文-PDF插件-支持下载；此处为PDF占位符")
                        ori_paper_c = gr.Gallery(label="论文原文", show_label=False, elem_id="gallery").style(
                            columns=[2], rows=[2], object_fit="contain", height="auto"
                        )
                        ori_pdf_c = gr.File(label="原文下载链接")
                    with gr.Accordion("   "):
                        gr.Dropdown(choices=["英文", "中文"], max_choices=1, label="整体翻译-PDF插件-支持下载，此处为PDF占位符")
                        sum_parper = gr.Markdown(label="精读全文", value="")
                        sum_down = gr.File(label="全文精度链接 ")
                        with gr.Group():
                            start_chatfile_sum = textbox = gr.Textbox(value="暂时无法问答", label="问答启动")
                            chatbot = gr.Chatbot(label="Chatbot")
                            state = gr.State()
                            with gr.Row():
                                textbox = gr.Textbox(
                                    container=False,
                                    show_label=False,
                                    placeholder="这篇论文最大的贡献是什么?",
                                    scale=10,
                                )
                                submit_button = gr.Button("🚀 提交", variant="primary", scale=2, min_width=0)
                                submit_button.click(
                                    chat_file, inputs=[textbox, state, file_sum], outputs=[textbox, chatbot, state]
                                )
            with gr.Tab("多文总结"):  # 包含下载功能
                with gr.Accordion("   "):
                    gr.Dropdown(choices=["英文", "中文"], max_choices=1, label="完整总结插件-支持下载，此处为多文总结占位符，需要支持上下拖动")
                    sum_mul_papers = gr.Textbox(label="多文档摘要", value="")
                    sum_mul_papers_down = gr.File(label="全文精度链接 ")
                    with gr.Group():
                        start_chatfile_mul = gr.Textbox(value="暂时无法问答", label="问答启动")
                        chatbot = gr.Chatbot(label="Chatbot")
                        state = gr.State()
                        with gr.Row():
                            textbox = gr.Textbox(
                                container=False,
                                show_label=False,
                                placeholder="这篇论文最大的贡献是什么?",
                                scale=10,
                            )
                            submit_button = gr.Button("🚀 提交", variant="primary", scale=2, min_width=0)
                            submit_button.click(chat_file, inputs=[textbox, state], outputs=[textbox, chatbot, state])
            file_upload.change(Dropdown_list, inputs=[file_upload, input1], outputs=[file_name, file_sum])
            input1.change(Dropdown_list, inputs=[file_upload, input1], outputs=[file_name, file_sum])
            submit.click(
                predict,
                inputs=[file_upload, input1, output2],
                outputs=[
                    start_chatfile_tr,
                    start_chatfile_sum,
                    start_chatfile_mul,
                    sum_mul_papers,
                    sum_mul_papers_down,
                ],
            )
            clear.click(clear_session, inputs=[], outputs=[file_upload, input1])
            file_name.change(tr_result, inputs=file_name, outputs=[ori_paper, ori_pdf, trans_paper, trans_down])
            # file_name.change(tr_result,inputs=file_name,outputs=[ori_paper,ori_pdf,trans_paper,trans_down])
            file_sum.change(sum_result, inputs=file_sum, outputs=[ori_paper_c, ori_pdf_c, sum_parper, sum_down])

    with gr.Tab("技术综述"):
        with gr.Group():
            chatbot = gr.Chatbot(label="Chatbot")
            with gr.Row():
                textbox = gr.Textbox(
                    container=False,
                    show_label=False,
                    placeholder="技术综述输入框口",
                    scale=10,
                )
                submit_button = gr.Button("🚀 提交", variant="primary", scale=2, min_width=0)

    with gr.Tab("学术检索"):
        with gr.Group():
            chatbot = gr.Chatbot(label="Chatbot")
            state = gr.State()
            with gr.Row():
                textbox = gr.Textbox(
                    container=False,
                    show_label=False,
                    placeholder="检索内容输入框口",
                    scale=10,
                )
                submit_button = gr.Button("🚀 提交", variant="primary", scale=2, min_width=0)
            submit_button.click(retriever_papers, inputs=[textbox, state], outputs=[textbox, chatbot, state])
demo.launch(server_name="0.0.0.0", server_port=8084)
