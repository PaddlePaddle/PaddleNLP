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
import copy
import json
import os

import arxiv
import erniebot as eb
import gradio as gr
from utils import _apply_token, get_shown_context, pdf2image, retrieval

paper_id_list = []
single_paper_id = ""
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
    "--temperature", type=float, default=0.95, help="The smaller the parameter, the more stable the generated result"
)
parser.add_argument("--max_length", type=int, default=1024, help="Maximum number of generated tokens")
parser.add_argument("--ernie_model", type=str, default="ernie-bot-3.5", help="Model type")
parser.add_argument("--system_prompt", type=str, default="你是我的AI助理", help="System settings for dialogue models")
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
parser.add_argument("--json_dir", type=str, default="", help="the directory of json files created by papers")
parser.add_argument("--max_token", type=int, default=11200, help=" the max number of tokens of LLM")
args = parser.parse_args()


def clear_input():
    """
    Clear input of paper
    """
    global single_paper_id
    single_paper_id = ""
    return "", "", None


def retrieval_clear_session():
    """
    Clear ids of retrieved papers
    """
    global paper_id_list
    paper_id_list = []
    return None, {}


def retrieval_papers(query, state={}):
    """
    Retrieve papers
    """
    query = query.strip().replace("<br>", "\n")
    context = state.setdefault("context", [])
    global paper_id_list
    if not paper_id_list:
        context.append({"system": args.system_prompt, "role": "user", "content": query})
        abstract = ""
        prediction = retrieval(
            query=query,
            es_host=args.es_host,
            es_port=args.es_port,
            es_username=args.es_username,
            es_password=args.es_password,
            es_index=args.es_index_abstract,
            es_chunk_size=args.es_chunk_size,
            es_thread_count=args.es_thread_count,
            es_queue_size=args.es_queue_size,
            retriever_batch_size=args.retriever_batch_size,
            retriever_api_key=args.retriever_api_key,
            retriever_secret_key=args.retriever_secret_key,
            retriever_embed_title=args.retriever_embed_title,
            retriever_topk=30,
            rank_topk=5,
        )
        documents = prediction["documents"]
        for i in range(len(documents)):
            if documents[i].meta["id"] not in paper_id_list:
                paper_id_list.append(documents[i].meta["id"])
                key_words = documents[i].meta.get("key_words", "")
                title = documents[i].meta.get("title", "")
                content = documents[i].content
                paper_content = "**" + str(len(paper_id_list)) + "." + title + "**" + "\n" + key_words + "\n" + content
                abstract += paper_content + "\n\n"
        context.append({"role": "assistant", "content": abstract})
        shown_context = get_shown_context(context)
    else:
        content = ""
        for id in paper_id_list:
            prediction = retrieval(
                query=query,
                file_id=id,
                es_host=args.es_host,
                es_port=args.es_port,
                es_username=args.es_username,
                es_password=args.es_password,
                es_index=args.es_index_full_text,
                es_chunk_size=args.es_chunk_size,
                es_thread_count=args.es_thread_count,
                es_queue_size=args.es_queue_size,
                retriever_batch_size=args.retriever_batch_size,
                retriever_api_key=args.retriever_api_key,
                retriever_secret_key=args.retriever_secret_key,
                retriever_embed_title=args.retriever_embed_title,
                retriever_topk=30,
                rank_topk=2,
            )
            content += "\n".join([item.content for item in prediction["documents"]])
        content = "请根据以下背景资料回答问题：\n 背景资料：{documents} \n问题：{query}".format(documents=content, query=query)
        content = content[: args.max_token]
        context.append({"system": args.system_prompt, "role": "user", "content": content})
        eb.api_type = args.api_type
        access_token = _apply_token(args.api_key, args.secret_key)
        eb.access_token = access_token
        model = "ernie-bot-3.5" if args.ernie_model is None or args.ernie_model.strip() == "" else args.ernie_model
        response = eb.ChatCompletion.create(model=model, messages=context, stream=False)
        bot_response = response.result
        context.append({"role": "assistant", "content": bot_response})
        context_new = copy.deepcopy(context)
        context_new[-2]["content"] = query
        shown_context = get_shown_context(context_new)
    return None, shown_context, state


def retrieval_title(title):
    """
    Retrieve the paper_id  of  the title
    """
    prediction = retrieval(
        title,
        es_host=args.es_host,
        es_port=args.es_port,
        es_username=args.es_username,
        es_password=args.es_password,
        es_index=args.es_index_abstract,
        es_chunk_size=args.es_chunk_size,
        es_thread_count=args.es_thread_count,
        es_queue_size=args.es_queue_size,
        retriever_batch_size=args.retriever_batch_size,
        retriever_api_key=args.retriever_api_key,
        retriever_secret_key=args.retriever_secret_key,
        retriever_embed_title=args.retriever_embed_title,
        retriever_topk=30,
        rank_topk=1,
    )
    if prediction["documents"][0].rank_score > args.retriever_threshold:
        return prediction["documents"][0].meta["full_path"], prediction["documents"][0].meta["id"]
    return None


def infer(query, state):
    """Model inference."""
    eb.api_type = args.api_type
    access_token = _apply_token(args.api_key, args.secret_key)
    eb.access_token = access_token
    query = query.strip().replace("<br>", "\n")
    context = state.setdefault("context", [])
    if single_paper_id:
        prediction = retrieval(
            query=query,
            file_id=single_paper_id,
            es_host=args.es_host,
            es_port=args.es_port,
            es_username=args.es_username,
            es_password=args.es_password,
            es_index=args.es_index_full_text,
            es_chunk_size=args.es_chunk_size,
            es_thread_count=args.es_thread_count,
            es_queue_size=args.es_queue_size,
            retriever_batch_size=args.retriever_batch_size,
            retriever_api_key=args.retriever_api_key,
            retriever_secret_key=args.retriever_secret_key,
            retriever_embed_title=args.retriever_embed_title,
            retriever_topk=30,
            rank_topk=2,
        )
        content = "\n".join([item.content for item in prediction["documents"]])
        content = "请根据以下背景资料回答问题：\n 背景资料：{documents} \n问题：{query}".format(documents=content, query=query)
        content = content[: args.max_token]
        context.append({"system": args.system_prompt, "role": "user", "content": content})
        model = "ernie-bot-3.5" if args.ernie_model is None or args.ernie_model.strip() == "" else args.ernie_model
        response = eb.ChatCompletion.create(model=model, messages=context, stream=False)
        bot_response = response.result
        context.append({"role": "assistant", "content": bot_response})
        context_new = copy.deepcopy(context)
        context_new[-2]["content"] = query
        shown_context = get_shown_context(context_new)
    else:
        context.append({"system": args.system_prompt, "role": "user", "content": query})
        response = eb.ChatFile.create(messages=context, stream=False)
        bot_response = response.result
        context.append({"role": "assistant", "content": bot_response})
        shown_context = get_shown_context(context)
    return None, shown_context, state


def upload_file(file_name, file_url, file_upload, state={}):
    """
    Upload the file to bos or retrieve the json_file of the paper
    """
    global single_paper_id
    if file_name:
        json_file_path, file_id = retrieval_title(file_name)
        json_file_path = json_file_path.replace("/", "_").replace(".pdf", "")
        json_file_path = os.path.join(args.json_dir, json_file_path)
        single_paper_id = file_id
        if os.path.isfile(json_file_path):
            with open(json_file_path, mode="r") as json_file:
                json_content = json.load(json_file)
            content = json_content["content"]
            return (
                gr.Gallery.update(visible=False),
                gr.File.update(visible=False),
                None,
                state,
                gr.Markdown.update(content, visible=True),
            )
        else:
            return gr.Gallery.update(visible=False), gr.File.update(visible=False), None, state, None
    elif file_url:
        single_paper_id = ""
        root_path = "./"
        paper = next(arxiv.Search(id_list=[file_url.split("/")[-1]]).results())
        real_filename = "{}.pdf".format(file_url.split("/")[-1])
        paper.download_pdf(dirpath=root_path, filename=real_filename)
        file_name = os.path.join(root_path, real_filename)
        imgs = pdf2image(pdfPath=file_name, imgPath=root_path)
    elif file_upload:
        single_paper_id = ""
        file_name = file_upload.name
        real_filename = os.path.split(file_name)[-1]
        root_path = os.path.dirname(file_name)
        imgs = pdf2image(pdfPath=file_name, imgPath=root_path)
    # 上传到bos后到文件是否需要删除
    filename_in_bos = real_filename
    url = eb.utils.upload_file_to_bos(
        file_name, filename_in_bos, access_key_id=args.bos_ak, secret_access_key=args.bos_sk
    )
    content = "<file>{}</file><url>{}</url>".format(real_filename, url)
    content = content.strip().replace("<br>", "\n")
    context = state.setdefault("context", [])
    context.append({"system": "你是一位AI小助手", "role": "user", "content": content})
    access_token = _apply_token(args.api_key, args.secret_key)
    eb.api_type = args.api_type
    eb.access_token = access_token
    response = eb.ChatFile.create(messages=context, stream=False)
    bot_response = response.result
    context.append({"role": "assistant", "content": bot_response})
    shown_context = get_shown_context(context)
    return (
        gr.Gallery.update(imgs, visible=True),
        gr.File.update(file_name, label="原文下载链接", visible=True),
        shown_context,
        state,
        gr.Markdown.update(visible=False),
    )


with gr.Blocks(title="维普小助手", theme=gr.themes.Base()) as demo:
    gr.HTML("""<h1 align="center">ChatPaper维普小助手</h1>""")
    with gr.Row(variant="panel"):
        # with gr.Column(scale=1):
        #     #cheetah = os.path.join(os.path.dirname(__file__), "weipu.jpeg")
        #     #gr.Image(cheetah, elem_id="banner-image", show_label=False, show_download_button=False)
        with gr.Column(scale=9):
            gr.HTML(
                """
                <p>【文章检索摘要】
                <p> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.适合从大量文章中，粗力度获取需要的信息。返回包含 : 文章题目+作者+关键词+术语+摘要.</p>
                <p> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.适合基于某个技术领域，生成对应的技术综述.</p>
                <p>【单篇精读翻译】：
                <p> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.适合针对具体一篇文章，详细了解细节内容.</p>
                <p> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.适合针对具体一篇文章，详细翻译摘要术语.</p>
            """
            )
    with gr.Tab("文章检索摘要"):
        retrieval_chatbot = gr.Chatbot(
            height=600, value=[[None, "你好, 我是维普Chatpaper小助手, 我这里收录了100w篇论文,可以提供您专业的学术咨询.请问有什么可以帮您的吗?"]]
        )  # height聊天框高度, value 默认语句
        retrieval_textbox = gr.Textbox(placeholder="最近自监督学习论文有哪些?")
        retrieval_state = gr.State({})
        with gr.Row():
            retrieval_submit_btn = gr.Button("🚀 提交", variant="primary", scale=2, min_width=0)
            retrieval_clear_btn = gr.Button("清除", variant="primary", scale=2, min_width=0)
    retrieval_submit_btn.click(
        retrieval_papers,
        inputs=[retrieval_textbox, retrieval_state],
        outputs=[retrieval_textbox, retrieval_chatbot, retrieval_state],
    )
    retrieval_clear_btn.click(retrieval_clear_session, inputs=[], outputs=[retrieval_chatbot, retrieval_state])
    with gr.Tab("单篇精读翻译"):  # 封装chatFile的能力
        with gr.Accordion("文章精读翻译：输入区（输入方式三选一，三种输入方式优先级依次降低）", open=True, elem_id="input-panel") as area_input_primary:
            with gr.Row():
                with gr.Group():
                    file_name = gr.Textbox(
                        label="(输入方式1) 论文/期刊标题（仅支持单篇文章精读）",
                        value="",
                        placeholder="Human-level control through deep reinforcement learning",
                        interactive=True,
                        scale=1,
                    )
                    file_url = gr.Textbox(
                        label="(输入方式2) 论文 axiv链接（仅支持单篇文章精读）",
                        value="",
                        placeholder="https://arxiv.org/abs/2303.08774",
                        interactive=True,
                        scale=1,
                    )
                file_upload = gr.File(
                    label="(输入方式3) 上传论文/期刊PDF(仅支持单篇PDF精读)", file_count="single", height=180, min_width=50
                )
            with gr.Row():
                clear = gr.Button(value="清空输入区")
                submit = gr.Button(value="全文精读")
        with gr.Accordion("文章精读翻译：输出区", open=True, elem_id="input-panel") as area_input_primary:
            with gr.Tab("单文解读"):  # 包含下载功能
                with gr.Row():
                    with gr.Column():
                        gr.Dropdown(choices=[""], max_choices=1, label="论文原文-PDF插件-支持下载；此处为PDF占位符")
                        ori_paper = gr.Gallery(label="论文原文", show_label=False, elem_id="gallery").style(
                            columns=[1], rows=[1], object_fit="contain", height="700px"
                        )
                        ori_json = gr.Markdown(label="论文原文", visible=False)
                        ori_pdf = gr.File(label="原文下载链接")
                    with gr.Accordion("   "):
                        gr.Dropdown(choices=[""], max_choices=1, label="文章摘要等总结-PDF插件-支持下载；此处为PDF占位符")
                        chatbot = gr.Chatbot(
                            value=[[None, "你好, 我是维普Chatpaper文章精读翻译小助手,可以提供您专业的学术咨询.请问有什么可以帮您的吗?"]],
                            scale=30,
                            height=600,
                        )
                        state = gr.State({})
                        message = gr.Textbox(placeholder="请问具体描述这篇文章的方法?", scale=7)
                        with gr.Row():
                            submit_btn = gr.Button("🚀 提交", variant="primary", scale=2, min_width=0)
                            clear_btn = gr.Button("清除", variant="primary", scale=2, min_width=0)
                submit.click(
                    upload_file,
                    inputs=[file_name, file_url, file_upload, state],
                    outputs=[ori_paper, ori_pdf, chatbot, state, ori_json],
                )
                clear.click(clear_input, inputs=[], outputs=[file_name, file_url, file_upload])
                submit_btn.click(infer, inputs=[message, state], outputs=[message, chatbot, state])
                clear_btn.click(
                    lambda _: (None, None, None, None, {}),
                    inputs=clear_btn,
                    outputs=[ori_paper, ori_pdf, chatbot, ori_json, state],
                    api_name="clear",
                    show_progress=False,
                )
demo.queue(concurrency_count=40, max_size=40)
demo.launch(server_name="0.0.0.0", server_port=8084)
