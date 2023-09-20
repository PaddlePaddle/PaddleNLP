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

import json

import erniebot
import gradio as gr
from prompt_utils import functions, get_parse_args

from pipelines.document_stores import BaiduElasticsearchDocumentStore
from pipelines.nodes import EmbeddingRetriever
from pipelines.pipelines import Pipeline

args = get_parse_args()
erniebot.api_type = "qianfan"
erniebot.ak = args.api_key
erniebot.sk = args.secret_key

document_store_with_docs = BaiduElasticsearchDocumentStore(
    host=args.host,
    port=args.port,
    username=args.username,
    password=args.password,
    embedding_dim=args.embedding_dim,
    similarity="dot_prod",
    vector_type="bpack_vector",
    search_fields=["content", "meta"],
    index=args.abstract_index_name,
)
dpr_retriever = EmbeddingRetriever(
    document_store=document_store_with_docs,
    retriever_batch_size=args.retriever_batch_size,
    api_key=args.embedding_api_key,
    embed_title=args.embed_title,
    secret_key=args.embedding_secret_key,
)

pipeline = Pipeline()
pipeline.add_node(component=dpr_retriever, name="DenseRetriever", inputs=["Query"])


def search_multi_paper(query):
    prediction = pipeline.run(
        query=query,
        params={
            "DenseRetriever": {
                "top_k": args.retriever_top_k,
                "index": args.abstract_index_name,
            },
        },
    )

    documents = []
    for doc in prediction["documents"]:
        documents.append(
            {
                "document": doc.content,
                "key_words": doc.meta["key_words"],
                "title": doc.meta["title"],
            }
        )
    return {"documents": documents}


def search_single_paper(query, title):
    filters = {
        "$and": {
            "title": {"$eq": title},
        }
    }
    prediction = pipeline.run(
        query=query,
        params={
            "DenseRetriever": {
                "top_k": 3,
                "index": args.full_text_index_name,
                "filters": filters,
            },
        },
    )

    documents = []
    for doc in prediction["documents"]:
        documents.append(
            {
                "document": doc.content,
                "key_words": doc.meta["key_words"],
                "title": doc.meta["title"],
            }
        )

    return {"documents": documents}


def history_transform(history=[]):
    messages = []
    if len(history) < 2:
        return messages

    for turn_idx in range(1, len(history)):
        messages.extend(
            [{"role": "user", "content": history[turn_idx][0]}, {"role": "assistant", "content": history[turn_idx][1]}]
        )
    return messages


def prediction(history):
    logs = []
    query = history.pop()[0]
    if query == "":
        return history, "注意：问题不能为空"
    for turn_idx in range(len(history)):
        if history[turn_idx][0] is not None:
            history[turn_idx][0] = history[turn_idx][0].replace("<br>", "")
        if history[turn_idx][1] is not None:
            history[turn_idx][1] = history[turn_idx][1].replace("<br>", "")

    messages = history_transform(history)
    messages.append({"role": "user", "content": query})
    # Step 1, decide whether we need function call
    response = erniebot.ChatCompletion.create(
        model="ernie-bot-3.5",
        messages=messages,
        functions=functions,
    )
    # Step 2: execute command
    if "function_call" not in response:
        logs.append("Function Call未触发")
        result = response["result"]
    else:
        function_call = response.function_call
        logs.append(f"Function Call已触发: {function_call}")
        name2function = {"search_multi_paper": search_multi_paper, "search_single_paper": search_single_paper}
        func = name2function[function_call["name"]]
        func_args = json.loads(function_call["arguments"])
        res = func(**func_args)
        # 对于多篇论文检索加入润色prompt
        if function_call["name"] == "search_multi_paper":
            res["prompt"] = "请根据论文检索工具的结果返回每篇论文的标题（加粗）, 内容以及关键词"
        logs.append(f"Function Call调用结果: {res}")
        # Step 3: return msg to erniebot
        messages.append({"role": "assistant", "content": None, "function_call": function_call})
        messages.append(
            {"role": "function", "name": function_call["name"], "content": json.dumps(res, ensure_ascii=False)}
        )
        response = erniebot.ChatCompletion.create(model="ernie-bot-3.5", messages=messages)
        result = response["result"]
    history.append([query, result])
    return history, "\n".join(logs)


def add_message_chatbot(messages, history):
    history.append([messages, None])
    return None, history


def launch_ui():
    with gr.Blocks(title="维普小助手", theme=gr.themes.Base()) as demo:
        gr.HTML("""<h1 align="center">ChatPaper维普小助手</h1>""")
        with gr.Tab("ChatPaper"):
            with gr.Column():
                chatbot = gr.Chatbot(value=[[None, "您好, 我是维普论文小助手"]], scale=35, height=500)
                message = gr.Textbox(placeholder="你能帮我找一些有关机器学习和强化学习方面的论文吗", lines=1, max_lines=20)
                with gr.Row():
                    submit = gr.Button("🚀 提交", variant="primary", scale=1)
                    clear = gr.Button("清除", variant="primary", scale=1)
                log = gr.Textbox(value="当前轮次日志")
            message.submit(add_message_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
                prediction, inputs=[chatbot], outputs=[chatbot, log]
            )
            submit.click(add_message_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
                prediction, inputs=[chatbot], outputs=[chatbot, log]
            )
            clear.click(lambda _: ([[None, "您好, 我是维普论文小助手"]]), inputs=[clear], outputs=[chatbot])
    demo.launch(server_name=args.serving_name, server_port=args.serving_port, debug=True)


if "__main__" == __name__:
    launch_ui()
