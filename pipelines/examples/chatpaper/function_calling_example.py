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
from pipelines.nodes import (
    BM25Retriever,
    DensePassageRetriever,
    EmbeddingRetriever,
    ErnieRanker,
)
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
    index=args.abstract_index_name,
    index_type=args.index_type,
    ef_construction=200,
    m=32,
    number_of_shard=3,
)
if args.model_type == "ernie-embedding-v1":
    dpr_retriever = EmbeddingRetriever(
        document_store=document_store_with_docs,
        retriever_batch_size=args.retriever_batch_size,
        api_key=args.embedding_api_key,
        embed_title=args.embed_title,
        secret_key=args.embedding_secret_key,
    )
else:
    dpr_retriever = DensePassageRetriever(
        document_store=document_store_with_docs,
        query_embedding_model=args.query_embedding_model,
        passage_embedding_model=args.passage_embedding_model,
        max_seq_len_query=args.max_seq_len_query,
        max_seq_len_passage=args.max_seq_len_passage,
        batch_size=args.retriever_batch_size,
        embed_title=args.embed_title,
        precision="fp16",
    )
ranker = ErnieRanker(model_name_or_path="rocketqa-base-cross-encoder", use_gpu=True)
bm_retriever = BM25Retriever(document_store=document_store_with_docs)

pipeline = Pipeline()
pipeline.add_node(component=dpr_retriever, name="DenseRetriever", inputs=["Query"])
pipeline.add_node(component=ranker, name="Ranker", inputs=["DenseRetriever"])


single_pipe = Pipeline()
single_pipe.add_node(component=bm_retriever, name="BMRetriever", inputs=["Query"])
# 向量检索会引入很多英文噪声数据，暂时放弃
# single_pipe.add_node(component=dpr_retriever, name="DenseRetriever", inputs=["Query"])
# single_pipe.add_node(
#     component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["BMRetriever", "DenseRetriever"]
# )
single_pipe.add_node(component=ranker, name="Ranker", inputs=["BMRetriever"])


def search_multi_paper(query, top_k=3):
    parameters = {
        "DenseRetriever": {
            "top_k": 10,
            "index": args.abstract_index_name,
        },
        "Ranker": {"top_k": top_k},
    }
    for i in range(3):
        try:
            prediction = pipeline.run(
                query=query,
                params=parameters,
            )
        except Exception as e:
            print(e)
            gr.Error(f"Connction error, try times {i}")
            continue

        documents = []
        for doc in prediction["documents"]:
            documents.append(
                {
                    "document": doc.content,
                    "key_words": doc.meta["key_words"],
                    "title": doc.meta["title"],
                }
            )
        if len(documents) > 0:
            break
        else:
            gr.Error(f"Connction error, try times {i}")

    return {"documents": documents}


def search_single_paper(query, title):
    filters = {
        "$and": {
            "title": {"$eq": title},
        }
    }
    parameters = {
        "BMRetriever": {"top_k": 5, "index": args.full_text_index_name, "filters": filters},
        # "DenseRetriever": {
        #     "top_k": 5,
        #     "index": args.full_text_index_name,
        #     "filters": filters,
        # },
        "Ranker": {"top_k": 3},
    }
    for i in range(3):
        try:
            prediction = single_pipe.run(
                query=query,
                params=parameters,
            )
        except Exception as e:
            print(e)
            gr.Error(f"Connction error, try times {i}")
            continue

        documents = []
        for doc in prediction["documents"]:
            documents.append(
                {
                    "document": doc.content,
                    "key_words": doc.meta["key_words"],
                    "title": doc.meta["title"],
                }
            )
        if len(documents) > 0:
            break
        else:
            gr.Error(f"Connction error, try times {i}")

    return {"documents": documents}


def get_literature_review(history, messages):
    base_prompt = """
    请根据聊天历史信息提到的几篇文章，生成综述。按照下面的方式进行输出：某某论文提出什么方法，这个方法有什么点，解决了什么问题。
    """
    literature_text = base_prompt
    messages = messages[:-1]
    messages.append({"role": "user", "content": literature_text})

    resp_stream = erniebot.ChatCompletion.create(model="ernie-bot-3.5", messages=messages, stream=False)
    return {"result": resp_stream["result"]}


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
    logs.append(f"Function Call的输入: {messages}")
    function_times = 0
    error_text = ""
    try:
        while True:
            # Step 1, decide whether we need function call
            resp_stream = erniebot.ChatCompletion.create(
                model="ernie-bot-3.5", messages=messages, functions=functions, stream=True
            )

            # Step 2: execute command
            stream_output = ""
            output_response = ""
            function_flag = False
            for resp in resp_stream:
                if not hasattr(resp, "function_call"):
                    if not function_flag:
                        logs.append("Function Call未触发")
                        function_flag = True
                    stream_output += resp["result"]
                    yield history + [[query, stream_output]], "\n".join(logs)

                else:
                    # Function Call triggered
                    output_response = resp
                    break

            function_times += 1
            # function call未触发
            if function_flag is True:
                break
            # Avoid endless function call
            elif function_times > 6:
                error_text = "当前文心一言服务繁忙，请重试"
                break

            # 2.1: execute function calling
            if hasattr(output_response, "function_call"):
                function_call = output_response.function_call
                logs.append(f"Function Call已触发: {function_call}")
                name2function = {
                    "search_multi_paper": search_multi_paper,
                    "search_single_paper": search_single_paper,
                    "get_literature_review": get_literature_review,
                }

                # 负样本, 目前不做处理，直接跳过
                if function_call["name"] not in name2function:
                    logs.append(f"Function Call的名称{function_call['name']}不存在")

                    response = erniebot.ChatCompletion.create(model="ernie-bot-3.5", messages=messages, stream=True)
                    stream_output = ""
                    for character in response:
                        result = character["result"]
                        stream_output += result
                        yield history + [[query, stream_output]], "\n".join(logs)
                    history.append([query, stream_output])
                    return history, "\n".join(logs)
                else:
                    func = name2function[function_call["name"]]
                    func_args = json.loads(function_call["arguments"])
                    if function_call["name"] == "get_literature_review":
                        func_args["history"] = history
                        func_args["messages"] = messages
                    res = func(**func_args)
                    # 对于多篇论文检索加入润色prompt
                    if function_call["name"] == "search_multi_paper":
                        res["prompt"] = "请根据论文工具的结果返回每篇论文的标题（加粗）, 内容以及关键词，使用自然语言的方式输出，不允许胡编乱造，不要使用json或者表格的形式。"
                    elif function_call["name"] == "get_literature_review":
                        res["prompt"] = "请根据生成的综述，先在开头加上总结性的话语，然后按照某某论文提出什么方法，这个方法有什么点，解决了什么问题的方式输出综述，不要使用json或者表格的形式。"
                    logs.append(f"Function Call调用结果: {res}")
                    # Step 3: return msg to erniebot
                    messages.append({"role": "assistant", "content": None, "function_call": function_call})
                    messages.append(
                        {
                            "role": "function",
                            "name": function_call["name"],
                            "content": json.dumps(res, ensure_ascii=False),
                        }
                    )
    except Exception as e:
        logs.append(f"Function Call执行异常: {e}")
        error_text = "当前文心一言服务繁忙，请重试"

    if error_text != "":
        stream_output = ""
        for token in error_text:
            stream_output += token
            yield history + [[query, stream_output]], "\n".join(logs)

    history.append([query, stream_output])
    return history, "\n".join(logs)


def add_message_chatbot(messages, history):
    history.append([messages, None])
    return None, history


def launch_ui():
    with gr.Blocks(title="维普论文助手", theme=gr.themes.Base()) as demo:
        gr.HTML("""<h1 align="center">维普论文助手</h1>""")
        with gr.Column():
            chatbot = gr.Chatbot(value=[[None, "您好, 我是维普论文助手。除了普通的大模型能力以外，我还特别了解来自维普的学术论文哦！"]], scale=35, height=600)
            message = gr.Textbox(placeholder="你能帮我找一些有关机器学习和强化学习方面的论文吗", lines=1, max_lines=20)
            gr.Examples(
                [["半监督学习的论文有哪些？"], ["请推荐3篇强化学习多智能体的论文"], ["请介绍一下机器学习"]],
                inputs=[message],
                outputs=[message],
                label="示例输入",
            )
            with gr.Row():
                submit = gr.Button("🚀 提交", variant="primary", scale=1)
                clear = gr.Button("清除", variant="primary", scale=1)
            # 默认日志可见
            log = gr.Textbox(value="当前轮次日志", visible=True)

        message.submit(add_message_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
            prediction, inputs=[chatbot], outputs=[chatbot, log]
        )
        submit.click(add_message_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
            prediction, inputs=[chatbot], outputs=[chatbot, log]
        )
        clear.click(
            lambda _: ([[None, "您好, 我是维普论文助手。除了普通的大模型能力以外，我还特别了解来自维普的学术论文哦！"]]), inputs=[clear], outputs=[chatbot]
        )

    demo.queue(concurrency_count=1)
    demo.launch(server_name=args.serving_name, server_port=args.serving_port, debug=True)


if "__main__" == __name__:
    launch_ui()
