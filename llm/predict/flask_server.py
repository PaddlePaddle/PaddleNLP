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

from __future__ import annotations

import json
import os
import socket
from contextlib import closing
from dataclasses import asdict, dataclass, field
from time import sleep

import requests
from filelock import FileLock
from predict.predictor import (
    BasePredictor,
    ModelArgument,
    PredictorArgument,
    create_predictor,
)

from paddlenlp.trainer import PdArgumentParser
from paddlenlp.utils.log import logger

STOP_SIGNAL = "[END]"
port_interval = 200
PORT_FILE = "port-info"
FILE_LOCK = "port-lock"


def find_free_ports(port_l, port_u):
    def __free_port(port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(("", port))
                return port
            except Exception:
                return -1

    for port in range(port_l, port_u):
        free = __free_port(port)
        if free != -1:
            return free
    return -1


@dataclass
class ServerArgument:
    port: int = field(default=8011, metadata={"help": "The port of ui service"})
    base_port: int = field(default=None, metadata={"help": "The port of flask service"})
    flask_port: int = field(default=None, metadata={"help": "The port of flask service"})
    title: str = field(default="LLM", metadata={"help": "The title of gradio"})
    sub_title: str = field(default="LLM-subtitle", metadata={"help": "The sub-title of gradio"})


class PredictorServer:
    def __init__(self, args: ServerArgument, predictor: BasePredictor):
        self.predictor = predictor
        self.args = args
        scan_l, scan_u = (
            self.args.flask_port + port_interval * predictor.tensor_parallel_rank,
            self.args.flask_port + port_interval * (predictor.tensor_parallel_rank + 1),
        )
        self.total_max_length = predictor.config.src_length + predictor.config.max_length

        if self.predictor.tensor_parallel_rank == 0:
            self.port = find_free_ports(scan_l, scan_u)
            self.peer_ports = {}
            while True and self.predictor.tensor_parallel_degree > 1:
                if os.path.exists(PORT_FILE):
                    with FileLock(FILE_LOCK), open(PORT_FILE, "r") as f:
                        cnt = 1
                        for line in f:
                            port_data = json.loads(line)
                            self.peer_ports[port_data["rank"]] = port_data["port"]
                            cnt += 1
                    if cnt == predictor.tensor_parallel_degree:
                        break
                    else:
                        print("waiting for port reach", cnt)
                sleep(1)
        else:
            self.port = find_free_ports(scan_l, scan_u)
            data = {"rank": predictor.tensor_parallel_rank, "port": self.port}
            with FileLock(FILE_LOCK), open(PORT_FILE, "a") as f:
                f.write(json.dumps(data) + "\n")
            print("rank:", predictor.tensor_parallel_rank, " port info saving done.")

    def predict(self, input_texts: str | list[str]):
        return self.predictor.stream_predict(input_texts)

    def broadcast_msg(self, data):
        for _, peer_port in self.peer_ports.items():
            if peer_port != self.port:
                _ = requests.post(f"http://0.0.0.0:{peer_port}/v1/chat/completions", json=data)

    def start_flask_server(self):
        from flask import Flask, request, stream_with_context

        app = Flask(__name__)

        @app.post("/v1/chat/completions")
        def _server():
            data = request.get_json()
            logger.info(f"Request: {json.dumps(data, indent=2, ensure_ascii=False)}")

            # 如果请求中包含 "messages"，则按照 OpenAI 标准格式处理
            if "messages" in data:
                messages = data["messages"]
                if not messages:
                    return json.dumps({"error": "Empty messages"}), 400
                # 假设最后一条消息来自用户，将其作为当前对话的输入(query)
                if messages[-1].get("role") == "user":
                    query = messages[-1].get("content", "")
                    history = []
                    if len(messages) > 1:
                        temp = []
                        for msg in messages[:-1]:
                            if msg.get("role") in ["user", "assistant"]:
                                temp.append(msg.get("content", ""))
                        # 如果数量为奇数，则忽略最早的一条，以保证交替格式
                        if len(temp) % 2 != 0:
                            temp = temp[1:]
                        history = temp
                else:
                    query = ""
                    history = [msg.get("content", "") for msg in messages if msg.get("role") in ["user", "assistant"]]
                data["context"] = query
                data["history"] = history
            else:
                # 向后兼容处理
                data["context"] = data.get("context", "")
                data["history"] = data.get("history", "")

            # 如果当前 rank 为 0，则向其它进程广播请求数据
            if self.predictor.tensor_parallel_rank == 0:
                self.broadcast_msg(data)

            def streaming(data):
                # 移除无用的 extra_info 字段
                data.pop("extra_info", None)
                query = data.pop("context", "")
                history = data.pop("history", [])
                if isinstance(history, str):
                    try:
                        history = json.loads(history)
                    except Exception:
                        history = [history]

                # 如果模型支持 chat_template，则转换为消息格式处理
                if self.predictor.tokenizer.chat_template is not None:
                    messages = []
                    # 将历史对话转换为交替格式（用户与助手成对出现）
                    for idx in range(0, len(history), 2):
                        user_msg = history[idx] if isinstance(history[idx], str) else history[idx].get("utterance", "")
                        messages.append({"role": "user", "content": user_msg})
                        if idx + 1 < len(history):
                            assistant_msg = (
                                history[idx + 1]
                                if isinstance(history[idx + 1], str)
                                else history[idx + 1].get("utterance", "")
                            )
                            messages.append({"role": "assistant", "content": assistant_msg})
                    # 将当前查询作为新的用户输入追加到消息列表中
                    messages.append({"role": "user", "content": query})
                    query = messages
                # 否则保持原有 context/history 格式

                generation_args = data
                self.predictor.config.max_length = generation_args.get("max_length", self.predictor.config.max_length)
                if "src_length" in generation_args:
                    self.predictor.config.src_length = generation_args["src_length"]

                if self.predictor.config.src_length + self.predictor.config.max_length > self.total_max_length:
                    output = {
                        "error_code": 1,
                        "error_msg": (
                            f"The sum of src_length<{self.predictor.config.src_length}> and max_length<{self.predictor.config.max_length}> "
                            f"should be smaller than or equal to the max-total-length<{self.total_max_length}>"
                        ),
                    }
                    yield json.dumps(output, ensure_ascii=False) + "\n"
                    return

                self.predictor.config.top_p = generation_args.get("top_p", self.predictor.config.top_p)
                self.predictor.config.temperature = generation_args.get(
                    "temperature", self.predictor.config.temperature
                )
                self.predictor.config.top_k = generation_args.get("top_k", self.predictor.config.top_k)
                self.predictor.config.repetition_penalty = generation_args.get(
                    "repetition_penalty", self.predictor.config.repetition_penalty
                )

                for key, value in generation_args.items():
                    setattr(self.args, key, value)

                streamer = self.predict(query)
                # 流式返回生成结果，构造 OpenAI 标准响应格式
                if self.predictor.tensor_parallel_rank == 0:
                    for new_text in streamer:
                        if not new_text:
                            continue
                        response_body = {
                            "id": self.args.model_name_or_path,
                            "object": "chat.completion",
                            "created": int(sleep(0) or 0),
                            "model": generation_args.get("model", "custom-model"),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": new_text,
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        yield json.dumps(response_body, ensure_ascii=False) + "\n"
                else:
                    return "done"

            return app.response_class(stream_with_context(streaming(data)))

        # 启动 Flask 服务（单线程预测）
        app.run(host="0.0.0.0", port=self.port, threaded=False)

    def start_ui_service(self, args, predictor_args):
        from multiprocessing import Process

        from gradio_ui import main

        p = Process(target=main, args=(args, predictor_args))
        p.daemon = True
        p.start()


if __name__ == "__main__":
    parser = PdArgumentParser((PredictorArgument, ModelArgument, ServerArgument))
    predictor_args, model_args, server_args = parser.parse_args_into_dataclasses()
    server_args.model_name_or_path = predictor_args.model_name_or_path

    if server_args.base_port is not None:
        logger.warning("`--base_port` is deprecated, please use `--flask_port` instead after 2023.12.30.")
        if server_args.flask_port is None:
            server_args.flask_port = server_args.base_port
        else:
            logger.warning("Both `--base_port` and `--flask_port` are set; `--base_port` will be ignored.")

    log_dir = os.getenv("PADDLE_LOG_DIR", "./")
    PORT_FILE = os.path.join(log_dir, PORT_FILE)
    if os.path.exists(PORT_FILE):
        os.remove(PORT_FILE)

    predictor = create_predictor(predictor_args, model_args)
    server = PredictorServer(
        server_args,
        predictor,
    )
    if server.predictor.tensor_parallel_rank == 0:
        server.start_ui_service(server_args, asdict(predictor.config))
    server.start_flask_server()
