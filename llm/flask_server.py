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
from contextlib import closing
import socket
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory

import requests

from predictor import BasePredictor, ModelArgument, PredictorArgument, create_predictor, predict

from paddlenlp.trainer import PdArgumentParser
from paddlenlp.utils.log import logger

STOP_SIGNAL = "[END]"

@dataclass
class ServerArgument:
    port: int = field(default=8011, metadata={"help": "The port of ui service"})
    base_port: int = field(default=8010, metadata={"help": "The port of flask service"})
    title: str = field(default="LLM", metadata={"help": "The title of gradio"})

class PredictorServer:
    def __init__(self, args: ServerArgument, predictor: BasePredictor):

        self.predictor = predictor
        self.args = args
        self.port = args.base_port + predictor.tensor_parallel_rank

        if self.predictor.tensor_parallel_rank == 0:
            self.peer_ports = {}
            for peer_id in range(self.predictor.tensor_parallel_degree):
                self.peer_ports[peer_id] = self.args.base_port + peer_id

    def predict(self, input_texts: str | list[str]):
        return self.predictor.stream_predict(input_texts)
    
    def broadcast_msg(self, data):
        for _, peer_port in self.peer_ports.items():
            if peer_port != self.port:
                _ = requests.post(f"http://0.0.0.0:{peer_port}/api/chat", json=data)

    def start_flask_server(self):
        from flask import Flask, request, stream_with_context

        app = Flask(__name__)

        @app.post("/api/chat")
        def _server():
            data = request.get_json()
            logger.info(f"Request: {json.dumps(data, indent=2, ensure_ascii=False)}")
            print("start to predict under data", data)

            if self.predictor.tensor_parallel_rank == 0:
                self.broadcast_msg(data)

            def streaming(data):
                query = data.pop("context", "")
                data.pop("extra_info", None)

                generation_args = data 
                self.predictor.config.max_length = generation_args["max_length"]
                self.predictor.config.top_p = generation_args["top_p"]
                self.predictor.config.temperature = generation_args["temperature"]
                self.predictor.config.top_k = generation_args["top_k"]
                self.predictor.config.repetition_penalty = generation_args["repetition_penalty"]

                for key, value in generation_args.items():
                    setattr(self.args, key, value)

                print("predict {}".format(self.predictor.tensor_parallel_rank), query)
                streamer = self.predict(query)
                if self.predictor.tensor_parallel_rank == 0:
                    for new_text in streamer:
                        print(new_text)
                        output = {
                            "error_code": 0,
                            "error_msg": "Success",
                            "result": {"response": {"role": "bot", "utterance": new_text}},
                        }
                        logger.info(f"Response: {json.dumps(output, indent=2, ensure_ascii=False)}")
                        yield json.dumps(output, ensure_ascii=False) + "\n"
                else:
                    return "done"

            return app.response_class(stream_with_context(streaming(data))) 

        app.run(host="0.0.0.0", port=self.port)

    def start_ui_service(self, args):
        # do not support start ui service in one command
        from multiprocessing import Process

        from gradio_ui import main

        p = Process(target=main, args=(args,))
        p.daemon = True
        p.start()

def find_free_ports(base_port, num):
    def __free_port(port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(('', port))
                return port
            except:
                return -1

    port_set = set()
    step = 0
    while True:
        port = base_port + step
        port = __free_port(port)
        if port != -1 and port not in port_set:
            port_set.add(port)

        if len(port_set) >= num:
            return port_set
        
        step += 1
        if step > 1000:
            break
    return None

if __name__ == "__main__":

    parser = PdArgumentParser((PredictorArgument, ModelArgument, ServerArgument))
    predictor_args, model_args, server_args = parser.parse_args_into_dataclasses()
    predictor = create_predictor(predictor_args, model_args)

    server = PredictorServer(server_args, predictor)

    if server.predictor.tensor_parallel_rank == 0:
        server.start_ui_service(server_args)

        #from multiprocessing import Process

        #p = Process(
        #    target=server.start_flask_server,
        #)
        #p.daemon = True
        #p.start()
    server.start_flask_server()

    #main(server_args, server)
