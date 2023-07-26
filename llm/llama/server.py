#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import re
import time
from multiprocessing.shared_memory import SharedMemory

from predict_generation import Predictor, get_parser

from paddlenlp.utils.log import logger


def read_shared_memory(memory: SharedMemory):
    """read content from shared memory

    Args:
        memory (SharedMemory): the instance of shared Memory
    """
    length = int(memory.buf[0]) * 256 + int(memory.buf[1])
    if length == 0:
        return ""

    sentence = bytes(memory.buf[2 : length + 2]).decode()
    return sentence


def write_shared_memory(memory: SharedMemory, sentence: str):
    """write content into shared memory

        [0:2]: store the length of sentence
        [2:]:  store the content of sentence

    Args:
        memory (SharedMemory): the instance of shared Memory
        sentence (str): the content which must be string
    """
    buffer = bytearray(memory.buf.nbytes)
    data = sentence.encode("utf-8")

    buffer[0:2] = bytearray([len(data) // 256, len(data) % 256])
    buffer[2 : len(data) + 2] = data
    memory.buf[:] = buffer


SLEEP_SECOND = 0.5
SHARED_MEMORY_NAME = "shared_memory"


def create_shared_memory(name: int, rank: int):
    """create shared memory between multi-process

    Args:
        name (int): the name of memory block
        rank (int): the rank of current process
    """
    file = f"{SHARED_MEMORY_NAME}-{name}"
    shared_memory = None
    if rank != 0:
        while True:
            try:
                shared_memory = SharedMemory(file, size=1024 * 100)
                print("success create shared_memory")
                break
            except FileNotFoundError:
                time.sleep(0.01)
                print("sleep for create shared memory")
    else:
        shared_memory = SharedMemory(file, create=True, size=1024 * 100)
    return shared_memory


def enforce_stop_tokens(text, stop) -> str:
    """Code by Langchain"""
    """Cut off the text as soon as any stop words occur."""
    return re.split(re.escape(stop), text)[0]


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class PredictorServer(Predictor):
    def __init__(self, args=None, tokenizer=None, model=None, **kwargs):
        super().__init__(args, tokenizer, model, **kwargs)

        self.input_shared_memory = create_shared_memory("input", self.rank)
        self.output_shared_memory = create_shared_memory("output", self.rank)

        if self.rank == 0:
            write_shared_memory(self.input_shared_memory, "")
            write_shared_memory(self.output_shared_memory, "")

    def start_predict(self, data):
        print("start to predict under data", data)

        data = json.dumps(data, ensure_ascii=False)
        write_shared_memory(self.input_shared_memory, data)

        while True:
            result = read_shared_memory(self.output_shared_memory)
            if result:
                write_shared_memory(self.output_shared_memory, "")
                return result

            else:
                print("not found result, so to sleep ...")

            time.sleep(0.5)

    def start_flask_server(self):
        from flask import Flask, jsonify, request

        app = Flask(__name__)

        @app.post("/api/chat")
        def _server():
            data = request.get_json()
            logger.info(f"Request: {json.dumps(data, indent=2, ensure_ascii=False)}")
            try:
                pred_seq = self.start_predict(data)
                output = {
                    "error_code": 0,
                    "error_msg": "Success",
                    "result": {"response": {"role": "bot", "utterance": pred_seq}},
                }
            except Exception as err:
                logger.error(f"Server error: {err}")
                output = {"error_code": 1000, "error_msg": f"Server error: {err}", "result": None}

            logger.info(f"Response: {json.dumps(output, indent=2, ensure_ascii=False)}")
            return jsonify(output)

        app.run(host="0.0.0.0", port=self.args.flask_port)

    def start_ui_service(self, args):
        # do not support start ui service in one command
        from multiprocessing import Process

        from ui import main

        p = Process(target=main, args=(args,))
        p.daemon = True
        p.start()


def main(args, predictor: Predictor):
    from time import sleep

    while True:
        sleep(0.5)
        content = read_shared_memory(predictor.input_shared_memory)

        if content:
            content = json.loads(content)

            context = content.pop("context", "")
            content.pop("extra_info", None)

            generation_args = content
            predictor.tgt_length = generation_args["max_length"]

            for key, value in generation_args.items():
                setattr(predictor.args, key, value)

            result = predictor.predict(context)
            result = result["result"][0]
            if not result:
                result = "invalid response"
            write_shared_memory(predictor.output_shared_memory, result)
            write_shared_memory(predictor.input_shared_memory, "")


def parse_arguments():
    parser = get_parser()
    parser.add_argument("--port", default=8011, type=int, help="the port of ui service")
    parser.add_argument("--flask_port", default=8010, type=int, help="the port of flask service")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    predictor = PredictorServer(args)

    if predictor.rank == 0:
        predictor.start_ui_service(args)

        from multiprocessing import Process

        p = Process(
            target=predictor.start_flask_server,
        )
        p.daemon = True
        p.start()

    main(args, predictor)
