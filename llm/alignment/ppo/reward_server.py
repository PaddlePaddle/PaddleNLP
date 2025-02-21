# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Launch Reward HTTP Server."""
import argparse
import json
import logging
import re
import threading
import traceback
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class Request(BaseModel):
    """The request for RM server."""

    src: List[str]
    tgt: List[str]
    response: List[str]


class Response(BaseModel):
    """The response for RM server."""

    error_code: int = 0
    error_msg: str = "Success"
    score: List[float] = None


def extract_solution(solution_str):
    answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
    final_answer = None
    if len(answer) == 0:
        # no reward is there is no answer
        pass
    else:
        invalid_str = ["", "."]
        # find the last number that is not '.'
        for final_answer in reversed(answer):
            if final_answer not in invalid_str:
                break
    return final_answer


def compute_score(solution_str, ground_truth, query=None):
    """
    计算给定答案的得分。

    Args:
        solution_str (str): 模型生成的答案字符串。
        ground_truth (str): 标准答案字符串。
        query (str, optional): 输入的问题字符串，默认为None。

    Returns:
        float: 计算得到的答案得分。

    """
    format_score = 0.0
    score = 1.0
    solution = extract_solution(solution_str=solution_str)
    answer = extract_solution(solution_str=ground_truth)
    logging.info(f"solution: {solution}, answer:{answer}")
    if solution is None:
        return 0.0
    else:
        if solution == answer:
            return score
        else:
            return format_score


def setup_args():
    """Setup inerance server arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--log_file", type=str, default="rm_server.log")
    args = parser.parse_args()
    return args


def server(args):
    """Launch RM server."""
    app = FastAPI()
    lock = threading.Lock()

    logging.basicConfig(
        level=logging.INFO,
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s - %(message)s",
    )

    @app.post("/")
    async def _server(request: Request) -> Response:
        lock.acquire()
        logging.info(f"Request: {request}")
        try:
            all_result = []
            if len(request.tgt) != len(request.response) or len(request.tgt) != len(request.src):
                raise ValueError("The length of response, tgt, and src should be equal.")
            for i in range(len(request.response)):
                reward = compute_score(request.response[i], request.tgt[i], request.src[i])
                all_result.append(reward)
            output = {
                "error_code": 0,
                "error_msg": "Success",
                "score": all_result,
            }
        except Exception as err:
            logging.error(f"Server error: when process {request}\n{traceback.format_stack()}")
            output = {
                "error_code": 500,
                "error_msg": f"{err}",
                "score": [0] * len(request.tgt),
            }
        logging.info(f"Response: {json.dumps(output, indent=2, ensure_ascii=False)}")
        lock.release()
        return output

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    args = setup_args()
    server(args)
