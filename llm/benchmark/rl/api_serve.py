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

import asyncio
import csv
import json
import logging
import os
from pathlib import Path
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import List, Tuple
import argparse

import pandas as pd
from openai import AsyncOpenAI

from paddlenlp.transformers import AutoTokenizer

# 配置根 Logger
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# 请求api的参数类
@dataclass
class RequestPayload:
    """请求有效载荷"""

    prompt: str = "你好"
    num_responses: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 20 * 1024
    idx: int = 0


# 响应api的参数类
@dataclass
class ResponsePayload:
    """响应有效载荷"""

    idx: int = 0
    question: str = ""
    question_token_length: int = 0
    responses: List[str] = field(default_factory=list)
    elapsed_times: List[float] = field(default_factory=list)
    token_lengths: List[int] = field(default_factory=list)
    total_length: int = 0


class StatisticsManager:
    def __init__(self, batch_path: str, group_path: str, res_path: str, batch_num: int, responses_num: int = 8):
        self.batch_path = batch_path
        self.group_path = group_path
        self.res_path = res_path
        self.batch_num = batch_num
        self.responses_num = responses_num
        self.batch_idx = 0

    def res_stats(self, response: List[ResponsePayload]):
        batch_group_pd = pd.DataFrame(response)
        res_batch_pd = batch_group_pd[["idx", "question", "responses"]]
        responses_batch_pd = pd.DataFrame(
            res_batch_pd["responses"].to_list(), columns=[f"response_{i+1}" for i in range(self.responses_num)]
        )
        res_batch_pd = pd.concat([res_batch_pd[["idx", "question"]], responses_batch_pd], axis=1)

        res_batch_pd.to_json(self.res_path, orient="records", lines=True, force_ascii=False, mode="a")

    def group_stats(self, responses: List[ResponsePayload]):
        batch_group_pd = pd.DataFrame(responses)

        batch_group_pd = batch_group_pd[["idx", "question_token_length", "elapsed_times", "token_lengths"]]

        batch_group_pd["min_elapsed_time"] = batch_group_pd["elapsed_times"].apply(lambda x: min(x))
        batch_group_pd["mean_elapsed_time"] = batch_group_pd["elapsed_times"].apply(lambda x: sum(x) / len(x))
        batch_group_pd["max_elapsed_time"] = batch_group_pd["elapsed_times"].apply(lambda x: max(x))
        batch_group_pd["group_token_length"] = batch_group_pd["token_lengths"].apply(lambda x: sum(x))

        batch_group_pd["elapsed_times"] = batch_group_pd["elapsed_times"].apply(lambda x: [round(y, 2) for y in x])

        # 将一些列移动到最前面
        front_cols_names = [
            "idx",
            "question_token_length",
            "min_elapsed_time",
            "mean_elapsed_time",
            "max_elapsed_time",
            "group_token_length",
        ]
        cols = front_cols_names + [col for col in batch_group_pd.columns if col not in front_cols_names]
        batch_group_pd = batch_group_pd[cols]

        if not os.path.exists(self.group_path):
            batch_group_pd.to_csv(self.group_path, mode="w", index=False, header=True, float_format="%.2f")
        else:
            batch_group_pd.to_csv(self.group_path, mode="a", index=False, header=False, float_format="%.2f")

        return batch_group_pd

    def batch_stats(self, batch_responses: List[ResponsePayload], batch_elapsed_time: float):
        self.res_stats(batch_responses)
        batch_group_pd = self.group_stats(batch_responses)

        if "idx" not in batch_group_pd.columns:
            raise ValueError("ResponsePayload objects must have 'idx' field set")

        group_idx = batch_group_pd["idx"].to_list()
        batch_token_length = batch_group_pd["group_token_length"].sum()

        batch_data = {
            "batch_idx": [self.batch_idx],
            "group_idx": [group_idx],
            "batch_elapsed_time": [batch_elapsed_time],
            "batch_min_elapsed_time": [batch_group_pd["min_elapsed_time"].min(axis=0)],
            "batch_mean_elapsed_time": [batch_group_pd["mean_elapsed_time"].mean(axis=0)],
            "batch_max_elapsed_time": [batch_group_pd["max_elapsed_time"].max(axis=0)],
            "batch_token_length": [batch_token_length],
            "batch_group_token_length": [batch_group_pd["group_token_length"].to_list()],
            "batch_throughput": [batch_token_length / batch_elapsed_time],
        }

        batch_pd = pd.DataFrame(batch_data)

        if not os.path.exists(self.batch_path):
            batch_pd.to_csv(self.batch_path, mode="w", index=False, header=True, float_format="%.2f")
        else:
            batch_pd.to_csv(self.batch_path, mode="a", index=False, header=False, float_format="%.2f")

        self.batch_idx += 1

        return batch_pd


class TokenizerCalculator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M"):
        self.tokenzizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(self, response: ResponsePayload) -> ResponsePayload:
        question = response.question
        responses = response.responses
        response.question_token_length = len(self.tokenzizer(question).input_ids)

        for i, resp in enumerate(responses):
            tokens = self.tokenzizer(resp).input_ids
            length = len(tokens)
            response.token_lengths.append(length)
            response.total_length += length

        return response


class AsyncStreamingClient:
    def __init__(
        self,
        model: str,
        stats_manager: StatisticsManager,
        tokenizer: TokenizerCalculator,
        clients_url: List[str],
        api_keys: List[str],
        max_concurrency: int,
    ) -> None:
        self.model = model
        self.stats_manager = stats_manager
        self.tokenizer = tokenizer
        self.clients = cycle(AsyncOpenAI(base_url=url, api_key=api) for url, api in zip(clients_url, api_keys))
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency

    def get_active_tasks_count(self) -> int:
        return self._max_concurrency - self.semaphore._value

    def get_client(self) -> AsyncOpenAI:
        # 返回一个AsyncOpenAI客户端实例
        return next(self.clients)

    async def process_dataset(self, dataset: List[RequestPayload], batch_size: int):
        logger.info("================= PROCESS START =================")
        start_time = time.perf_counter()

        for index in range(0, len(dataset), batch_size):
            batch_data = dataset[index : index + batch_size]
            batch_res_results, batch_elapsed_time = await self.batch_call(batch_data)

            # token统计
            for i in range(len(batch_res_results)):
                batch_res_results[i] = self.tokenizer.tokenize(batch_res_results[i])

            # 对batch相应进行统计
            self.stats_manager.batch_stats(batch_res_results, batch_elapsed_time)

        end_time = time.perf_counter()
        logger.info("================== PROCESS END ==================")
        logger.info("total processing took %.4f seconds", end_time - start_time)

    async def batch_call(self, requests: List[RequestPayload]) -> Tuple[List[ResponsePayload], int]:
        """批量执行请求"""
        start_time = time.perf_counter()
        batch_results = await asyncio.gather(*[self.group_call(request) for request in requests])
        end_time = time.perf_counter()
        batch_elapsed_time = end_time - start_time
        logger.debug("total batch took %.4f seconds", batch_elapsed_time)
        return batch_results, batch_elapsed_time

    async def group_call(self, request: RequestPayload) -> ResponsePayload:
        # 采用异步一次调用num_responses次 get_respose方法，并返回结果
        tasks = [self(request) for _ in range(request.num_responses)]

        result = ResponsePayload()
        result.idx = request.idx
        result.question = request.prompt
        start_time = time.perf_counter()
        for task, elapsed_time in await asyncio.gather(*tasks):
            result.responses.append(task)
            result.elapsed_times.append(elapsed_time)
        end_time = time.perf_counter()
        group_elapsed_time = end_time - start_time
        logger.debug("total group took %.2f seconds", group_elapsed_time)
        return result

    async def __call__(self, request: RequestPayload) -> Tuple[str, float]:
        client = self.get_client()
        try:
            async with self.semaphore:
                logger.debug("client is : %s", client.base_url)
                logger.debug(f"当前有 {self.get_active_tasks_count()} 个异步任务正在工作")
                start_time = time.perf_counter()
                response = await client.completions.create(
                    model=self.model,
                    prompt=request.prompt,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    n=1,
                    stream=True,
                )
                # 流式文字存储在chunks列表中
                chunks = []
                # 流式处理响应
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].text:
                        chunks.append(chunk.choices[0].text)
                text = "".join(chunks)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.debug("Streaming response took %.4f seconds", elapsed_time)
                return text, elapsed_time

        except Exception as e:
            logger.error("Error while streaming: %s", e)
            raise ValueError(e)


class ResponseLengthCalculator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M"):
        self.tokenzizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    @staticmethod
    def save_to_csv_serialized(response_payloads: List[ResponsePayload], filename: str):
        if not response_payloads:
            print("没有数据可保存。")
            return

        fieldnames = [
            "request_payload.idx",
            "elapsed_times",
            "lengths",
            "mean_elapsed_time",
            "min_elapsed_time",
            "max_elapsed_time",
            "total_elapsed_time",
            "total_length",
        ]

        file_exists = os.path.exists(filename)
        existing_columns = set()

        if file_exists:
            with open(filename, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    existing_columns = set(reader.fieldnames)

        # 打开文件并写入数据
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 如果文件不存在或列名不匹配，写入列名
            if not file_exists or existing_columns != set(fieldnames):
                writer.writeheader()

            for rp in response_payloads:
                row = {
                    "request_payload.idx": rp.request_payload.idx,
                    "elapsed_times": json.dumps([round(t, 2) for t in rp.elapsed_times]),
                    "lengths": json.dumps(rp.lengths),
                    "mean_elapsed_time": round(rp.mean_elapsed_time, 2),
                    "min_elapsed_time": round(rp.min_elapsed_time, 2),
                    "max_elapsed_time": round(rp.max_elapsed_time, 2),
                    "total_elapsed_time": round(rp.total_elapsed_time, 2),
                    "total_length": rp.total_length,
                }
                writer.writerow(row)

    def tokenize(self, texts: List[str]) -> List[int]:
        result = []
        input_ids = self.tokenzizer(texts, padding=False)["input_ids"]
        for input_id in input_ids:
            result.append(len(input_id))
        return result

    def calculate_stats(self, responses: List[ResponsePayload]) -> List[ResponsePayload]:
        for response in responses:
            start_time = time.perf_counter()
            response_texts = response.responses
            response.lengths = self.tokenize(response_texts)
            for i, length in enumerate(response.lengths):
                response.mean_elapsed_time += response.elapsed_times[i] / len(response.lengths)
                response.min_elapsed_time = min(response.elapsed_times)
                response.max_elapsed_time = max(response.elapsed_times)
                response.total_elapsed_time += response.elapsed_times[i]
                response.total_length += length
            end_time = time.perf_counter()
            logger.debug("tokenization took %.4f seconds", end_time - start_time)
        return responses


def parse_args():
     # 初始化 ArgumentParser
    parser = argparse.ArgumentParser(description="Process prompts with OpenAI clients.")
    # 添加参数
    parser.add_argument("--openai_services", type=str, nargs="+", required=True, help="List of OpenAI service URLs")
    parser.add_argument("--api_keys", type=str, nargs="+", default=None, help="List of API keys (default: 'NONE' for each service)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Qwen2.5-7B-Instruct-1M)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer name (e.g., Qwen/Qwen2.5-7B-Instruct-1M)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for requests")
    parser.add_argument("--response_num", type=int, default=8, help="Number of responses per request")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Key in the DataFrame for prompts (default: 'prompt')")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input Parquet file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for output CSV files (default: './output')")
    # 解析参数
    return parser.parse_args()

def main():
    args = parse_args()

    # 处理 API_KEYS 的默认值
    if args.api_keys is None:
        args.api_keys = ["NONE"] * len(args.openai_services)
    elif len(args.api_keys) != len(args.openai_services):
        raise ValueError("Length of --api_keys must match --openai_services")

    os.makedirs(args.output_dir, exist_ok=True)

    # 构建输出文件路径
    batch_csv = f"{args.output_dir}/batch_stats.csv"
    group_csv = f"{args.output_dir}/group_stats.csv"
    response_csv = f"{args.output_dir}/rollout_details.jsonl"

    # 初始化组件
    stats_manager = StatisticsManager(batch_csv, group_csv, response_csv, args.response_num)
    token_calc = TokenizerCalculator(args.tokenizer)
    client = AsyncStreamingClient(
        model=args.model,
        stats_manager=stats_manager,
        tokenizer=token_calc,
        clients_url=args.openai_services,
        api_keys=args.api_keys,
        max_concurrency=1000,
    )

    # 读取数据并处理
    dataframe = pd.read_parquet(args.data_path)
    all_prompts = []

    for idx, prompt in enumerate(dataframe[args.prompt_key]):
        all_prompts.append(RequestPayload(prompt=prompt[0]["content"], idx=idx))

    asyncio.run(client.process_dataset(all_prompts, args.batch_size))


if __name__ == "__main__":
    main()
