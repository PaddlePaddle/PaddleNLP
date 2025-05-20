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

import argparse
import asyncio
import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm
from utils import RangeSet

from paddlenlp.transformers import AutoTokenizer

from transformers import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

@dataclass
class RequestPayload:
    prompt: str = ""
    num_responses: int = 8
    idx: int = 0


@dataclass
class ResponsePayload:
    idx: int = 0
    question: str = ""
    question_token_length: int = 0
    responses: List[str] = field(default_factory=list)
    elapsed_times: List[float] = field(default_factory=list)
    token_lengths: List[int] = field(default_factory=list)
    total_length: int = 0


class StatisticsManager:
    """Manages statistics collection and analysis for batch inference operations.

    This class provides methods to compute both per-group (dispersed) and aggregated (global)
    statistics from batch inference responses, including token lengths, processing times,
    and throughput metrics.
    """
    def __init__(self, batch_size: int, rollout_n: int):
        self.batch_size = batch_size
        self.rollout_n = rollout_n

    def dispersed_stats(self, responses: List[ResponsePayload], batch_elapsed_time: float):
        batch_group_pd = pd.DataFrame(responses)

        dispersed_stats_dict = {
            "batch_index": self.batch_index,
            "rollout_lengths": batch_group_pd["token_lengths"].to_list(),
            "min_length": batch_group_pd["token_lengths"].apply(lambda x: min(x)).tolist(),
            "max_length": batch_group_pd["token_lengths"].apply(lambda x: max(x)).tolist(),
            "avg_length": batch_group_pd["token_lengths"].apply(lambda x: sum(x) / len(x)).tolist(),
            "completion_time": batch_elapsed_time,
            "throughput_tokens_per_sec": batch_group_pd["token_lengths"].apply((lambda x: sum(x))).sum()
            / batch_elapsed_time,
            # "elapsed_times": batch_group_pd["elapsed_times"].to_list(),
            # "min_time": batch_group_pd["elapsed_times"].apply(lambda x: min(x)).tolist(),
            # "max_time": batch_group_pd["elapsed_times"].apply(lambda x: max(x)).tolist(),
            # "avg_time": batch_group_pd["elapsed_times"].apply(lambda x: round(sum(x) / len(x), 2)).tolist(),
        }

        return dispersed_stats_dict

    def global_stats(self, responses: List[ResponsePayload], batch_elapsed_time: float):
        dispersed_stats_dict = self.dispersed_stats(responses, batch_elapsed_time)

        total_response_tokens = 0
        for lengths in dispersed_stats_dict["rollout_lengths"]:
            total_response_tokens += sum(lengths)

        global_stats_dict = {}
        global_stats_dict["batch_index"] = dispersed_stats_dict["batch_index"]
        global_stats_dict["min_response_tokens"] = min(dispersed_stats_dict["min_length"])
        global_stats_dict["max_response_tokens"] = max(dispersed_stats_dict["max_length"])
        global_stats_dict["avg_response_tokens"] = total_response_tokens / (self.batch_size * self.rollout_n)
        global_stats_dict["total_response_tokens"] = total_response_tokens
        global_stats_dict["group_max_response_tokens"] = dispersed_stats_dict["max_length"]
        # global_stats_dict["min_time"] = min(dispersed_stats_dict["min_time"])
        # global_stats_dict["avg_time"] = round(sum(dispersed_stats_dict["avg_time"]) / len(responses), 2)
        global_stats_dict["completion_time"] = dispersed_stats_dict["completion_time"]
        global_stats_dict["throughput_tokens_per_sec"] = dispersed_stats_dict["throughput_tokens_per_sec"]

        return global_stats_dict, dispersed_stats_dict


class ApiTask:
    def __init__(self, args, max_concurrency: int = 1000):
        self.args = args
        self.model = args.model
        self.tokenizer = TokenizerCalculator(model_name=self.args.tokenizer)
        self.clients = cycle(
            AsyncOpenAI(base_url=url, api_key=api) for url, api in zip(args.openai_urls, args.api_keys)
        )
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency

        self.output_dir = Path(self.args.output_dir)

        self.global_stats_path = self.output_dir / "global_stats.csv"
        self.dispersed_stats_path = self.output_dir / "dispersed_stats.csv"
        self.rollout_details_path = self.output_dir / "rollout_details.jsonl"
        self.status_file_path = self.output_dir / "status.txt"

        self.stats_manager = StatisticsManager(self.args.rollout_input_batch_size, self.args.rollout_n)

        self._load_status()

    def get_active_tasks_count(self) -> int:
        return self._max_concurrency - self.semaphore._value

    def get_client(self) -> AsyncOpenAI:
        # Returns an AsyncOpenAI client instance
        return next(self.clients)

    def _save_status(self, batch_index):
        """Save current processing status to file"""
        self.processed_set.add(batch_index)
        content = self.processed_set.to_file_format()
        with open(self.status_file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _load_status(self):
        """Load processing status from file"""
        try:
            with open(self.status_file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                self.processed_set = RangeSet.from_file(content)
                logger.info(f"Resumed processed ranges: {self.processed_set.to_file_format()}")
        except FileNotFoundError:
            self.processed_set = RangeSet([])

    def process_data(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing data from {file_path}...")
        start_time = time.time()
        df = pd.read_parquet(file_path)
        if self.args.limit_rows != -1:
            df = df.iloc[:self.args.limit_rows]
        logger.info(f"Loaded {len(df)} samples in {time.time() - start_time:.2f}s")
        return df

    def batch_process(self, dataframe: pd.DataFrame):
        batch_prompts = []
        for idx, prompt in enumerate(dataframe[self.args.prompt_key]):
            batch_prompts.append(
                RequestPayload(prompt=prompt[0]["content"], idx=idx, num_responses=self.args.rollout_n)
            )
            if len(batch_prompts) == self.args.rollout_input_batch_size:
                yield batch_prompts
                batch_prompts = []

    async def call(self, request: RequestPayload) -> Tuple[str, float]:
        client = self.get_client()
        try:
            async with self.semaphore:
                # logger.debug("client is : %s", client.base_url)
                # logger.debug(f"There are currently {self.get_active_tasks_count()} asynchronous tasks working")
                start_time = time.perf_counter()
                response = await client.completions.create(
                    model=self.model,
                    prompt=request.prompt,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_response_length,
                    n=1,
                    stream=True,
                ) 
                # Streaming text is stored in a list of chunks
                chunks = []
                # Streaming responses
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].text:
                        chunks.append(chunk.choices[0].text)
                text = "".join(chunks)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.debug("Streaming response took %.2f seconds", elapsed_time)
                return text, round(elapsed_time, 2)

        except Exception as e:
            logger.error("Error while streaming: %s", e)
            raise ValueError(e)

    async def group_call(self, request: RequestPayload) -> ResponsePayload:
        """Performs n complete token generation rollouts for the given query."""
        tasks = [self.call(request) for _ in range(request.num_responses)]

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

    async def batch_call(self, requests: List[RequestPayload]) -> Tuple[List[ResponsePayload], int]:
        """Batch execution requests"""
        start_time = time.perf_counter()
        batch_results = await asyncio.gather(*[self.group_call(request) for request in requests])
        end_time = time.perf_counter()
        batch_elapsed_time = end_time - start_time
        logger.debug("total batch took %.4f seconds", batch_elapsed_time)
        return batch_results, batch_elapsed_time

    def execute(self):
        dataframe = self.process_data(self.args.input_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.global_stats_path, "a", newline="") as global_f, open(
            self.dispersed_stats_path, "a", newline=""
        ) as dispersed_f, open(self.rollout_details_path, "a", encoding="utf-8") as jsonl_f:
            global_writer = csv.writer(global_f)
            dispersed_writer = csv.writer(dispersed_f)

            if self.processed_set.processed_count <= 0:
                global_writer.writerow(
                    [
                        "batch_index",
                        "min_response_tokens",
                        "max_response_tokens",
                        "avg_response_tokens",
                        "total_response_tokens",
                        "group_max_response_tokens",
                        # "min_time",
                        # "avg_time",
                        "completion_time",
                        "throughput_tokens_per_sec",
                    ]
                )
                dispersed_writer.writerow(
                    [
                        "batch_index",
                        "rollout_lengths",
                        "min_length",
                        "max_length",
                        "avg_length",
                        "completion_time",
                        "throughput_tokens_per_sec",
                        # "elapsed_times",
                        # "min_time",
                        # "max_time",
                        # "avg_time",
                    ]
                )

            for batch_index, input_ids in tqdm(
                enumerate(self.batch_process(dataframe)),
                total=math.ceil(len(dataframe) / self.args.rollout_input_batch_size),
            ):
                if self.processed_set.contains(batch_index):
                    continue

                self.stats_manager.batch_index = batch_index
                batch_results, batch_elapsed_time = asyncio.run(self.batch_call(input_ids))

                for i in range(len(batch_results)):
                    batch_results[i] = self.tokenizer.tokenize(batch_results[i])

                global_stats_dict, dispersed_stats_dict = self.stats_manager.global_stats(
                    batch_results, batch_elapsed_time
                )

                global_writer.writerow(
                    [
                        batch_index,
                        global_stats_dict["min_response_tokens"],
                        global_stats_dict["max_response_tokens"],
                        round(global_stats_dict["avg_response_tokens"], 2),
                        global_stats_dict["total_response_tokens"],
                        global_stats_dict["group_max_response_tokens"],
                        # global_stats_dict["min_time"],
                        # global_stats_dict["avg_time"],
                        round(global_stats_dict["completion_time"], 2),
                        round(global_stats_dict["throughput_tokens_per_sec"], 2),
                    ]
                )

                dispersed_writer.writerow(
                    [
                        batch_index,
                        dispersed_stats_dict["rollout_lengths"],
                        dispersed_stats_dict["min_length"],
                        dispersed_stats_dict["max_length"],
                        dispersed_stats_dict["avg_length"],
                        round(dispersed_stats_dict["completion_time"], 2),
                        round(dispersed_stats_dict["throughput_tokens_per_sec"], 2),
                        # dispersed_stats_dict["elapsed_times"],
                        # dispersed_stats_dict["min_time"],
                        # dispersed_stats_dict["max_time"],
                        # dispersed_stats_dict["avg_time"],
                    ]
                )

                record = [
                    {
                        "batch_index": batch_index,
                        "prompt_text": result.question,
                        "rollouts": [
                            {"response": res, "token_length": token_length}
                            for res, token_length in zip(result.responses, result.token_lengths)
                        ],
                        "total_time": dispersed_stats_dict["completion_time"],
                        "throughput_tokens_per_sec": dispersed_stats_dict["throughput_tokens_per_sec"],
                    }
                    for result in batch_results
                ]

                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                global_f.flush()
                dispersed_f.flush()
                jsonl_f.flush()
                self._save_status(batch_index)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Process prompts with OpenAI clients.")
    parser.add_argument("--openai_urls", type=str, nargs="+", required=True, help="List of OpenAI service URLs")
    parser.add_argument(
        "--api_keys", type=str, nargs="+", default=None, help="List of API keys (default: 'NONE' for each service)"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Qwen2.5-7B-Instruct-1M)")
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Tokenizer name (e.g., Qwen/Qwen2.5-7B-Instruct-1M)"
    )
    parser.add_argument("--rollout_input_batch_size", type=int, default=4, help="Batch size for requests")
    parser.add_argument("--rollout_n", type=int, default=8, help="Number of responses per request")
    parser.add_argument(
        "--prompt_key", type=str, default="prompt", help="Key in the DataFrame for prompts (default: 'prompt')"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input Parquet file")
    parser.add_argument(
        "--output_dir", type=str, default="./api_infer_results", help="Directory for output CSV files (default: './api_infer_results')"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter for text generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter for text generation")
    parser.add_argument("--max_prompt_length", type=int, default=1024 * 2, help="Maximum prompt length (in tokens)")
    parser.add_argument(
        "--max_response_length", type=int, default=1024 * 2, help="Maximum response length (in tokens)"
    )
    parser.add_argument("--limit_rows", type=int, default=-1, help="Maximum number of rows to read from the dataset (-1 means all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task = ApiTask(args)
    task.execute()
