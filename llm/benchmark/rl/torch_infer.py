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
import csv
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import tqdm
from transformers import AutoTokenizer, logging
from utils import RangeSet
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
        help="Name or path of the pretrained model",
    )
    parser.add_argument("--input_file", type=str, default="./combined.parquet", help="Path to input parquet file")
    parser.add_argument("--output_dir", type=str, default="./pt_infer_results", help="Path to output directory")
    parser.add_argument(
        "--rollout_input_batch_size", type=int, default=2, help="Batch size for each inference engine input"
    )
    parser.add_argument(
        "--rollout_n", type=int, default=2, help="Number of repeated inference runs (for performance testing)"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for computation")
    parser.add_argument("--tensor_parallel_degree", type=int, default=1, help="Degree of model parallelism")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter for text generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter for text generation")
    parser.add_argument("--should_log", type=bool, default=True, help="Whether to print logging information")
    parser.add_argument("--max_src_len", type=int, default=1024 * 2, help="Maximum prompt length (in tokens)")
    parser.add_argument("--max_dec_len", type=int, default=1024 * 2, help="Maximum response length (in tokens)")
    parser.add_argument("--min_dec_len", type=int, default=0, help="Minimum response length (in tokens)")
    parser.add_argument(
        "--limit_rows", type=int, default=-1, help="Maximum number of rows to read from the dataset (-1 means all)"
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Percentage of GPU memory usage")
    args = parser.parse_args()

    return args


def chunk(all_input_ids, size):
    if size <= 0:
        raise ValueError("Size must be greater than 0")
    return [all_input_ids[i : i + size] for i in range(0, len(all_input_ids), size)]


class DumpyInferenceTask:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_stats_path = self.output_dir / "global_stats.csv"
        self.dispersed_stats_path = self.output_dir / "dispersed_stats.csv"
        self.rollout_details_path = self.output_dir / "rollout_details.jsonl"
        self.status_file_path = self.output_dir / "status.txt"

        self._load_status()
        self._load_model()
        self._prepare_tokenizer()

    def _load_status(self):
        """Load processing status from file"""
        try:
            with open(self.status_file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                self.processed_set = RangeSet.from_file(content)
                logger.info(f"Resumed processed ranges: {self.processed_set.to_file_format()}")
        except FileNotFoundError:
            self.processed_set = RangeSet([])

    def _save_status(self, batch_index):
        """Save current processing status to file"""
        self.processed_set.add(batch_index)
        content = self.processed_set.to_file_format()
        with open(self.status_file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _load_model(self):
        logger.info(f"Loading model from {self.args.actor_model_name_or_path}...")
        start_time = time.time()
        self.model = LLM(
            model=self.args.actor_model_name_or_path,
            dtype=self.args.dtype,
            enable_sleep_mode=False,
            tensor_parallel_size=self.args.tensor_parallel_degree,
            # distributed_executor_backend="external_launcher",
            enforce_eager=False,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=self.args.max_src_len + self.args.max_dec_len,
            disable_log_stats=True,
            max_num_batched_tokens=self.args.max_src_len + self.args.max_dec_len,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
        )
        self.sampling_params = SamplingParams(
            top_p=self.args.top_p,
            temperature=self.args.temperature,
            min_tokens=self.args.min_dec_len,
            max_tokens=self.args.max_dec_len,
            n=self.args.rollout_n,
            seed=42,
        )
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")

    def _prepare_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.actor_model_name_or_path, use_fast=True)
        self.tokenizer.padding_side = "left"

    def process_data(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Processing data from {file_path}...")
        start_time = time.time()
        df = pd.read_parquet(file_path)
        if self.args.limit_rows != -1:
            df = df.iloc[: self.args.limit_rows]
        logger.info(f"Loaded {len(df)} samples in {time.time() - start_time:.2f}s")
        return df

    def run_inference(self, prompts, batch_index=0):
        input_ids = [self.tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
        token_prompt_ids = [TokensPrompt(prompt_token_ids=prompt) for prompt in input_ids]

        start_time = time.time()
        request_outputs = self.model.generate(
            prompts=token_prompt_ids, sampling_params=self.sampling_params, use_tqdm=True
        )
        end_time = time.time()

        batch_token_ids = []
        for output in request_outputs:
            for each_output in output.outputs:
                batch_token_ids.append(each_output.token_ids)

        if self.args.should_log:
            statistics = self.postprocess_data(input_ids, batch_token_ids, batch_index=batch_index)
            statistics["total_time"] = end_time - start_time
            return statistics
        return None

    def postprocess_data(self, input_ids, output_ids, batch_index=0):
        global_prompt_tokens = []
        global_prompt_tokens_len = []
        group_prompt_texts = []
        for row in input_ids:
            row_ids = row
            global_prompt_tokens.append(row_ids)
            global_prompt_tokens_len.append(len(row_ids))
            group_prompt_texts.append(self.tokenizer.decode(row_ids, skip_special_tokens=False))

        # Process responses
        response_tokens = []
        response_tokens_len = []
        response_texts = []
        for row in output_ids:
            row_ids = row
            response_tokens.append(row_ids)
            response_tokens_len.append(len(row_ids))
            response_texts.append(self.tokenizer.decode(row_ids, skip_special_tokens=False))

        # Group processing
        group_response_tokens = chunk(response_tokens, self.args.rollout_n)
        group_response_tokens_len = chunk(response_tokens_len, self.args.rollout_n)
        group_response_texts = chunk(response_texts, self.args.rollout_n)

        # Calculate statistics
        global_response_tokens_total = sum(response_tokens_len)
        global_response_tokens_len_min = min(response_tokens_len)
        global_response_tokens_len_max = max(response_tokens_len)
        global_response_tokens_len_mean = round(global_response_tokens_total / len(response_tokens_len), 2)

        group_response_tokens_len_mean = [round(sum(x) / len(x), 2) for x in group_response_tokens_len]
        group_response_tokens_len_max = [max(x) for x in group_response_tokens_len]
        group_response_tokens_len_min = [min(x) for x in group_response_tokens_len]

        return {
            # Global prompt stats
            "batch_index": batch_index,
            "global_prompt_tokens_len": global_prompt_tokens_len,
            "group_prompt_texts": group_prompt_texts,
            # Group response stats
            "group_response_tokens": group_response_tokens,
            "group_response_tokens_len": group_response_tokens_len,
            "group_response_texts": group_response_texts,
            "group_response_tokens_len_min": group_response_tokens_len_min,
            "group_response_tokens_len_max": group_response_tokens_len_max,
            "group_response_tokens_len_mean": group_response_tokens_len_mean,
            # Global response stats
            "global_response_tokens_total": global_response_tokens_total,
            "global_response_tokens_len_min": global_response_tokens_len_min,
            "global_response_tokens_len_max": global_response_tokens_len_max,
            "global_response_tokens_len_mean": global_response_tokens_len_mean,
        }

    def batch_process(self, dataframe: pd.DataFrame):
        all_prompts = []

        for idx, row in dataframe.iterrows():
            prompt = row["prompt"].tolist()[0]["content"]
            all_prompts.append(prompt)

            if len(all_prompts) >= self.args.rollout_input_batch_size:
                yield all_prompts
                all_prompts = []

        if all_prompts:
            yield all_prompts

    def execute(self):
        dataframe = self.process_data(self.args.input_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.global_stats_path, "a", newline="") as global_f, open(
            self.dispersed_stats_path, "a", newline=""
        ) as dispersed_f, open(self.rollout_details_path, "a", encoding="utf-8") as jsonl_f:

            if self.args.should_log:
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
                        ]
                    )

            for batch_index, prompts in tqdm.tqdm(
                enumerate(self.batch_process(dataframe)),
                total=math.ceil(len(dataframe) / self.args.rollout_input_batch_size),
                disable=not self.args.should_log,
            ):
                if self.processed_set.contains(batch_index):
                    continue

                statistics = self.run_inference(prompts, batch_index=batch_index)

                if self.args.should_log:
                    total_time = round(statistics["total_time"], 2)
                    total_tokens = statistics["global_response_tokens_total"]
                    throughput = round(total_tokens / total_time if total_time > 0 else 0, 2)

                    global_writer.writerow(
                        [
                            batch_index,
                            statistics["global_response_tokens_len_min"],
                            statistics["global_response_tokens_len_max"],
                            statistics["global_response_tokens_len_mean"],
                            total_tokens,
                            statistics["group_response_tokens_len_max"],
                            total_time,
                            throughput,
                        ]
                    )

                    dispersed_writer.writerow(
                        [
                            batch_index,
                            statistics["group_response_tokens_len"],
                            statistics["group_response_tokens_len_min"],
                            statistics["group_response_tokens_len_max"],
                            statistics["group_response_tokens_len_mean"],
                            total_time,
                            throughput,
                        ]
                    )

                    prompt_text = statistics["group_prompt_texts"]
                    response_tokens = statistics["group_response_tokens"]
                    response_texts = statistics["group_response_texts"]

                    record = {
                        "batch_index": batch_index,
                        "prompt_text": prompt_text,
                        "rollouts": [
                            {"token_ids": tokens, "text": text, "length": len(tokens)}
                            for tokens, text in zip(response_tokens, response_texts)
                        ],
                        "total_time": total_time,
                        "throughput_tokens_per_sec": throughput,
                    }
                    jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    global_f.flush()
                    dispersed_f.flush()
                    jsonl_f.flush()
                    self._save_status(batch_index)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    task = DumpyInferenceTask(args)
    task.execute()
