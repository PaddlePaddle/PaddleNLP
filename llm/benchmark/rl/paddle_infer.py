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

import csv
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import paddle
import pandas as pd
from tqdm import tqdm
from utils import RangeSet

from paddlenlp.rl.trainer import process_row
from paddlenlp.rl.utils import TrainingArguments
from paddlenlp.rl.utils.infer_utils import get_policy_predictor, infer_guard
from paddlenlp.trainer import PdArgumentParser, set_seed
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.trl.llm_utils import init_dist_env
from paddlenlp.utils.log import logger


@contextmanager
def switch_level_context(level="ERROR"):
    original_level = logger.logLevel
    logger.set_level(level)

    try:
        yield
    finally:
        logger.set_level(original_level)


def chunk(all_input_ids, size):
    if size <= 0:
        raise ValueError("Size must be greater than 0")
    return [all_input_ids[i : i + size] for i in range(0, len(all_input_ids), size)]


@dataclass
class DumpyTrainingArguments(TrainingArguments):
    actor_model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct-1M", metadata={"help": "pretrained model name or path"}
    )
    input_file: str = field(
        default="kk/instruct/3-7ppl/combined.parquet", metadata={"help": "input the Parquet file path"}
    )
    limit_rows: int = field(
        default=-1, metadata={"help": "Maximum number of rows to read from the dataset (-1 means all)"}
    )
    output_dir: str = field(default="./pt_infer_results", metadata={"help": "output directory path"})
    rollout_input_batch_size: int = field(default=32, metadata={"help": "batch size for inference engine inputs"})
    rollout_n: int = field(default=8, metadata={"help": "number of rollouts (for performance testing)"})
    log_interval: int = field(default=1, metadata={"help": "logging interval (in batches)"})

    def __post_init__(self):
        super().__post_init__()
        self.dtype = "bfloat16"
        if self.rollout_quant_type.lower() in ["bfloat16", "bf16"]:
            self.rollout_quant_type = ""
        elif self.rollout_quant_type.lower() in ["wint8", "weight_only_int8"]:
            self.rollout_quant_type = "weight_only_int8"


class DumpyInferenceTask:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ​​Initialize output file path​
        self.global_stats_path = self.output_dir / "global_stats.csv"
        self.dispersed_stats_path = self.output_dir / "dispersed_stats.csv"
        self.rollout_details_path = self.output_dir / "rollout_details.jsonl"
        self.status_file_path = self.output_dir / "status.txt"

        self._init_environment()
        self._load_status()
        self._load_model()
        self._prepare_tokenizer()

    def _init_environment(self):
        self.tensor_parallel_rank, self.tensor_parallel_degree = init_dist_env()
        self.use_fusemt = True
        self.amp_dtype = self.args.dtype

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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.actor_model_name_or_path,
            dtype=self.args.dtype,
            low_cpu_mem_usage=True,
            tensor_parallel_degree=self.tensor_parallel_degree,
            tensor_parallel_rank=self.tensor_parallel_rank,
        )
        self.model.eval()
        self.model.to(device=paddle.CUDAPinnedPlace())
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

    @switch_level_context(level="ERROR")
    def run_inference(self, input_ids, batch_index=0):
        set_seed(42)
        start_time = time.time()
        output_ids = get_policy_predictor().predict(
            input_ids=input_ids,
            repeat_num=self.args.rollout_n,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
        )
        if self.args.world_size > 1:
            paddle.distributed.barrier()
        end_time = time.time()
        if self.args.should_log:
            statistics = self.postprocess_data(input_ids, output_ids, batch_index=batch_index)
            statistics["total_time"] = end_time - start_time
            return statistics
        return None

    def postprocess_data(self, input_ids, output_ids, batch_index=0):
        # Process prompts
        global_prompt_tokens = []
        global_prompt_tokens_len = []
        group_prompt_texts = []
        for row in input_ids:
            row_ids = process_row(row, remove_value=self.tokenizer.pad_token_id, remove_side="both").tolist()
            global_prompt_tokens.append(row_ids)
            global_prompt_tokens_len.append(len(row_ids))
            group_prompt_texts.append(self.tokenizer.decode(row_ids, skip_special_tokens=False))

        # Process responses
        response_tokens = []
        response_tokens_len = []
        response_texts = []
        for row in output_ids:
            row_ids = process_row(row, remove_value=self.tokenizer.pad_token_id, remove_side="both").tolist()
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
        all_input_ids = []

        def _pad_batch(input_ids_list):
            return self.tokenizer.pad({"input_ids": input_ids_list}, padding=True, return_tensors="pd")["input_ids"]

        for idx, row in dataframe.iterrows():
            prompt = row["prompt"].tolist()[0]["content"]
            input_ids = self.tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            all_input_ids.append(input_ids)

            if len(all_input_ids) >= self.args.rollout_input_batch_size:
                yield _pad_batch(all_input_ids)
                all_input_ids = []

        if all_input_ids:
            yield _pad_batch(all_input_ids)

    def execute(self):
        """Main Execution Pipeline"""
        # Data Preparation​
        dataframe = self.process_data(self.args.input_file)

        # Create Output Directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.global_stats_path, "a", newline="") as global_f, open(
            self.dispersed_stats_path, "a", newline=""
        ) as dispersed_f, open(self.rollout_details_path, "a", encoding="utf-8") as jsonl_f:

            if self.args.should_log:
                # Initialize the CSV writer
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

            with infer_guard(self):
                for batch_index, input_ids in tqdm(
                    enumerate(self.batch_process(dataframe)),
                    total=math.ceil(len(dataframe) / self.args.rollout_input_batch_size),
                    disable=not self.args.should_log,
                ):
                    if self.processed_set.contains(batch_index):
                        continue

                    statistics = self.run_inference(input_ids, batch_index=batch_index)

                    if self.args.should_log:
                        # Write global statistics
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

                        # Write detailed records (one row per query)
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


def main():
    parser = PdArgumentParser((DumpyTrainingArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    task = DumpyInferenceTask(args)
    task.execute()


if __name__ == "__main__":
    main()
