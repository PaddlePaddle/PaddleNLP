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
import json
import logging
import subprocess
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import aiofiles
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


@dataclass
class RangeSet:
    """Manage processed line ranges with efficient storage and querying"""

    ranges: List[tuple]

    def add(self, number: int):
        """Add a number to the range set and merge adjacent ranges"""
        new_ranges = []
        added = False
        for start, end in sorted(self.ranges):
            if number < start - 1:
                if not added:
                    new_ranges.append((number, number))
                    added = True
                new_ranges.append((start, end))
            elif number == start - 1:
                new_ranges.append((number, end))
                added = True
            elif number <= end:
                new_ranges.append((start, end))
                added = True
            else:
                new_ranges.append((start, end))
        if not added:
            new_ranges.append((number, number))
        self.ranges = self.merge_ranges(new_ranges)

    @staticmethod
    def merge_ranges(ranges: List[tuple]) -> List[tuple]:
        """Merge overlapping or adjacent ranges"""
        if not ranges:
            return []
        sorted_ranges = sorted(ranges)
        merged = [sorted_ranges[0]]
        for current in sorted_ranges[1:]:
            last = merged[-1]
            if current[0] <= last[1] + 1:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        return merged

    def contains(self, number: int) -> bool:
        """Check if a number exists in any range"""
        for start, end in self.ranges:
            if start <= number <= end:
                return True
        return False

    def to_file_format(self) -> str:
        """Serialize ranges to compact string format"""
        return ",".join(f"{start}-{end}" if start != end else str(start) for start, end in self.ranges)

    @classmethod
    def from_file(cls, content: str) -> "RangeSet":
        """Deserialize from string format"""
        if not content:
            return cls(ranges=[])
        ranges = []
        for part in content.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                ranges.append((start, end))
            else:
                num = int(part)
                ranges.append((num, num))
        return cls(ranges=ranges)

    @property
    def processed_count(self) -> int:
        """Total number of processed items"""
        return sum(end - start + 1 for start, end in self.ranges)


class OpenAIClientPool:
    """Manage round-robin distribution of API clients"""

    def __init__(self, base_urls: List[str], api_keys: List[str] = None):
        if isinstance(base_urls, str):
            base_urls = base_urls.split(",")
        if isinstance(api_keys, str):
            api_keys = api_keys.split(",")

        if api_keys is None:
            api_keys = ["NONE" for _ in range(len(base_urls))]

        if len(api_keys) != len(base_urls):
            raise ValueError("API keys and base URLs should have the same length!")

        self.clients = cycle([AsyncOpenAI(base_url=url, api_key=key) for url, key in zip(base_urls, api_keys)])

    def get_client(self) -> AsyncOpenAI:
        """Get next available client in rotation"""
        return next(self.clients)


class OpenAIProcessor:
    """Async processor for batch processing with OpenAI-compatible APIs"""

    def __init__(
        self,
        input_file: str,
        output_file: str,
        prompt_key: str,
        base_urls: List[str],
        api_keys: List[str] = None,
        prompt_suffix: str = "",
        status_file: str = "status.txt",
        concurrency: int = 8,
        model: str = "deepseek-r1",
        temperature: float = 0.6,
        top_p: float = 1.0,
        max_tokens: int = 65536,
        timeout: int = 3600,
        response_key: str = "response",
        reasoning_key: str = "reasoning",
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.client_pool = OpenAIClientPool(base_urls, api_keys)
        self.status_file = status_file
        self.processed_set = RangeSet([])
        self.concurrency = concurrency
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.total_lines = 0
        self.progress_bar = None
        self.write_lock = asyncio.Lock()
        self.status_lock = asyncio.Lock()
        self.prompt_key = prompt_key
        self.prompt_suffix = prompt_suffix
        self.timeout = timeout
        self.response_key = response_key
        self.reasoning_key = reasoning_key

        self._load_status()

    def _load_status(self):
        """Load processing status from file"""
        try:
            with open(self.status_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                self.processed_set = RangeSet.from_file(content)
                logger.info(f"Resumed processed ranges: {self.processed_set.to_file_format()}")
        except FileNotFoundError:
            self.processed_set = RangeSet([])

    async def _save_status(self):
        """Save current processing status to file"""
        async with self.status_lock:
            content = self.processed_set.to_file_format()
            async with aiofiles.open(self.status_file, "w", encoding="utf-8") as f:
                await f.write(content)

    def _count_total_lines(self) -> int:
        """Count total lines in input file"""
        try:
            result = subprocess.run(
                ["wc", "-l", self.input_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return int(result.stdout.strip().split()[0])
        except Exception as e:
            print(f"Failed to count lines using `wc -l` command: {str(e)}")
            with open(self.input_file, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)

    async def _line_generator(self):
        """Generate unprocessed lines with line numbers"""
        self.total_lines = self._count_total_lines()
        with open(self.input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not self.processed_set.contains(line_num):
                    yield line_num, json.loads(line.strip())

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=60))
    async def _call_openai(self, client: AsyncOpenAI, line_num: int, data: Dict[str, str]) -> str:
        """Execute API call with retry logic"""
        try:
            # Prepend processing instructions to the content
            content = data.get(self.prompt_key, "")
            if isinstance(content, (tuple, list)):
                content = content[0]

            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content + self.prompt_suffix}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                stream=False,
            )
            return {"line_num": line_num, **data, **self._parse_response(response)}

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            if "rate limit" in str(e).lower():
                await asyncio.sleep(5)
            raise

    def _parse_response(self, response) -> Dict[str, str]:
        """Parse API response into structured format"""
        response_text = response.result
        reasoning_text = ""

        if "</think>" in response_text and response_text.count("</think>") == 1:
            reasoning_text, _, response_text = response_text.partition("</think>")

        if reasoning_text and not reasoning_text.startswith("<think>"):
            reasoning_text = f"<think>\n{reasoning_text.strip()}"
        if reasoning_text and not reasoning_text.endswith("</think>"):
            reasoning_text = f"{reasoning_text.strip()}\n</think>"

        return {
            self.response_key: response_text,
            self.reasoning_key: reasoning_text,
        }

    async def _write_result(self, line_num: int, result: Dict[str, str]):
        """Write processed result and update status"""
        async with self.write_lock:
            # Append result to output file
            async with aiofiles.open(self.output_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Update processing status
            self.processed_set.add(line_num)
            await self._save_status()

    async def worker(self, queue: asyncio.Queue):
        """Process items from the queue"""
        client = self.client_pool.get_client()
        while True:
            line_num, data = await queue.get()
            try:
                result = await self._call_openai(client, line_num, data)
                await self._write_result(line_num, result)
                self.progress_bar.update(1)
            except Exception as e:
                logger.error(f"Failed to process line {line_num}: {str(e)}")
            finally:
                queue.task_done()

    async def run(self):
        """Main processing loop"""
        total_lines = self._count_total_lines()
        remaining = total_lines - self.processed_set.processed_count

        if remaining <= 0:
            logger.info("No data requires distilling!")
            return

        # Initialize progress bar with current progress
        self.progress_bar = tqdm(
            total=total_lines,
            desc=f"[{self.model}] Data Distilling Progress",
            dynamic_ncols=True,
            initial=self.processed_set.processed_count,
        )

        queue = asyncio.Queue(maxsize=self.concurrency * 2)

        # Ensure output file exists
        async with aiofiles.open(self.output_file, "a", encoding="utf-8"):
            pass

        # Start worker tasks
        workers = [asyncio.create_task(self.worker(queue)) for _ in range(self.concurrency)]

        try:
            # Feed unprocessed items to queue
            async for line_num, data in self._line_generator():
                await queue.put((line_num, data))

            await queue.join()
        finally:
            # Cleanup resources
            for worker_task in workers:
                worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self.progress_bar.close()
            logger.info(f"Processing complete. Total processed: {self.processed_set.processed_count} / {total_lines}.")
            logger.info("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="OpenAI Multi-Processing Interface")
    parser.add_argument("--input_file", type=Path, required=True, help="Input JSONL filename")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory path")
    parser.add_argument("--prompt_key", required=True, help="Key name for the input JSONL data prompt")
    parser.add_argument("--response_key", required=True, help="Key name for the output JSONL data response")
    parser.add_argument("--reasoning_key", required=True, help="Key name for the output JSONL data reasoning")
    parser.add_argument("--base_urls", required=True, help="Comma-separated list of API endpoints")
    parser.add_argument("--api_keys", default=None, help="Comma-separated list of API keys, Default: `None`")
    parser.add_argument("--model", default="deepseek-r1", type=str, help="Model name to use, Default: `deepseek-r1`")
    parser.add_argument("--prompt_suffix", default="", type=str, help="Suffix appended after each prompt, Default: ``")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature, Default: `0.6`")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling value, Default: `1.0`")
    parser.add_argument("--timeout", type=int, default=3600, help="API request timeout in seconds, Default: `3600`s")
    parser.add_argument(
        "--max_tokens", type=int, default=65536, help="Maximum number of tokens to generate, Default: `65536`"
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Maximum number of concurrent threads, Default: `8`"
    )
    parser.add_argument("--status_file", default=None, help="Status file path, Default: `None`")
    parser.add_argument("--logging_file", default=None, help="Logging file path, Default: `None`")
    args = parser.parse_args()

    if not str(args.input_file).endswith(".jsonl"):
        raise NotImplementedError("Currently only JSONL files are supported!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    filename_prefix = "distilled-" + args.input_file.stem
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.logging_file or args.output_dir / f"{filename_prefix}.log"),
        ],
    )
    PROCESSOR_CONFIG = {
        "input_file": args.input_file,
        "output_file": args.output_dir / f"{filename_prefix}.jsonl",
        "status_file": args.status_file or args.output_dir / f"{filename_prefix}.status",
        "prompt_key": args.prompt_key,
        "response_key": args.response_key,
        "reasoning_key": args.reasoning_key,
        "prompt_suffix": args.prompt_suffix,
        "base_urls": args.base_urls,
        "api_keys": args.api_keys,
        "concurrency": args.concurrency,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
    }
    processor = OpenAIProcessor(**PROCESSOR_CONFIG)
    asyncio.run(processor.run())


if __name__ == "__main__":
    main()
