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

import json
import os
import re

from grader import math_equal

# dataset_name = "aime2024"
dataset_name = "gsm8k"
# dataset_name = "math500"

test_dataset_path = f"./data/{dataset_name}/dev.json"
output_dataset_path = f"./results-{dataset_name}/Qwen/Qwen2.5-Math-7B/output.json"
output_result_path = os.path.join(os.path.dirname(output_dataset_path), f"{dataset_name}.log")
IS_BASE = False


def extract_answer(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    final_answer = solution.group(0)
    final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    return final_answer


def extract_solution(solution_str, base=False):
    """Extract the answer number from the sentence using regular expressions."""
    # Remove commas for easier extraction
    sentence = solution_str.replace(",", "")
    # Find all numbers in the sentence
    if base:
        pattern = r"-?\d+\.?\d*"
        numbers = [s for s in re.findall(pattern, sentence)]
    else:
        if dataset_name in ["aime2024", "gsm8k"]:
            pattern = r"boxed\{([0-9]+(\.[0-9]+)?)"
            numbers = [s for s in re.findall(pattern, sentence)]
        elif dataset_name == "math500":
            # 提取boxed{}中的任意值
            pattern = r"boxed\{(.*)\}"
            numbers = [s for s in re.findall(pattern, sentence)]

    if not numbers:
        return None  # Return 'inf' if no number is found
    else:
        # Return the last number found as a float
        if dataset_name == "math500":
            return str(numbers[-1]) if not base else str(numbers[-1])
        else:
            return str(numbers[-1][0]) if not base else str(numbers[-1])


ground, solution = [], []
with open(test_dataset_path, "r") as f:
    for line in f.readlines():
        line = line.strip()
        if not line:
            continue
        jsline = json.loads(line)
        if "answer" in jsline.keys():
            ground.append(str(jsline["answer"]))
        else:
            ground.append(extract_answer(jsline["tgt"]))


bad_format = 0
with open(output_dataset_path, "r") as f:
    for line in f.readlines():
        line = line.strip()
        if not line:
            continue
        jsline = json.loads(line)
        solution.append(extract_solution(jsline["output"], base=IS_BASE))
        if solution[-1] is None:
            bad_format += 1

print(f"after line, bad_format= {bad_format}")

# ground = ground[:len(solution)]
# print(solution)
# breakpoint()

assert len(ground) == len(solution)

cnt = 0
with open(output_result_path, "w") as f:
    for idx, (i, j) in enumerate(zip(ground, solution)):
        if math_equal(i, j):
            cnt += 1
        else:
            print(f"{idx}: ground: {i}, answer: {j}", flush=True)
            f.write(f"{idx}: ground: {i}, answer: {j} \n")

    print(
        f"accuracy: {cnt / len(ground)}, right: {cnt}, format_error: {bad_format}, answer_error: {len(ground) - cnt - bad_format}"
    )
    f.write(
        f"accuracy: {cnt / len(ground)}, right: {cnt}, format_error: {bad_format}, answer_error: {len(ground) - cnt - bad_format}"
    )
