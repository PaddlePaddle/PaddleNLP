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
from typing import Dict, List, Optional, Tuple

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


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.

    Args:
        solution_text: Formatted solution text from dataset

    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")

    for line in solution_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.search(r"\b([A-Za-z]+)\b.*?\b(knight|knave)\b", line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")

    return status_dict


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.

    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification

    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count("knight")
    knave_count = answer_text.lower().count("knave")

    print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(rf"\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b", re.IGNORECASE)
        match = pattern.search(answer_text)

        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None

    return status_dict


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        "think_start": ("<think>", 1),
        "think_end": ("</think>", 1),
        "answer_start": ("<answer>", 1),
        "answer_end": ("</answer>", 1),
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (
        positions["think_start"] > positions["think_end"]
        or positions["think_end"] > positions["answer_start"]
        or positions["answer_start"] > positions["answer_end"]
    ):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def compute_score(
    solution_str: str, ground_truth: str, query=None, format_reward: int = 1, answer_reward: float = 1.0
):
    """Computes comprehensive score for model response.

    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness

    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "=" * 80)
    print(" Processing New Sample ".center(80, "="))

    if "\n<|im_start|>assistant\n<think>" not in solution_str:
        solution_str = "\n<|im_start|>assistant\n<think>" + solution_str

    # Parse ground truth data
    solution_text = ground_truth
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final identities: {gt_status}")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    if format_correct and answer_text:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            print("\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")

            if pred_status == gt_status:
                answer_score = 2
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -1.5
                print("  Content validation: MISMATCH")
        else:
            answer_score = -2
            print("Fail to parse answer")
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-" * 80)
    print(" Final Score ".center(80, "-"))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("=" * 80 + "\n")

    return float(total_score)


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
