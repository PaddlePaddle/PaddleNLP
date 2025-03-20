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
import re
from dataclasses import dataclass, field

import paddle
from grader import math_equal
from paddle.distributed import fleet

from llm.predict.predictor import (
    ModelArgument,
    PredictorArgument,
    batchfy_text,
    create_predictor,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger


@dataclass
class EvalArgument:
    eval_file: str = field(
        default="gsm8k",
        metadata={"help": "the name of dataset for evaluation. Supported values: aime2024, gsm8k, math500"},
    )
    eval_question_key: str = field(default="input_ids", metadata={"help": "the question key of dataset"})
    eval_answer_key: str = field(default="output_ids", metadata={"help": "the answer key of dataset"})
    eval_prompt: str = field(
        default="\nPlease reason step by step, and put your final answer within \\boxed{}.",
        metadata={"help": "the prompt used during evaluation"},
    )

    eval_results: str = field(default="output.json", metadata={"help": "predict result file directory"})


def extract_answer(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    final_answer = solution.group(0)
    final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    return final_answer


def extract_solution(solution_str, just_last_number=False):
    """Extract the answer number from the sentence using regular expressions."""
    # Remove commas for easier extraction
    sentence = solution_str.replace(",", "")
    # Find all numbers in the sentence
    # 提取boxed{}中的任意值
    pattern = r"boxed\{(.*)\}"
    numbers = [s for s in re.findall(pattern, sentence)]

    # when boxed{} has not results, try fetch last number as result.
    if not numbers:
        pattern = r"-?\d+\.?\d*"
        numbers = [s for s in re.findall(pattern, sentence)]

    if not numbers:
        return None  # Return 'inf' if no number is found
    else:
        # Return the last number found as a float
        return str(numbers[-1])


def predict():
    parser = PdArgumentParser((PredictorArgument, ModelArgument, EvalArgument))
    predictor_args, model_args, eval_args = parser.parse_args_into_dataclasses()

    llm_utils.set_triton_cache(predictor_args.model_name_or_path, predictor_args.mode)

    tensor_parallel_degree = paddle.distributed.get_world_size()
    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    predictor = create_predictor(predictor_args, model_args)

    source_texts = []
    target_texts = []
    assert eval_args.eval_file is not None, "eval_file is None, please set a file (.json or .jsonl) to eval"

    with open(eval_args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            source_texts.append(example[eval_args.eval_question_key] + eval_args.eval_prompt)
            target_texts.append(example[eval_args.eval_answer_key])

    batch_source_texts = batchfy_text(source_texts, predictor_args.batch_size)
    batch_target_texts = batchfy_text(target_texts, predictor_args.batch_size)

    with open(eval_args.eval_results, "w", encoding="utf-8") as f:
        cnt, bad_format = 0, 0

        for bs, batch_source_text in enumerate(batch_source_texts):
            # logger.info("Start predict")
            outputs = predictor.predict(batch_source_text)
            # logger.info("End predict")

            if predictor.tensor_parallel_rank > 0:
                continue

            for output, source, target in zip(outputs, batch_source_texts[bs], batch_target_texts[bs]):
                target_answer = extract_solution(target, just_last_number=True)
                output_answer = extract_solution(output)
                equal = False
                if output_answer is None:
                    bad_format += 1
                else:
                    equal = math_equal(output_answer, target_answer)
                cnt = cnt + 1 if equal else cnt

                logger.info("***********Source**********")
                logger.info(source)
                logger.info("***********Target**********")
                logger.info(target_answer)
                logger.info("***********Output**********")
                logger.info(output_answer)
                logger.info("***********IS EQUAL**********")
                logger.info(equal)

                out = {
                    "src": source,
                    "tgt": target,
                    "output": output,
                    "output_answer": output_answer,
                    "is_equal": equal,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        f.write(f"accuracy: {cnt / len(target_texts)}")
        logger.info(f"accuracy: {cnt / len(target_texts)}")


if __name__ == "__main__":
    predict()
