# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import sys
from dataclasses import dataclass, field
from functools import partial

import paddle
from argument import (
    DataArgument,
    GenerateArgument,
    ModelArgument,
    QuantArgument,
    TrainingArguments,
)
from data import get_convert_example
from function_call.schema import load_messages_from_file
from tqdm import tqdm
from utils import (
    CausalLMTrainer,
    InTokensIterDatasetCallback,
    compute_metrics,
    get_lora_target_modules,
    get_prefix_tuning_params,
    init_chat_template,
)

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import InTokensIterableDataset, InTokensMapDataset, load_dataset
from paddlenlp.metrics import BLEU, Rouge1, Rouge2, RougeL
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.trainer.trainer_callback import TrainerState
from paddlenlp.trainer.trainer_utils import IntervalStrategy
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)
from paddlenlp.utils.log import logger


@dataclass
class EvalArguments(TrainingArguments):
    steps_generation: bool = field(default=False, metadata={"help": "Whether use step_generation to inference"})


def main():
    # Arguments
    parser = PdArgumentParser((GenerateArgument, QuantArgument, ModelArgument, DataArgument, TrainingArguments))
    # Support format as "args.json --arg1 value1 --arg2 value2.”
    # In case of conflict, command line arguments take precedence.
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        dtype=dtype,
    )

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, from_aistudio=model_args.from_aistudio)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
    )

    if isinstance(tokenizer, LlamaTokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    """run with messages which contains multi-turns message"""
    # 0. trans List[Messages] to qwen tuple messages
    assistant_index = 0
    history = []
    text = ""
    if model.base_model_prefix == "qwen":
        from qwen.react import eval_function_call_messages

    raw_messages = load_messages_from_file(data_args.dataset_name_or_path)
    tool_use_count, tool_parameter_count = [], []
    invalid_response_count = []
    for index, raw_message in enumerate(raw_messages):
        logger.info(json.dumps(dict(index=index), ensure_ascii=False))
        function_call_message = eval_function_call_messages(
            model, tokenizer, raw_message["messages"], raw_message["tools"]
        )
        last_function_call_message = [m for m in raw_message["messages"] if m.is_assistant_function_call_message][-1]

        logger.info(json.dumps(function_call_message.model_dump(), ensure_ascii=False))
        tool_use_count.append(
            int(function_call_message.function_call.name == last_function_call_message.function_call.name)
        )
        try:
            tool_parameter_count.append(
                int(
                    json.loads(function_call_message.function_call.parameters)
                    == json.loads(last_function_call_message.function_call.parameters)
                )
            )
            tool_parameter_count.append(1)
            invalid_response_count.append(0)
        except Exception:
            logger.error(json.dumps(dict(stage="invalid response json schema"), ensure_ascii=False))
            invalid_response_count.append(1)
            tool_parameter_count.append(0)
        if index == 100:
            break

    logger.info(f"tool_use_count: {tool_use_count}")
    logger.info(f"tool_parameter_count:{tool_parameter_count}")
    logger.info(f"invalid_response_count: {invalid_response_count}")

    logger.info(f"tool use acc: {sum(tool_use_count) / len(tool_use_count):.4f}")
    logger.info(f"tool parameter acc: {sum(tool_parameter_count) / len(tool_parameter_count):.4f}")
    logger.info(f"invalid parameter generation: {sum(invalid_response_count) / len(invalid_response_count):.6f}")


if __name__ == "__main__":
    main()
