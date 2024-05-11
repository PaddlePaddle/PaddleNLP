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
import asyncio
import json
import os
import sys
import time
from functools import partial
from typing import List, Optional

import paddle
from argument import (
    DataArgument,
    GenerateArgument,
    ModelArgument,
    QuantArgument,
    TrainingArguments,
)
from data import get_convert_example
from fastapi import FastAPI, HTTPException, Request
from function_call.schema import load_messages_from_file
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
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
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint
from paddlenlp.trainer.trainer_callback import TrainerState
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)
from paddlenlp.utils.log import logger


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


def init_model():
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
    return model, tokenizer


app = FastAPI(title="OpenAI-compatible API")
model, tokenizer = init_model()


# data models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    tools: List[dict] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


async def _resp_async_generator(text_resp: str):
    # let's pretend every word is a token and return it over time
    tokens = text_resp.split(" ")

    for i, token in enumerate(tokens):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": request.model,
            "choices": [{"delta": {"content": token + " "}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(1)
    yield "data: [DONE]\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.messages:
        resp_content = "As a mock AI Assitant, I can only echo your last message: " + request.messages[-1].content
    else:
        resp_content = "As a mock AI Assitant, I can only echo your last message, but there wasn't one!"
    if request.stream:
        return StreamingResponse(_resp_async_generator(resp_content), media_type="application/x-ndjson")

    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=resp_content)}],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8882)
