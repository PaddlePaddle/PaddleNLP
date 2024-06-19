# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import random
import string
import time

import paddle
import uvicorn
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from paddlenlp.transformers import CodeGenForCausalLM, CodeGenTokenizer
from paddlenlp.utils.log import logger


class DefaultConfig:
    model_name_or_path = "Salesforce/codegen-350M-mono"
    device = "gpu"
    temperature = 0.5
    top_k = 10
    top_p = 1.0
    repetition_penalty = 1.0
    min_length = 0
    max_length = 16
    decode_strategy = "greedy_search"
    use_faster = True
    use_fp16_decoding = True
    default_dtype = "float16" if use_faster and use_fp16_decoding else "float32"


class Input(BaseModel):
    prompt: str
    stream: bool = False


class Output(BaseModel):
    id: str
    model: str = "codegen"
    object: str = "text_completion"
    created: int = int(time.time())
    choices: list = None
    usage = {
        "completion_tokens": None,
        "prompt_tokens": None,
        "total_tokens": None,
    }


generate_config = DefaultConfig()
paddle.set_device(generate_config.device)
paddle.set_default_dtype(generate_config.default_dtype)

tokenizer = CodeGenTokenizer.from_pretrained(generate_config.model_name_or_path)
model = CodeGenForCausalLM.from_pretrained(generate_config.model_name_or_path)

app = FastAPI()


def random_completion_id():
    return "cmpl-" + "".join(random.choice(string.ascii_letters + string.digits) for _ in range(29))


@app.post("/v1/engines/codegen/completions", status_code=200)
async def gen(item: Input):
    item = item.dict()
    logger.info(f"Request: {item}")
    temperature = item.get("temperature", generate_config.temperature)
    top_k = item.get("top_k", generate_config.top_k)
    if temperature == 0.0:
        temperature = 1.0
        top_k = 1
    repetition_penalty = item.get("frequency_penalty", generate_config.repetition_penalty)

    start_time = time.time()
    logger.info("Start generating code")
    tokenized = tokenizer([item["prompt"]], truncation=True, return_tensors="pd")
    output, _ = model.generate(
        tokenized["input_ids"],
        max_length=16,
        min_length=generate_config.min_length,
        decode_strategy=generate_config.decode_strategy,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        use_fast=generate_config.use_faster,
        use_fp16_decoding=generate_config.use_fp16_decoding,
    )
    logger.info("Finish generating code")
    end_time = time.time()
    logger.info(f"Time cost: {end_time - start_time}")
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Generated code: {output}")
    output_json = Output(
        id=random_completion_id(),
        choices=[
            {
                "text": output,
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        usage={
            "completion_tokens": None,
            "prompt_tokens": None,
            "total_tokens": None,
        },
    ).json()

    def stream_response(response):
        yield f"{response}\n\n"
        yield "data: [DONE]\n\n"

    if item.get("stream", False):
        return EventSourceResponse(stream_response(output_json))
    else:
        return Response(
            status_code=status.HTTP_200_OK,
            content=output_json,
            media_type="application/json",
        )


if __name__ == "__main__":
    uvicorn.run("codegen_server:app", host="0.0.0.0", port=8978)
