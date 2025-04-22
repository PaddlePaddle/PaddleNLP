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
import time
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from server.http_server.api import Req, chat_completion_generator


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[int] = None
    finish_reason: Optional[Literal["stop", "length"]]


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[float] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]


def format_openai_message_completions(req: Req, result: Dict) -> CompletionResponse:
    choice_data = CompletionResponseChoice(
        index=0,
        text=result["token"],
        logprobs=result.get("logprobs", None),
        finish_reason=result.get("finish_reason", None),
    )
    response = CompletionResponse(
        id=req.req_id,
        choices=[choice_data],
        model=req.model,
        created=int(time.time()),
        usage=UsageInfo(
            completion_tokens=result["usage"]["completion_tokens"],
            prompt_tokens=result["usage"]["prompt_tokens"],
            total_tokens=result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"],
        ),
    )
    return response


def format_openai_message_chat_completions(req: Req, result: Dict) -> ChatCompletionResponse:
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            content=result["token"],
            role="assistant",
        ),
        finish_reason=result.get("finish_reason", None),
    )
    response = ChatCompletionResponse(
        id=req.req_id,
        choices=[choice_data],
        model=req.model,
        created=int(time.time()),
        usage=UsageInfo(
            completion_tokens=result["usage"]["completion_tokens"],
            prompt_tokens=result["usage"]["prompt_tokens"],
            total_tokens=result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"],
        ),
    )
    return response


def format_openai_message_stream_chat_completions(req: Req, result: Dict) -> ChatCompletionStreamResponse:
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=result["token"]),
        finish_reason=result.get("finish_reason", None),
    )
    response = ChatCompletionStreamResponse(id=req.req_id, choices=[choice_data], model=req.model)
    return response


def format_openai_message_stream_completions(req: Req, result: Dict) -> CompletionResponse:
    choice_data = CompletionResponseStreamChoice(
        index=0,
        text=result["token"],
        logprobs=result.get("logprobs", None),
        finish_reason=result.get("finish_reason", None),
    )
    response = CompletionStreamResponse(
        id=req.req_id,
        choices=[choice_data],
        model=req.model,
    )
    return response


def openai_chat_commpletion_generator(infer_grpc_url: str, req: Req, chat_interface: bool) -> Dict:
    def _openai_format_resp(resp_dict):
        return f"data: {resp_dict}\n\n"

    for resp in chat_completion_generator(infer_grpc_url, req, yield_json=False):
        if resp.get("error_msg") or resp.get("error_code"):
            return ErrorResponse(message=resp.get("error_msg"), code=resp.get("error_code"))

        if resp.get("is_end") == 1:
            yield _openai_format_resp("[DONE]")

        if chat_interface:
            json_response = json.dumps(
                format_openai_message_stream_chat_completions(req, resp).dict(exclude_unset=True), ensure_ascii=False
            )
        else:
            json_response = json.dumps(
                format_openai_message_stream_completions(req, resp).dict(exclude_unset=True), ensure_ascii=False
            )
        yield _openai_format_resp(json_response)


def openai_chat_completion_result(infer_grpc_url: str, req: Req, chat_interface: bool):
    result = ""
    error_resp = None
    for resp in chat_completion_generator(infer_grpc_url, req, yield_json=False):
        if resp.get("error_msg") or resp.get("error_code"):
            error_resp = resp
            error_resp["result"] = ""
        else:
            result += resp.get("token")
        usage = resp.get("usage", None)

    if error_resp:
        return ErrorResponse(message=error_resp.get("error_msg"), code=error_resp.get("error_code"))

    response = {"token": result, "error_msg": "", "error_code": 0, "usage": usage}
    if chat_interface:
        return format_openai_message_chat_completions(req, response)
    else:
        return format_openai_message_completions(req, response)
