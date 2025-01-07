import time
import json
import queue

import numpy as np
from typing import Dict
from datetime import datetime
from functools import partial

import tritonclient.grpc as grpcclient
from tritonclient import utils as triton_utils
from openai.types.completion_usage import CompletionUsage
from openai.types.completion_choice import CompletionChoice
from openai.types.completion import Completion
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChatCompletionChunk,
    Choice as ChatCompletionChoice
)

from server.http_server.api import Req, chat_completion_generator
from server.utils import http_server_logger


def format_openai_message_completions(req: Req, result: Dict) -> Completion:
    choice_data = CompletionChoice(
                index=0,
                text=result['token'],
                finish_reason=result.get("finish_reason", "stop"),
            )
    chunk = Completion(
                id=req.req_id,
                choices=[choice_data],
                model=req.model,
                created=int(time.time()),
                object="text_completion",
                usage=CompletionUsage(
                    completion_tokens=result["usage"]["completion_tokens"],
                    prompt_tokens=result["usage"]["prompt_tokens"],
                    total_tokens=result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"],
                ),
            )
    return chunk.model_dump_json(exclude_unset=True)


def format_openai_message_chat_completions(req: Req, result: Dict) -> ChatCompletionChunk:
    choice_data = ChatCompletionChoice(
                index=0,
                delta=ChoiceDelta(
                    content=result['token'],
                    role="assistant",
                ),
                finish_reason=result.get("finish_reason", "stop"),
            )
    chunk = ChatCompletionChunk(
                id=req.req_id,
                choices=[choice_data],
                model=req.model,
                created=int(time.time()),
                object="chat.completion.chunk",
                usage=CompletionUsage(
                    completion_tokens=result["usage"]["completion_tokens"],
                    prompt_tokens=result["usage"]["prompt_tokens"],
                    total_tokens=result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"],
                ),
            )
    return chunk.model_dump_json(exclude_unset=True)


def openai_chat_commpletion_generator(infer_grpc_url: str, req: Req, chat_interface: bool) -> Dict:

    def _openai_format_resp(resp_dict):
        return f"data: {resp_dict}\n\n"

    for resp in chat_completion_generator(infer_grpc_url, req, yield_json=False):
        if resp.get("is_end") == 1:
            yield _openai_format_resp("[DONE]")

        if chat_interface:
            yield _openai_format_resp(format_openai_message_chat_completions(req, resp))
        else:
            yield _openai_format_resp(format_openai_message_completions(req, resp))


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
        return error_resp
    response = {'token': result, 'error_msg': '', 'error_code': 0, 'usage': usage}

    if chat_interface:
        return format_openai_message_chat_completions(req, response)
    else:
        return format_openai_message_completions(req, response)
