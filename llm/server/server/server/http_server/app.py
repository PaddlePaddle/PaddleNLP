# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import os
from typing import Dict, Union

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from server.http_server.adapter_openai import (
    ErrorResponse,
    openai_chat_commpletion_generator,
    openai_chat_completion_result,
)
from server.http_server.api import (
    Req,
    chat_completion_generator,
    chat_completion_result,
)
from server.utils import http_server_logger

http_server_logger.info("create fastapi app...")
app = FastAPI()

def _is_Req(request: Dict):
    if "input_ids" in request or "text" in request:
        return True
    return False

@app.post("/v1")
def create_openai_adapter():
    pass

@app.post("/v1/completions")
def openai_v1_completions(request: Union[Dict, Req]):
    if isinstance(request, Req) or _is_Req(request):
        return create_completion(request)
    elif isinstance(request, dict):
        return create_openai_completion(request, chat_interface=False)

@app.post("/v1/chat/completions")
def openai_v1_chat_completions(request: Union[Dict, Req]):
    if isinstance(request, Req) or _is_Req(request):
        return create_completion(request)
    elif isinstance(request, dict):
        return create_openai_completion(request, chat_interface=True)

def create_completion(req: Req):
    """
    HTTP Server for chat completion
    Return:
        In Stream:
            Normal, return {'token': xxx, 'is_end': xxx, 'send_idx': xxx, ..., 'error_msg': '', 'error_code': 0}
            Others, return {'error_msg': xxx, 'error_code': xxx}, error_msg not None, error_code != 0
        Not In Stream:
            Normal, return {'tokens_all': xxx, ..., 'error_msg': '', 'error_code': 0}
            Others, return {'error_msg': xxx, 'error_code': xxx}, error_msg not None, error_code != 0
    """
    try:
        http_server_logger.info(f"receive request: {req.req_id}")
        grpc_port = int(os.getenv("SERVICE_GRPC_PORT", 0))
        if grpc_port == 0:
            return {"error_msg": f"SERVICE_GRPC_PORT ({grpc_port}) for infer service is invalid", "error_code": 400}
        grpc_url = f"localhost:{grpc_port}"

        if req.stream:
            generator = chat_completion_generator(infer_grpc_url=grpc_url, req=req, yield_json=True)
            resp = StreamingResponse(generator, media_type="text/event-stream")
        else:
            resp = chat_completion_result(infer_grpc_url=grpc_url, req=req)
    except Exception as e:
        resp = {"error_msg": str(e), "error_code": 501}
    finally:
        http_server_logger.info(f"finish request: {req.req_id}")
        return resp





def create_openai_completion(request: Dict, chat_interface: bool):
    try:
        req = Req()
        req.load_openai_request(request)
    except Exception as e:
        error_resp = ErrorResponse(message=f"request body is not a valid json format, {str(e)}", code=400)
        return JSONResponse(error_resp.dict(), status_code=400)

    try:
        http_server_logger.info(f"receive request: {req.req_id}")

        grpc_port = int(os.getenv("SERVICE_GRPC_PORT", 0))
        if grpc_port == 0:
            error_resp = ErrorResponse(
                message=f"SERVICE_GRPC_PORT ({grpc_port}) for infer service is invalid", code=400
            )
            return JSONResponse(error_resp.dict(), status_code=400)
        grpc_url = f"localhost:{grpc_port}"

        if req.stream:
            generator = openai_chat_commpletion_generator(
                infer_grpc_url=grpc_url,
                req=req,
                chat_interface=chat_interface,
            )
            if isinstance(generator, ErrorResponse):
                resp = JSONResponse(generator.dict(), status_code=400)
            else:
                resp = StreamingResponse(generator, media_type="text/event-stream")
        else:
            response_obj = openai_chat_completion_result(
                infer_grpc_url=grpc_url, req=req, chat_interface=chat_interface
            )
            if isinstance(response_obj, ErrorResponse):
                resp = JSONResponse(response_obj.dict(), status_code=400)
            else:
                resp = response_obj
    except Exception as e:
        resp = JSONResponse(ErrorResponse(message=str(e), code=501).dict(), status_code=400)
    finally:
        http_server_logger.info(f"finish request: {req.req_id}")
        return resp


def launch_http_server(port: int, workers: int) -> None:
    """
    launch http server
    """
    http_server_logger.info(f"launch http server with port: {port}, workers: {workers}")
    try:
        uvicorn.run(app="server.http_server.app:app", host="0.0.0.0", port=port, workers=workers, log_level="error")
    except Exception as e:
        http_server_logger.error(f"launch http server error, {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=9904, type=int, help="port to the http server")
    parser.add_argument("--workers", default=1, type=int, help="set the number of workers for the http service")
    args = parser.parse_args()
    launch_http_server(port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
