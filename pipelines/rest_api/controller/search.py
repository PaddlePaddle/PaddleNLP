# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from numpy import ndarray
from pydantic import BaseConfig
from rest_api.config import (
    CONCURRENT_REQUEST_PER_WORKER,
    FILE_UPLOAD_PATH,
    LOG_LEVEL,
    PIPELINE_YAML_PATH,
    QUERY_PIPELINE_NAME,
    QUERY_QA_PAIRS_NAME,
)
from rest_api.controller.utils import RequestLimiter
from rest_api.schema import (
    DocumentRequest,
    DocumentResponse,
    QueryImageResponse,
    QueryQAPairRequest,
    QueryQAPairResponse,
    QueryRequest,
    QueryResponse,
    SentaRequest,
    SentaResponse,
)

import pipelines
from pipelines.pipelines.base import Pipeline

logging.getLogger("pipelines").setLevel(LOG_LEVEL)
logger = logging.getLogger("pipelines")

BaseConfig.arbitrary_types_allowed = True

router = APIRouter()

PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_PIPELINE_NAME)

try:
    QA_PAIR_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_QA_PAIRS_NAME)
except Exception:
    logger.warning(f"Request pipeline ('{QUERY_QA_PAIRS_NAME}: is null'). ")

DOCUMENT_STORE = PIPELINE.get_document_store()
logging.info(f"Loaded pipeline nodes: {PIPELINE.graph.nodes.keys()}")

concurrency_limiter = RequestLimiter(CONCURRENT_REQUEST_PER_WORKER)
logging.info("Concurrent requests per worker: {CONCURRENT_REQUEST_PER_WORKER}")


@router.get("/initialized")
def check_status():
    """
    This endpoint can be used during startup to understand if the
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True


@router.get("/hs_version")
def pipelines_version():
    """
    Get the running pipelines version.
    """
    return {"hs_version": pipelines.__version__}


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the pipelines pipeline.
    """
    with concurrency_limiter.run():
        result = _process_request(PIPELINE, request)
        return result


@router.post("/query_images", response_model=QueryResponse, response_model_exclude_none=True)
def query_images_for_retrieval(
    files: List[UploadFile] = File(...),
    # JSON serialized string
    meta: Optional[str] = Form("null"),
):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the pipelines pipeline.
    """
    file_paths: list = []
    file_metas: list = []
    meta_form = json.loads(meta) or {}  # type: ignore

    for file in files:
        try:
            file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_paths.append(file_path)
            # meta_form["name"] = file.filename
            file_metas.append(meta_form)
        finally:
            file.file.close()
    result = PIPELINE.run(query=str(file_paths[0]), params=meta_form, debug=True)
    return result


@router.post("/query_text_to_images", response_model=QueryImageResponse, response_model_exclude_none=True)
def query_images(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the pipelines pipeline.
    """
    result = {}
    result["query"] = request.query
    params = request.params or {}
    res = PIPELINE.run(query=request.query, params=params, debug=request.debug)
    # Ensure answers and documents exist, even if they're empty lists
    result["answers"] = res["results"]
    if "documents" not in result:
        result["documents"] = []
    if "answers" not in result:
        result["answers"] = []
    return result


@router.post("/query_documents", response_model=DocumentResponse, response_model_exclude_none=True)
def query_documents(request: DocumentRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the pipelines pipeline.
    """
    result = {}
    result["meta"] = request.meta
    params = request.params or {}
    res = PIPELINE.run(meta=request.meta, params=params, debug=request.debug)
    result["results"] = res["results"]
    return result


@router.post("/senta_file", response_model=SentaResponse, response_model_exclude_none=True)
def senta_file(request: SentaRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the pipelines pipeline.
    """
    result = {}
    result["meta"] = request.meta
    params = request.params or {}
    res = PIPELINE.run(meta=request.meta, params=params, debug=request.debug)
    result["img_dict"] = res["img_dict"]
    return result


@router.post("/query_qa_pairs", response_model=QueryQAPairResponse, response_model_exclude_none=True)
def query_qa_pairs(request: QueryQAPairRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the pipelines pipeline.
    """
    print("request", request)
    result = {}
    result["meta"] = request.meta
    params = request.params or {}
    res = QA_PAIR_PIPELINE.run(meta=request.meta, params=params, debug=request.debug)
    result["filtered_cqa_triples"] = res["filtered_cqa_triples"]
    return result


def _process_request(pipeline, request) -> Dict[str, Any]:
    start_time = time.time()
    params = request.params or {}

    # format global, top-level filters (e.g. "params": {"filters": {"name": ["some"]}})
    if "filters" in params.keys():
        params["filters"] = _format_filters(params["filters"])

    # format targeted node filters (e.g. "params": {"Retriever": {"filters": {"value"}}})
    for key in params.keys():
        if "filters" in params[key].keys():
            params[key]["filters"] = _format_filters(params[key]["filters"])

    result = pipeline.run(query=request.query, params=params, debug=request.debug)

    # Ensure answers and documents exist, even if they're empty lists
    if "documents" not in result:
        result["documents"] = []
    if "answers" not in result:
        result["answers"] = []
    # if any of the documents contains an embedding as an ndarray the latter needs to be converted to list of float
    for document in result["documents"]:
        if isinstance(document.embedding, ndarray):
            document.embedding = document.embedding.tolist()

    logger.info(
        json.dumps({"request": request, "response": result, "time": f"{(time.time() - start_time):.2f}"}, default=str)
    )
    return result


def _format_filters(filters):
    """
    Adjust filters to compliant format:
    Put filter values into a list and remove filters with null value.
    """
    new_filters = {}
    if filters is None:
        logger.warning(
            "Request with deprecated filter format ('\"filters\": null'). "
            "Remove empty filters from params to be compliant with future versions"
        )
    else:
        for key, values in filters.items():
            if values is None:
                logger.warning(
                    f"Request with deprecated filter format ('{key}: null'). "
                    f"Remove null values from filters to be compliant with future versions"
                )
                continue

            if not isinstance(values, list):
                logger.warning(
                    f"Request with deprecated filter format ('{key}': {values}). "
                    f"Change to '{key}':[{values}]' to be compliant with future versions"
                )
                values = [values]

            new_filters[key] = values
    return new_filters
