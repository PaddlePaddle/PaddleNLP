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

from typing import Optional, List, Union

import json
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipelines.pipelines.base import Pipeline
from pipelines.pipelines.config import get_component_definitions, get_pipeline_definition, read_pipeline_config_from_yaml
from rest_api.config import PIPELINE_YAML_PATH, FILE_UPLOAD_PATH, INDEXING_PIPELINE_NAME, INDEXING_QA_GENERATING_PIPELINE_NAME, FILE_PARSE_PATH
from rest_api.controller.utils import as_form

logger = logging.getLogger(__name__)
router = APIRouter()

try:
    pipeline_config = read_pipeline_config_from_yaml(Path(PIPELINE_YAML_PATH))
    pipeline_definition = get_pipeline_definition(
        pipeline_config=pipeline_config, pipeline_name=INDEXING_PIPELINE_NAME)
    component_definitions = get_component_definitions(
        pipeline_config=pipeline_config, overwrite_with_env_variables=True)
    # Since each instance of FAISSDocumentStore creates an in-memory FAISS index, the Indexing & Query Pipelines would
    # end up with different indices. The same applies for InMemoryDocumentStore. The check below prevents creation of
    # Indexing Pipelines with FAISSDocumentStore or InMemoryDocumentStore.
    is_faiss_or_inmemory_present = False
    for node in pipeline_definition["nodes"]:
        if (component_definitions[node["name"]]["type"] == "FAISSDocumentStore"
                or component_definitions[node["name"]]["type"]
                == "InMemoryDocumentStore"):
            is_faiss_or_inmemory_present = True
            break
    if is_faiss_or_inmemory_present:
        logger.warning(
            "Indexing Pipeline with FAISSDocumentStore or InMemoryDocumentStore is not supported with the REST APIs."
        )
        INDEXING_PIPELINE = None
        INDEXING_QA_GENERATING_PIPELINE = None
    else:
        INDEXING_QA_GENERATING_PIPELINE = Pipeline.load_from_yaml(
            Path(PIPELINE_YAML_PATH),
            pipeline_name=INDEXING_QA_GENERATING_PIPELINE_NAME)
        INDEXING_PIPELINE = Pipeline.load_from_yaml(
            Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)

except KeyError:
    INDEXING_PIPELINE = None
    INDEXING_QA_GENERATING_PIPELINE = None
    logger.warning(
        "Indexing Pipeline not found in the YAML configuration. File Upload API will not be available."
    )

# create directory for uploading files
os.makedirs(FILE_UPLOAD_PATH, exist_ok=True)


@as_form
class FileConverterParams(BaseModel):
    remove_numeric_tables: Optional[bool] = None
    valid_languages: Optional[List[str]] = None


@as_form
class PreprocessorParams(BaseModel):
    clean_whitespace: Optional[bool] = None
    clean_empty_lines: Optional[bool] = None
    clean_header_footer: Optional[bool] = None
    split_by: Optional[str] = None
    split_length: Optional[int] = None
    split_overlap: Optional[int] = None
    split_respect_sentence_boundary: Optional[bool] = None


class Response(BaseModel):
    file_id: str


@router.post("/file-upload-qa-generate")
def upload_file(
        files: List[UploadFile] = File(...),
        # JSON serialized string
        meta: Optional[str] = Form("null"),  # type: ignore
        fileconverter_params: FileConverterParams = Depends(
            FileConverterParams.as_form),  # type: ignore
):
    """
    You can use this endpoint to upload a file for indexing
    """
    if not INDEXING_QA_GENERATING_PIPELINE:
        raise HTTPException(
            status_code=501,
            detail="INDEXING_QA_GENERATING_PIPELINE  is not configured.")

    file_paths: list = []
    file_metas: list = []
    meta_form = json.loads(meta) or {}  # type: ignore
    if not isinstance(meta_form, dict):
        raise HTTPException(
            status_code=500,
            detail=
            f"The meta field must be a dict or None, not {type(meta_form)}")

    for file in files:
        try:
            file_path = Path(
                FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_paths.append(file_path)
            meta_form["name"] = file.filename
            file_metas.append(meta_form)
        finally:
            file.file.close()

    result = INDEXING_QA_GENERATING_PIPELINE.run(
        file_paths=file_paths,
        meta=file_metas,
        params={
            "TextFileConverter": fileconverter_params.dict(),
            "PDFFileConverter": fileconverter_params.dict(),
        },
    )
    return {'message': "OK"}


@router.post("/file-upload")
def upload_file(
        files: List[UploadFile] = File(...),
        # JSON serialized string
        meta: Optional[str] = Form("null"),  # type: ignore
        fileconverter_params: FileConverterParams = Depends(
            FileConverterParams.as_form),  # type: ignore
        preprocessor_params: PreprocessorParams = Depends(
            PreprocessorParams.as_form),  # type: ignore
):
    """
    You can use this endpoint to upload a file for indexing
    """
    if not INDEXING_PIPELINE:
        raise HTTPException(status_code=501,
                            detail="Indexing Pipeline is not configured.")

    file_paths: list = []
    file_metas: list = []
    meta_form = json.loads(meta) or {}  # type: ignore
    if not isinstance(meta_form, dict):
        raise HTTPException(
            status_code=500,
            detail=
            f"The meta field must be a dict or None, not {type(meta_form)}")

    for file in files:
        try:
            file_path = Path(
                FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_paths.append(file_path)
            meta_form["name"] = file.filename
            file_metas.append(meta_form)
        finally:
            file.file.close()

    result = INDEXING_PIPELINE.run(
        file_paths=file_paths,
        meta=file_metas,
        params={
            "TextFileConverter": fileconverter_params.dict(),
            "PDFFileConverter": fileconverter_params.dict(),
            "Preprocessor": preprocessor_params.dict(),
        },
    )
    return {'message': "OK"}


@router.get("/files")
def download_file(
        file_name: str = '1fc0aeac9900487a8c6cec8dda6499bd_demo_1.png'):
    file_path = os.path.join(FILE_PARSE_PATH, file_name)
    if (os.path.exists(file_path)):
        return FileResponse(file_path)
    return {'message': "File not Found"}
