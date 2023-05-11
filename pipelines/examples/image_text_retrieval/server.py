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
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from examples.image_text_retrieval.data_utils import (
    FILE_UPLOAD_PATH,
    PIPELINE_YAML_PATH,
    QUERY_PIPELINE_NAME,
    QueryDocument,
    document_embedding_model,
    host,
    index_name,
    port,
    query_embedding_model,
)
from fastapi import FastAPI, File, Form, UploadFile

from pipelines.document_stores import MilvusDocumentStore
from pipelines.nodes import MultiModalRetriever
from pipelines.pipelines.base import Pipeline
from pipelines.schema import Document

app = FastAPI()
_task_path = "checkpoints/checkpoint-370"
document_store = MilvusDocumentStore(
    host=host,
    index=index_name,
    port=port,
    index_param={"M": 16, "efConstruction": 50},
    index_type="HNSW",
)
if os.path.exists(_task_path):
    retriever_mm = MultiModalRetriever(
        document_store=document_store,
        query_embedding_model=query_embedding_model,
        query_type="image",
        document_embedding_models={"text": document_embedding_model},
        task_path=_task_path,
    )
else:
    retriever_mm = MultiModalRetriever(
        document_store=document_store,
        query_embedding_model=query_embedding_model,
        query_type="image",
        document_embedding_models={"text": document_embedding_model},
    )


PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_PIPELINE_NAME)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query_images/")
def query_images_for_retrieval(
    files: List[UploadFile] = File(...),
    # JSON serialized string
    meta: Optional[str] = Form("null"),
):
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


@app.post("/get_document")
def get_document(id: str):
    try:
        document = document_store.get_document_by_id(id=id)
    except Exception as e:
        return {"message": "error", "error": str(e)}
    return document


@app.post("/insert_text/")
def insert_data(documents: List[QueryDocument]):
    my_documents = []
    for doc in documents:
        doc_dict = {
            "content": doc.content,
            "meta": {"name": doc.name},
        }
        doc_obj = Document.from_dict(doc_dict)
        my_documents.append(doc_obj)
    try:
        document_store.write_documents(my_documents, retriever_mm)
    except Exception as e:
        return {"message": "error", "error": str(e)}

    return {"message": "ok"}


@app.post("/delete_text/")
def delete_data(ids: list):
    try:
        document_store.delete_documents(ids=ids)
    except Exception as e:
        return {"message": "error", "error": str(e)}
    return {"message": "ok"}
