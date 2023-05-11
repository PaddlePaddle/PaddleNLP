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

import os

from pydantic import BaseModel

host = "localhost"
embedding_dim = 768
port = 8530
index_name = "wukong_text"
query_embedding_model = "PaddlePaddle/ernie_vil-2.0-base-zh"
document_embedding_model = "PaddlePaddle/ernie_vil-2.0-base-zh"

os.makedirs("file-upload", exist_ok=True)

PIPELINE_YAML_PATH = "examples/image_text_retrieval/image_to_text_retrieval.yaml"
QUERY_PIPELINE_NAME = "query"

FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", "file-upload")


class Item(BaseModel):
    query: str


class QueryDocument(BaseModel):
    name: str
    content: str
