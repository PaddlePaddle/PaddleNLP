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

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, Extra
from pipelines.schema import Answer, Document, Label, Span
from pydantic import BaseConfig
from pydantic.dataclasses import dataclass as pydantic_dataclass

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

BaseConfig.arbitrary_types_allowed = True


class QueryRequest(BaseModel):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False

    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class FilterRequest(BaseModel):
    filters: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None


@pydantic_dataclass
class AnswerSerialized(Answer):
    context: Optional[str] = None


@pydantic_dataclass
class DocumentSerialized(Document):
    content: str
    embedding: Optional[List[float]]  # type: ignore


@pydantic_dataclass
class LabelSerialized(Label, BaseModel):
    document: DocumentSerialized
    answer: Optional[AnswerSerialized] = None


class CreateLabelSerialized(BaseModel):
    id: Optional[str] = None
    query: str
    document: DocumentSerialized
    is_correct_answer: bool
    is_correct_document: bool
    origin: Literal["user-feedback", "gold-label"]
    answer: Optional[AnswerSerialized] = None
    no_answer: Optional[bool] = None
    pipeline_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    meta: Optional[dict] = None
    filters: Optional[dict] = None

    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class QueryResponse(BaseModel):
    query: str
    answers: List[AnswerSerialized] = []
    documents: List[DocumentSerialized] = []
    debug: Optional[Dict] = Field(None, alias="_debug")
