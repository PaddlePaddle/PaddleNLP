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

from dataclasses import dataclass, field


@dataclass
class StructuredIndexerArguments:
    """
    Arguments for StructuredIndexer.
    """

    log_dir: str = field(default=".logs", metadata={"help": "log directory"})


@dataclass
class StructuredIndexerEncodeArguments(StructuredIndexerArguments):
    """
    Arguments for encoding corpus in StructuredIndexer.
    """

    encode_model_name_or_path: str = field(
        default="BAAI/bge-large-en-v1.5", metadata={"help": "encode model name or path"}
    )


@dataclass
class StructuredIndexerPipelineArguments(StructuredIndexerEncodeArguments):
    """
    Arguments for building StructuredIndex pipeline for a single corpus file.
    """

    source: str = field(default="data/source", metadata={"help": "source file or directory"})
    parse_model_name_or_path: str = field(
        default="Qwen/Qwen2-7B-Instruct", metadata={"help": "parse model name or path"}
    )
    parse_model_url: str = field(default=None, metadata={"help": "parse model url if you use api"})
    summarize_model_name_or_path: str = field(
        default="Qwen/Qwen2-7B-Instruct",
        metadata={"help": "summarize model name or path"},
    )
    summarize_model_url: str = field(default=None, metadata={"help": "summarize model url if you use api"})


@dataclass
class RetrievalArguments(StructuredIndexerEncodeArguments):
    """
    Arguments for StructuredIndex to retrieve.
    """

    search_result_dir: str = field(default="search_result", metadata={"help": "search result directory"})
    query_filepath: str = field(default="query.json", metadata={"help": "query file path"})
    query_text: str = field(default=None, metadata={"help": "query text"})
    top_k: int = field(default=5, metadata={"help": "top k results for each query"})
    embedding_batch_size: int = field(default=128, metadata={"help": "embedding batch size for queries"})
