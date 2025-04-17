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
# flake8: noqa
from pipelines.utils.import_utils import safe_import  # isort: skip
from pipelines.nodes.answer_extractor import (
    AnswerExtractor,
    AnswerExtractorPreprocessor,
    QAFilter,
    QAFilterPostprocessor,
)
from pipelines.nodes.base import BaseComponent
from pipelines.nodes.document import DocOCRProcessor, DocPrompter
from pipelines.nodes.file_classifier import FileTypeClassifier
from pipelines.nodes.file_converter import (
    BaseConverter,
    DocxToTextConverter,
    ImageToTextConverter,
    MarkdownConverter,
    PDFToTextConverter,
    PDFToTextOCRConverter,
    TextConverter,
)
from pipelines.nodes.llm import ChatGLMBot
from pipelines.nodes.llm.ernie_bot import ErnieBot
from pipelines.nodes.llm.history import TruncatedConversationHistory
from pipelines.nodes.llm.prompt_template import LLMPromptTemplate as PromptTemplate
from pipelines.nodes.other import JoinDocuments
from pipelines.nodes.preprocessor import (
    BasePreProcessor,
    CharacterTextSplitter,
    PreProcessor,
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
)
from pipelines.nodes.prompt import PromptModel, PromptNode, Shaper
from pipelines.nodes.question_generator import QuestionGenerator
from pipelines.nodes.ranker import BaseRanker, ErnieRanker
from pipelines.nodes.reader import BaseReader, ErnieReader
from pipelines.nodes.retriever import (
    BaseRetriever,
    BM25Retriever,
    DensePassageRetriever,
    EmbeddingRetriever,
    MultiModalRetriever,
    WebRetriever,
)
from pipelines.nodes.sentiment_analysis import (
    SentaProcessor,
    SentaVisualization,
    UIESenta,
)
from pipelines.nodes.text_to_image_generator import ErnieTextToImageGenerator
