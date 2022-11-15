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

from pipelines.utils.import_utils import safe_import

from pipelines.nodes.base import BaseComponent
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
from pipelines.nodes.preprocessor import BasePreProcessor, PreProcessor
from pipelines.nodes.ranker import BaseRanker, ErnieRanker
from pipelines.nodes.reader import BaseReader, ErnieReader
from pipelines.nodes.retriever import BaseRetriever, DensePassageRetriever
from pipelines.nodes.document import DocOCRProcessor, DocPrompter
from pipelines.nodes.text_to_image_generator import ErnieTextToImageGenerator
from pipelines.nodes.answer_extractor import AnswerExtractor, QAFilter, AnswerExtractorPreprocessor, QAFilterPostprocessor
from pipelines.nodes.question_generator import QuestionGenerator
