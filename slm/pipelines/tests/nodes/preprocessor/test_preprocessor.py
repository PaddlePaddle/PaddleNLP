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

import unittest

from pipelines.nodes.preprocessor.preprocessor import PreProcessor

TEXT = """
This is a sample sentence in paragraph_1. This is a sample sentence in paragraph_1. This is a sample sentence in
paragraph_1. This is a sample sentence in paragraph_1. This is a sample sentence in paragraph_1.\f

This is a sample sentence in paragraph_2. This is a sample sentence in paragraph_2. This is a sample sentence in
paragraph_2. This is a sample sentence in paragraph_2. This is a sample sentence in paragraph_2.

This is a sample sentence in paragraph_3. This is a sample sentence in paragraph_3. This is a sample sentence in
paragraph_3. This is a sample sentence in paragraph_3. This is to trick the test with using an abbreviation\f like Dr.
in the sentence.
"""


class TestPreProcessor(unittest.TestCase):
    def test_preprocess_sentence_split(self):
        parameters = [(1, 15), (10, 2)]

        for split_length, expected_documents_count in parameters:
            document = {"content": TEXT}
            preprocessor = PreProcessor(
                split_length=split_length, split_overlap=0, split_by="sentence", split_respect_sentence_boundary=False
            )
            documents = preprocessor.process(document)
            assert len(documents) == expected_documents_count

    def test_preprocess_word_split(self):
        document = {"content": TEXT}
        preprocessor = PreProcessor(
            split_length=10, split_overlap=0, split_by="word", split_respect_sentence_boundary=False
        )
        documents = preprocessor.process(document)
        assert len(documents) == 11

        preprocessor = PreProcessor(
            split_length=15, split_overlap=0, split_by="word", split_respect_sentence_boundary=True
        )
        documents = preprocessor.process(document)
        for i, doc in enumerate(documents):
            if i == 0:
                assert len(doc["content"].split()) == 14
            assert len(doc["content"].split()) <= 15 or doc["content"].startswith("This is to trick")
        assert len(documents) == 8

        preprocessor = PreProcessor(
            split_length=40, split_overlap=10, split_by="word", split_respect_sentence_boundary=True
        )
        documents = preprocessor.process(document)
        assert len(documents) == 5

        preprocessor = PreProcessor(
            split_length=5, split_overlap=0, split_by="word", split_respect_sentence_boundary=True
        )
        documents = preprocessor.process(document)
        assert len(documents) == 15

    def test_preprocess_passage_split(self):
        parameters = [(1, 3), (2, 2)]

        for split_length, expected_documents_count in parameters:
            document = {"content": TEXT}
            preprocessor = PreProcessor(
                split_length=split_length, split_overlap=0, split_by="passage", split_respect_sentence_boundary=False
            )
            documents = preprocessor.process(document)
            assert len(documents) == expected_documents_count
