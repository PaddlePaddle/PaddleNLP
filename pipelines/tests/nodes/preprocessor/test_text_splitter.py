# Copyright 2023 The LangChain Authors. All rights reserved.
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

from pipelines.nodes.preprocessor.text_splitter import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
)


class TestCharacterTextSplitter(unittest.TestCase):
    def test_character_text_splitter(self) -> None:
        """Test splitting by character count."""
        text = "foo bar baz 123"
        splitter = CharacterTextSplitter(separator=" ", chunk_size=7, chunk_overlap=3)
        output = splitter.split_text(text)
        expected_output = ["foo bar", "bar baz", "baz 123"]
        assert output == expected_output

    def test_character_text_splitter_empty_doc(self) -> None:
        """Test splitting by character count doesn't create empty documents."""
        text = "foo  bar"
        splitter = CharacterTextSplitter(separator=" ", chunk_size=2, chunk_overlap=0)
        output = splitter.split_text(text)
        expected_output = ["foo", "bar"]
        assert output == expected_output

    def test_character_text_splitter_separtor_empty_doc(self) -> None:
        """Test edge cases are separators."""
        text = "f b"
        splitter = CharacterTextSplitter(separator=" ", chunk_size=2, chunk_overlap=0)
        output = splitter.split_text(text)
        expected_output = ["f", "b"]
        assert output == expected_output

    def test_character_text_splitter_long(self) -> None:
        """Test splitting by character count on long words."""
        text = "foo bar baz a a"
        splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=1)
        output = splitter.split_text(text)
        expected_output = ["foo", "bar", "baz", "a a"]
        assert output == expected_output

    def test_character_text_splitter_short_words_first(self) -> None:
        """Test splitting by character count when shorter words are first."""
        text = "a a foo bar baz"
        splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=1)
        output = splitter.split_text(text)
        expected_output = ["a a", "foo", "bar", "baz"]
        assert output == expected_output

    def test_character_text_splitter_longer_words(self) -> None:
        """Test splitting by characters when splits not found easily."""
        text = "foo bar baz 123"
        splitter = CharacterTextSplitter(separator=" ", chunk_size=1, chunk_overlap=1)
        output = splitter.split_text(text)
        expected_output = ["foo", "bar", "baz", "123"]
        assert output == expected_output

    def test_character_text_splitting_args(self) -> None:
        """Test invalid arguments."""
        with self.assertRaises(ValueError):
            CharacterTextSplitter(chunk_size=2, chunk_overlap=4)

    def test_create_documents(self) -> None:
        """Test create documents method."""
        texts = ["foo bar", "baz"]
        splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
        docs = splitter.create_documents(texts)
        expected_docs = [
            {"content": "foo", "meta": {}},
            {"content": "bar", "meta": {}},
            {"content": "baz", "meta": {}},
        ]
        assert docs == expected_docs

    def test_create_documents_with_metadata(
        self,
    ) -> None:
        """Test create documents with metadata method."""
        texts = ["foo bar", "baz"]
        splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
        docs = splitter.create_documents(texts, [{"source": "1"}, {"source": "2"}])
        expected_docs = [
            {"content": "foo", "meta": {"source": "1"}},
            {"content": "bar", "meta": {"source": "1"}},
            {"content": "baz", "meta": {"source": "2"}},
        ]
        assert docs == expected_docs

    def test_metadata_not_shallow(self) -> None:
        """Test that metadatas are not shallow."""
        texts = ["foo bar"]
        splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
        docs = splitter.create_documents(texts, [{"source": "1"}])
        expected_docs = [{"content": "foo", "meta": {"source": "1"}}, {"content": "bar", "meta": {"source": "1"}}]
        assert docs == expected_docs
        docs[0]["meta"]["foo"] = 1
        assert docs[0]["meta"] == {"source": "1", "foo": 1}
        assert docs[1]["meta"] == {"source": "1"}

    def test_iterative_text_splitter(self) -> None:
        """Test iterative text splitter."""
        text = """Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
        This is a weird text to write, but gotta test the splittingggg some how.
        Bye!\n\n-H."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=1)
        output = splitter.split_text(text)
        expected_output = [
            "Hi.",
            "I'm",
            "Harrison.",
            "How? Are?",
            "You?",
            "Okay then",
            "f f f f.",
            "This is",
            "a weird",
            "text to",
            "write, but",
            "gotta test",
            "the",
            "splittingg",
            "ggg",
            "some how.",
            "Bye!",
            "-H.",
        ]
        assert output == expected_output

    def test_spcay_text_splitter(self) -> None:
        text = """Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
        This is a weird text to write, but gotta test the splittingggg some how.
        Bye!\n\n-H."""
        splitter = SpacyTextSplitter(chunk_size=10, chunk_overlap=1, pipeline="en_core_web_sm")
        output = splitter.split_text(text)
        expected_output = [
            "Hi.\n\nI'm Harrison.",
            "How? Are?",
            "You?",
            "Okay then f f f f.",
            "This is a weird text to write, but gotta test the splittingggg some how.",
            "Bye!\n\n-H.",
        ]
        assert expected_output == output

    def test_markdown_text_splitter(self) -> None:
        md = "## Bar\n\nHi this is Jim  \nHi this is Joe\n\n ## Baz\n\n Hi this is Molly"
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            separator="\n",
            chunk_size=10,
            headers_to_split_on=headers_to_split_on,
            return_each_line=False,
            filters=["\n"],
        )
        output = markdown_splitter.split_text(md)
        expected_output = ["Bar\nHi this is Jim\nHi this is Joe", "Baz\nHi this is Molly"]
        assert expected_output == output
