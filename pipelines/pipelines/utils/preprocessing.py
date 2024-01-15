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

import functools
import logging
import multiprocessing
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pipelines.nodes.base import BaseComponent
from pipelines.nodes.file_converter import (
    BaseConverter,
    DocxToTextConverter,
    ImageToTextConverter,
    MarkdownConverter,
    PDFToTextConverter,
    TextConverter,
)
from pipelines.nodes.file_converter.docx import DocxTotxtConverter
from pipelines.nodes.file_converter.markdown import MarkdownRawTextConverter
from pipelines.nodes.preprocessor.text_splitter import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    SpacyTextSplitter,
)

logger = logging.getLogger(__name__)
import copy


def document_rough_split(document_list, max_token=4500):
    document_index_rough = []
    for item in document_list:
        if len(item["content"]) < max_token or "\n" in item:
            document_index_rough.append(item)
        else:
            all_token = len(item["content"])
            token_index = [i for i in range(0, all_token + 1, max_token)]
            if all_token > token_index[-1]:
                token_index.append(all_token)
            token_index_combine = [item["content"][start:end] for start, end in zip(token_index, token_index[1:])]
            for txt in token_index_combine:
                txt_split = copy.deepcopy(item)
                txt_split["content"] = txt
                document_index_rough.append(txt_split)
    return document_index_rough


def split_document(document_index, all_document, splitter, split_paragraphs: bool, clean_func, path, split_answers):
    start = document_index[0]
    end = document_index[1]
    documents = []
    for item in all_document[start:end]:
        text = item["content"]
        if clean_func:
            text = clean_func(text)
        if split_paragraphs is True:
            text_splits = splitter.split_text(text)
            for txt in text_splits:
                if not txt.strip():  # skip empty paragraphs
                    continue
                if split_answers:
                    query, answer = txt.split("\t")
                    meta_data = {"name": path.name, "answer": answer}
                    # Add image list parsed from docx into meta
                    if item["meta"] is not None and "images" in item["meta"]:
                        meta_data["images"] = item["meta"]["images"]
                    documents.append({"content": query, "meta": meta_data})
                else:
                    meta_data = {
                        "name": path.name,
                    }
                    # Add image list parsed from docx into meta
                    if item["meta"] is not None and "images" in item["meta"]:
                        meta_data["images"] = item["meta"]["images"]
                    documents.append({"content": txt, "meta": meta_data})
        else:
            documents.append({"content": text, "meta": item["meta"] if "meta" in item else {"name": path.name}})
    return documents


def run_process(
    document_combination_index,
    list_documents,
    splitter,
    process_num,
    split_paragraphs,
    clean_func,
    path,
    split_answers,
):
    process_num = min(os.cpu_count(), process_num)
    pool = multiprocessing.Pool(process_num)
    split_document_c = functools.partial(
        split_document,
        all_document=list_documents,
        splitter=splitter,
        split_paragraphs=split_paragraphs,
        clean_func=clean_func,
        path=path,
        split_answers=split_answers,
    )
    result = pool.map_async(split_document_c, document_combination_index)
    pool.close()
    pool.join()
    return result.get()


def convert_files_to_dicts(
    dir_path: str,
    clean_func: Optional[Callable] = None,
    split_paragraphs: bool = False,
    split_answers: bool = False,
    encoding: Optional[str] = None,
    process_num: int = 20,
) -> List[dict]:
    """
    Convert all files(.txt, .pdf, .docx) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param dir_path: path for the documents to be written to the DocumentStore
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.
    :param split_answers: split text into two columns, including question column, answer column.
    :param encoding: character encoding to use when converting pdf documents.
    """
    file_paths = [p for p in Path(dir_path).glob("**/*")]
    allowed_suffixes = [".pdf", ".txt", ".docx", ".png", ".jpg", ".md"]
    suffix2converter: Dict[str, BaseConverter] = {}

    suffix2paths: Dict[str, List[Path]] = {}
    for path in file_paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            if file_suffix not in suffix2paths:
                suffix2paths[file_suffix] = []
            suffix2paths[file_suffix].append(path)
        elif not path.is_dir():
            logger.warning(
                "Skipped file {0} as type {1} is not supported here. "
                "See pipelines.file_converter for support of more file types".format(path, file_suffix)
            )
    # No need to initialize converter if file type not present
    for file_suffix in suffix2paths.keys():
        if file_suffix == ".pdf":
            suffix2converter[file_suffix] = PDFToTextConverter()
        if file_suffix == ".txt":
            suffix2converter[file_suffix] = TextConverter()
        if file_suffix == ".docx":
            suffix2converter[file_suffix] = DocxToTextConverter()
        if file_suffix == ".png" or file_suffix == ".jpg":
            suffix2converter[file_suffix] = ImageToTextConverter()
        if file_suffix == ".md":
            suffix2converter[file_suffix] = MarkdownConverter()
    documents = []
    for suffix, paths in suffix2paths.items():
        for path in paths:
            logger.info("Converting {}".format(path))
            list_documents = suffix2converter[suffix].convert(
                file_path=path,
                meta=None,
                encoding=encoding,
            )  # PDFToTextConverter, TextConverter, ImageToTextConverter and DocxToTextConverter return a list containing a single dict
            for document in list_documents:
                text = document["content"]
                if clean_func:
                    text = clean_func(text)

                if split_paragraphs:
                    for para in text.split("\n"):
                        if not para.strip():  # skip empty paragraphs
                            continue
                        if split_answers:
                            query, answer = para.split("\t")
                            meta_data = {"name": path.name, "answer": answer}
                            # Add image list parsed from docx into meta
                            if document["meta"] is not None and "images" in document["meta"]:
                                meta_data["images"] = document["meta"]["images"]

                            documents.append({"content": query, "meta": meta_data})
                        else:
                            meta_data = {
                                "name": path.name,
                            }
                            # Add image list parsed from docx into meta
                            if document["meta"] is not None and "images" in document["meta"]:
                                meta_data["images"] = document["meta"]["images"]
                            documents.append({"content": para, "meta": meta_data})
                else:
                    documents.append(
                        {"content": text, "meta": document["meta"] if "meta" in document else {"name": path.name}}
                    )
    return documents


def convert_files_to_dicts_splitter(
    dir_path: str,
    clean_func: Optional[Callable] = None,
    split_paragraphs: bool = False,
    split_answers: bool = False,
    encoding: Optional[str] = None,
    separator: str = "\n",
    filters: list = ["\n"],
    chunk_size: int = 300,
    chunk_overlap: int = 0,
    language: str = "chinese",
    process_num: int = 10,
) -> List[dict]:
    """
    Convert all files(.txt, .pdf, .docx) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param dir_path: path for the documents to be written to the DocumentStore
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.
    :param split_answers: split text into two columns, including question column, answer column.
    :param encoding: character encoding to use when converting pdf documents.
    """
    file_paths = [p for p in Path(dir_path).glob("**/*")]
    allowed_suffixes = [".pdf", ".txt", ".docx", ".png", ".jpg", ".md"]
    suffix2converter: Dict[str, BaseConverter] = {}

    suffix2paths: Dict[str, List[Path]] = {}
    suffix2splitter: Dict[str, BaseComponent] = {}
    for path in file_paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            if file_suffix not in suffix2paths:
                suffix2paths[file_suffix] = []
            suffix2paths[file_suffix].append(path)
        elif not path.is_dir():
            logger.warning(
                "Skipped file {0} as type {1} is not supported here. "
                "See pipelines.file_converter for support of more file types".format(path, file_suffix)
            )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        headers_to_split_on=headers_to_split_on,
        return_each_line=True,
        filters=filters,
    )
    if language == "chinese":
        docx_splitter = SpacyTextSplitter(
            separator=separator, filters=filters, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        pdf_splitter = SpacyTextSplitter(
            separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, filters=filters
        )
    else:
        docx_splitter = SpacyTextSplitter(
            separator=separator,
            filters=filters,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pipeline="en_core_web_sm",
        )
        pdf_splitter = SpacyTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            filters=filters,
            pipeline="en_core_web_sm",
        )
    text_splitter = CharacterTextSplitter(
        separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, filters=filters
    )

    imgage_splitter = CharacterTextSplitter(
        separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, filters=filters
    )
    documents = []
    # No need to initialize converter if file type not present
    for file_suffix in suffix2paths.keys():
        if file_suffix == ".pdf":
            suffix2converter[file_suffix] = PDFToTextConverter()
            suffix2splitter[file_suffix] = pdf_splitter
        if file_suffix == ".txt":
            suffix2converter[file_suffix] = TextConverter()
            suffix2splitter[file_suffix] = text_splitter
        if file_suffix == ".docx":
            suffix2converter[file_suffix] = DocxTotxtConverter()
            suffix2splitter[file_suffix] = docx_splitter
        if file_suffix == ".png" or file_suffix == ".jpg":
            suffix2converter[file_suffix] = ImageToTextConverter()
            suffix2splitter[file_suffix] = imgage_splitter
        if file_suffix == ".md":
            suffix2converter[file_suffix] = MarkdownRawTextConverter()
            suffix2splitter[file_suffix] = markdown_splitter
    for suffix, paths in suffix2paths.items():
        for path in paths:
            logger.info("Converting {}".format(path))
            list_documents = suffix2converter[suffix].convert(
                file_path=path,
                meta=None,
                encoding=encoding,
                language=language,
            )
            list_documents = document_rough_split(list_documents)
            document_number = len(list_documents)
            split_len = document_number // process_num
            if split_len == 0:
                split_len = document_number
            document_list = [i for i in range(0, document_number, split_len)]
            if document_number > document_list[-1]:
                document_list.append(document_number)
            document_combination_index = [(start, end) for start, end in zip(document_list, document_list[1:])]
            document_mul = run_process(
                document_combination_index=document_combination_index,
                list_documents=list_documents,
                splitter=suffix2splitter[suffix],
                process_num=process_num,
                split_paragraphs=split_paragraphs,
                clean_func=clean_func,
                path=path,
                split_answers=split_answers,
            )
            for item in document_mul:
                documents.extend(item)
    if filters is not None and len(filters) > 0:
        documents = clean(documents, filters)
    return documents


def clean(documents: List[dict], filters):
    for special_character in filters:
        for doc in documents:
            doc["content"] = doc["content"].replace(special_character, "")
    return documents


def tika_convert_files_to_dicts(
    dir_path: str,
    clean_func: Optional[Callable] = None,
    split_paragraphs: bool = False,
    merge_short: bool = True,
    merge_lowercase: bool = True,
) -> List[dict]:
    """
    Convert all files(.txt, .pdf) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param merge_lowercase: allow conversion of merged paragraph to lowercase
    :param merge_short: allow merging of short paragraphs
    :param dir_path: path for the documents to be written to the DocumentStore
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.
    """
    try:
        from pipelines.nodes.file_converter import TikaConverter
    except Exception as ex:
        logger.error("Tika not installed. Please install tika and try again. Error: {}".format(ex))
        raise ex
    converter = TikaConverter()
    paths = [p for p in Path(dir_path).glob("**/*")]
    allowed_suffixes = [".pdf", ".txt"]
    file_paths: List[Path] = []

    for path in paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            file_paths.append(path)
        elif not path.is_dir():
            logger.warning(
                "Skipped file {0} as type {1} is not supported here. "
                "See pipelines.file_converter for support of more file types".format(path, file_suffix)
            )

    documents = []
    for path in file_paths:
        logger.info("Converting {}".format(path))
        document = converter.convert(path)[
            0
        ]  # PDFToTextConverter, TextConverter, and DocxToTextConverter return a list containing a single dict
        meta = document["meta"] or {}
        meta["name"] = path.name
        text = document["content"]
        pages = text.split("\f")

        if split_paragraphs:
            if pages:
                paras = pages[0].split("\n\n")
                # pop the last paragraph from the first page
                last_para = paras.pop(-1) if paras else ""
                for page in pages[1:]:
                    page_paras = page.split("\n\n")
                    # merge the last paragraph in previous page to the first paragraph in this page
                    if page_paras:
                        page_paras[0] = last_para + " " + page_paras[0]
                        last_para = page_paras.pop(-1)
                        paras += page_paras
                if last_para:
                    paras.append(last_para)
                if paras:
                    last_para = ""
                    for para in paras:
                        para = para.strip()
                        if not para:
                            continue

                        # this paragraph is less than 10 characters or 2 words
                        para_is_short = len(para) < 10 or len(re.findall(r"\s+", para)) < 2
                        # this paragraph starts with a lower case and last paragraph does not end with a punctuation
                        para_is_lowercase = (
                            para and para[0].islower() and last_para and last_para[-1] not in r'.?!"\'\]\)'
                        )

                        # merge paragraphs to improve qa
                        if (merge_short and para_is_short) or (merge_lowercase and para_is_lowercase):
                            last_para += " " + para
                        else:
                            if last_para:
                                documents.append({"content": last_para, "meta": meta})
                            last_para = para
                    # don't forget the last one
                    if last_para:
                        documents.append({"content": last_para, "meta": meta})
        else:
            if clean_func:
                text = clean_func(text)
            documents.append({"content": text, "meta": meta})

    return documents
