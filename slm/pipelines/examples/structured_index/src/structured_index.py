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

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional

import faiss
import numpy as np

from .index import Indexer
from .load import Loader
from .parse import DocumentStructureParser
from .summarize import Summarizer


class StructuredIndexer(object):
    def __init__(self, log_dir: Optional[str] = None):
        self.set_logging(log_dir=log_dir)

    def set_logging(self, log_dir: Optional[str] = None):
        """
        Configures the logging settings for the application.

        Args:
            log_dir (Optional[str]): The directory where log files will be stored. If None, the default directory '.logs' is used.

        Returns:
            None
        """
        if log_dir is None:
            log_dir = ".logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"{str(datetime.now())}.log")

        level = logging.DEBUG
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s",
            datefmt="%Y-%m-%d(%a)%H:%M:%S",
            filename=log_filename,
            filemode="w",
        )

        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter("[%(levelname)-8s] %(message)s")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

    def prepare_dir(self, input_dir: str, mode: str, output_dir: str = None, save: bool = True):
        """
        Prepares the input directory for processing and optionally creates an output directory.

        Args:
            input_dir (str): The path to the input directory.
            mode (str): The mode in which the files are being processed (e.g., "read", "write").
            output_dir (str, optional): The path to the output directory. Defaults to None.
            save (bool, optional): Whether to create the output directory if it doesn't exist. Defaults to True.

        Raises:
            NotADirectoryError: If the `input_dir` is not a directory.
            FileNotFoundError: If the `input_dir` does not exist.

        Returns:
            None

        Logs:
            - An info message indicating the mode of operation and the input directory.
            - An error message if the `input_dir` is not a directory.
            - An error message if the `input_dir` does not exist.
        """
        if not os.path.isdir(input_dir):
            logging.error(f"Path {input_dir} is not a directory.")
            raise NotADirectoryError(f"Path {input_dir} is not a directory.")
        if not os.path.exists(input_dir):
            logging.error(f"Path {input_dir} does not exist.")
            raise FileNotFoundError(f"Path {input_dir} does not exist.")
        logging.info(f"{mode} files in {input_dir}")
        if save and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    def search(
        self,
        queries_dict: Dict,
        input_dir: str = "data/index",
        output_dir: str = "data/search_result",
        model_name_or_path: str = "BAAI/bge-large-en-v1.5",
        top_k: int = 10,
        embedding_batch_size: int = 128,
        save: bool = True,
    ) -> List[Dict]:
        """
        Perform a semantic search on a pre-indexed corpus using a specified embedding model.

        Args:
            queries_dict (Dict): A dictionary where keys are query identifiers and values are the actual queries.
            input_dir (str, optional): The directory containing the pre-indexed corpus and embeddings. Defaults to "data/index".
            output_dir (str, optional): The directory where the search results will be saved. Defaults to "data/search_result".
            model_name_or_path (str, optional): The name or path of the embedding model to use. Defaults to "BAAI/bge-large-en-v1.5".
            top_k (int, optional): The number of top results to retrieve for each query. Defaults to 10.
            embedding_batch_size (int, optional): The batch size for embedding the queries. Defaults to 128.
            save (bool, optional): Whether to save the search results to a file. Defaults to True.

        Returns:
            List[Dict]: A list of dictionaries containing the search results for each query.

        Raises:
            AssertionError: If the number of index cache files does not match the number of corpus files, or if the corpus embeddings do not match in shape.

        Notes:
            - The input directory should contain the pre-computed embeddings and corresponding corpus files.
            - The output directory will be created if it does not exist.
            - The search results are saved in a JSON file with a timestamp in the filename.
        """
        input_dir = os.path.join(input_dir, model_name_or_path)  # The embedding model must be the same
        output_dir = os.path.join(output_dir, model_name_or_path)
        self.prepare_dir(input_dir=input_dir, output_dir=output_dir, save=True, mode="Searching")

        indexcachefiles = []
        corpusfiles = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".npy"):
                    indexcachefiles.append(os.path.join(root, file))
                    document_name = os.path.splitext(os.path.basename(file))[0]
                    corpusfiles.append(os.path.join(root, document_name + ".pkl"))
                    logging.info(f"Loading Index cache from {indexcachefiles[-1]}")
                    logging.info(f"Loading Corpus from {corpusfiles[-1]}")
        assert len(indexcachefiles) == len(corpusfiles) and len(indexcachefiles) > 0

        indexer = Indexer(model_name_or_path=model_name_or_path)

        corpus_infos: List[Dict] = []
        for file in corpusfiles:
            corpus_infos.extend(pickle.load(open(file, "rb")))
        logging.info(f"corpus/index quantity: {len(corpus_infos)}")
        corpus_embeddings = np.load(indexcachefiles[0])
        for file in indexcachefiles[1:]:
            temp_corpus_embeddings = np.load(file)
            assert corpus_embeddings.shape[1] == temp_corpus_embeddings.shape[1]
            corpus_embeddings = np.concatenate((corpus_embeddings, temp_corpus_embeddings), axis=0)
        index: faiss.IndexFlatIP = indexer.build_engine(corpus_embeddings)
        assert len(corpus_infos) == index.ntotal

        queries = list(queries_dict.values())
        keys = list(queries_dict.keys())
        query_embeddings = indexer.get_corpus_embedding(queries, batch_size=embedding_batch_size)
        hits = indexer._search(index, query_embeddings, top_k)
        assert len(hits) == len(queries)

        for passages in hits:
            for p_dict in passages:
                id = p_dict["corpus_id"]
                if id < len(corpus_infos) and id >= 0:
                    p_dict["content"] = corpus_infos[id]["content"]
                    p_dict["source"] = corpus_infos[id]["source"]
                    p_dict["level"] = corpus_infos[id]["level"]
        if save:
            result_dict = {}
            for i in range(len(hits)):
                hit_dict = {"query": queries[i], "hits": hits[i]}
                assert queries_dict[keys[i]] == queries[i]
                result_dict[keys[i]] = hit_dict
            queryfilename = f'query_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
            with open(os.path.join(output_dir, queryfilename), "w") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)
            logging.info(f"Saved to: {queryfilename}")
        return result_dict

    def pipeline(
        self,
        filepath: str,
        parse_model_name_or_path: str = "Qwen/Qwen2-7B-Instruct",
        parse_model_url: str = None,
        summarize_model_name_or_path: str = "Qwen/Qwen2-7B-Instruct",
        summarize_model_url: str = None,
        encode_model_name_or_path: str = "BAAI/bge-large-en-v1.5",
    ):
        """
        Process a document through a series of steps including loading, parsing, summarizing and indexing to get Structured Index.

        Args:
            filepath (str): The path to the document file to be processed.
            parse_model_name_or_path (str, optional): The name or path of the model used for document parsing. Defaults to "Qwen/Qwen2-7B-Instruct".
            parse_model_url (str, optional): The URL of the model used for document parsing. Defaults to None.
            summarize_model_name_or_path (str, optional): The name or path of the model used for document summarization. Defaults to "Qwen/Qwen2-7B-Instruct".
            summarize_model_url (str, optional): The URL of the model used for document summarization. Defaults to None.
            encode_model_name_or_path (str, optional): The name or path of the model used for document encoding (indexing). Defaults to "BAAI/bge-large-en-v1.5".

        Returns:
            dict: A dictionary containing the processed document data, or None if the document could not be loaded.

        Steps:
            1. Load the document from the specified file path.
            2. Parse the document structure using the specified parsing model.
            3. Summarize the document content using the specified summarization model.
            4. Index the document using the specified encoding model.

        Notes:
            - The document is processed in a sequence of steps: loading, parsing, summarizing, and indexing.
            - Each step uses a different model specified by the respective model name or path and URL.
            - If the document cannot be loaded, the function returns None.
        """
        doc = Loader.load_file(filepath=filepath)
        if doc is None:
            return None
        processor = DocumentStructureParser(model_name_or_path=parse_model_name_or_path, url=parse_model_url)
        doc = processor.parse_doc(data=doc)
        processor = Summarizer(model_name_or_path=summarize_model_name_or_path, url=summarize_model_url)
        doc = processor.summarize_doc(data=doc)
        processor = Indexer(model_name_or_path=encode_model_name_or_path)
        doc = processor.index_doc(file_path=filepath, data=doc)
        return doc
