#!/usr/bin/env python3

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

# Copyright GC-DPR authors.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
 Command line tool to get dense results and validate them
"""

import argparse
import csv
import glob
import gzip
import json
import logging
import pickle
import time
from typing import Dict, Iterator, List, Tuple

import numpy as np
import paddle
from biencoder_base_model import BiEncoder
from faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer, DenseIndexer
from NQdataset import BertTensorizer
from paddle import Tensor as T
from paddle import nn
from qa_validation import calculate_matches

from paddlenlp.transformers.bert.modeling import BertModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(self, question_encoder: nn.Layer, batch_size: int, tensorizer: BertTensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with paddle.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q) for q in questions[batch_start : batch_start + bsz]
                ]
                q_ids_batch = paddle.stack(batch_token_tensors, axis=0)
                q_seg_batch = paddle.zeros_like(q_ids_batch)
                out = self.question_encoder.get_question_pooled_embedding(q_ids_batch, q_seg_batch)
                query_vectors.extend(out)
                if len(query_vectors) % 100 == 0:
                    logger.info("Encoded queries %d", len(query_vectors))

        query_tensor = paddle.to_tensor(query_vectors)
        logger.info("Total encoded queries tensor %s", query_tensor.shape[0])
        assert query_tensor.shape[0] == len(questions)
        return query_tensor

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        return results


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info("Reading data from: %s", ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, "rt") as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    return docs


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append(
            {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):

    tensorizer = BertTensorizer()
    question_model = BertModel.from_pretrained(args.que_model_path)
    context_model = BertModel.from_pretrained(args.con_model_path)
    model = BiEncoder(question_encoder=question_model, context_encoder=context_model)
    model.eval()
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(768, args.index_buffer)
    else:
        index = DenseFlatIndexer(768, args.index_buffer)

    retriever = DenseRetriever(model, args.batch_size, tensorizer, index)
    # get questions & answers
    questions = []
    question_answers = []
    for ds_item in parse_qa_csv_file(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)
    questions_tensor = retriever.generate_question_vectors(questions)
    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)

    logger.info("Reading all passages data from files: %s", input_paths)
    retriever.index.index_data(input_paths)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)
    all_passages = load_passages(args.ctx_file)
    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    questions_doc_hits = validate(
        all_passages, question_answers, top_ids_and_scores, args.validation_workers, args.match
    )
    if args.out_file:
        save_results(all_passages, questions, question_answers, top_ids_and_scores, questions_doc_hits, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa_file",
        required=True,
        type=str,
        default=None,
        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]",
    )
    parser.add_argument(
        "--ctx_file",
        required=True,
        type=str,
        default=None,
        help="All passages file in the tsv format: id \\t passage_text \\t title",
    )
    parser.add_argument(
        "--encoded_ctx_file",
        type=str,
        default=None,
        help="Glob path to encoded passages (from generate_dense_embeddings tool)",
    )
    parser.add_argument("--out_file", type=str, default=None, help="output .json file path to write results to ")
    parser.add_argument(
        "--match", type=str, default="string", choices=["regex", "string"], help="Answer matching logic type"
    )
    parser.add_argument("--n-docs", type=int, default=200, help="Amount of top docs to return")
    parser.add_argument(
        "--validation_workers", type=int, default=16, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument(
        "--index_buffer", type=int, default=50000, help="Temporal memory data buffer size (in samples) for indexer"
    )
    parser.add_argument(
        "--hnsw_index", action="store_true", help="If enabled, use inference time efficient HNSW index"
    )
    parser.add_argument("--que_model_path", required=True, type=str)
    parser.add_argument("--con_model_path", required=True, type=str)
    args = parser.parse_args()

    main(args)
