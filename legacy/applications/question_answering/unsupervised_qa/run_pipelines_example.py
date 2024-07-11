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

import argparse
import os
from pprint import pprint

from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import (
    AnswerExtractor,
    DensePassageRetriever,
    ErnieRanker,
    QAFilter,
    QuestionGenerator,
)
from pipelines.pipelines import QAGenerationPipeline, SemanticSearchPipeline
from pipelines.utils import convert_files_to_dicts, print_documents

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--index_name", default='faiss_index', type=str, help="The ann index name of FAISS.")
parser.add_argument("--max_seq_len_query", default=64, type=int, help="The maximum total length of query after tokenization.")
parser.add_argument("--max_seq_len_passage", default=256, type=int, help="The maximum total length of passage after tokenization.")
parser.add_argument("--retriever_batch_size", default=16, type=int, help="The batch size of retriever to extract passage embedding for building ANN index.")
parser.add_argument("--doc_dir", default="data/my_data", type=str, help="The question-answer pairs file to be loaded when building ANN index.")
parser.add_argument("--source_file", default=None, type=str, help="The source raw texts file to be loaded when creating question-answer pairs.")

args = parser.parse_args()
# yapf: enable


def dense_faq_pipeline():
    use_gpu = True if args.device == "gpu" else False
    faiss_document_store = "faiss_document_store.db"
    if os.path.exists(args.index_name) and os.path.exists(faiss_document_store):
        # connect to existed FAISS Index
        document_store = FAISSDocumentStore.load(args.index_name)
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="rocketqa-zh-dureader-query-encoder",
            passage_embedding_model="rocketqa-zh-dureader-query-encoder",
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            batch_size=args.retriever_batch_size,
            use_gpu=use_gpu,
            embed_title=False,
        )
    else:
        dicts = convert_files_to_dicts(
            dir_path=args.doc_dir, split_paragraphs=True, split_answers=True, encoding="utf-8"
        )

        if os.path.exists(args.index_name):
            os.remove(args.index_name)
        if os.path.exists(faiss_document_store):
            os.remove(faiss_document_store)

        document_store = FAISSDocumentStore(embedding_dim=768, faiss_index_factory_str="Flat")
        document_store.write_documents(dicts)

        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="rocketqa-zh-dureader-query-encoder",
            passage_embedding_model="rocketqa-zh-dureader-query-encoder",
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            batch_size=args.retriever_batch_size,
            use_gpu=use_gpu,
            embed_title=False,
        )

        # update Embedding
        document_store.update_embeddings(retriever)

        # save index
        document_store.save(args.index_name)

    # Ranker
    ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=use_gpu)

    pipe = SemanticSearchPipeline(retriever, ranker)

    pipeline_params = {"Retriever": {"top_k": 50}, "Ranker": {"top_k": 1}}
    prediction = pipe.run(query="世界上最早的地雷发明者是谁？", params=pipeline_params)

    print_documents(prediction, print_name=False, print_meta=True)


def qa_generation_pipeline():
    answer_extractor = AnswerExtractor(
        model="uie-base-answer-extractor",
        device=args.device,
        schema=["答案"],
        max_answer_candidates=3,
        position_prob=0.01,
        batch_size=1,
    )

    question_generator = QuestionGenerator(
        model="unimo-text-1.0-question-generation",
        device=args.device,
        num_return_sequences=2,
    )

    qa_filter = QAFilter(
        model="uie-base-qa-filter",
        device=args.device,
        schema=["答案"],
        position_prob=0.1,
    )

    pipe = QAGenerationPipeline(
        answer_extractor=answer_extractor, question_generator=question_generator, qa_filter=qa_filter
    )
    pipeline_params = {"QAFilter": {"is_filter": True}}

    # list example
    meta = [
        "世界上最早的电影院是美国洛杉矶的“电气剧场”，建于1902年。",
        "以脸书为例，2020年时，54%的成年人表示，他们从该平台获取新闻。而现在，这个数字下降到了44%。与此同时，YouTube在过去几年里一直保持平稳，约有三分之一的用户在该平台上获取新闻。",
    ]
    prediction = pipe.run(meta=meta, params=pipeline_params)
    prediction = prediction["filtered_cqa_triples"]
    pprint(prediction)

    # file example
    if args.source_file:
        meta = []
        with open(args.source_file, "r", encoding="utf-8") as rf:
            for line in rf:
                meta.append(line.strip())
        prediction = pipe.run(meta=meta, params=pipeline_params)
        prediction = prediction["filtered_cqa_triples"]
        if not os.path.exists(args.doc_dir):
            os.makedirs(args.doc_dir)
        with open(os.path.join(args.doc_dir, "generated_qa_pairs.txt"), "w", encoding="utf-8") as wf:
            for pair in prediction:
                wf.write(pair["synthetic_question"].strip() + "\t" + pair["synthetic_answer"].strip() + "\n")


if __name__ == "__main__":
    qa_generation_pipeline()
    dense_faq_pipeline()
