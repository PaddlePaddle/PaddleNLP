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
import argparse
import csv
import glob
import math
import time
from collections import defaultdict
from typing import Dict, List, cast

from datasets import load_dataset
from mteb.abstasks import AbsTaskRetrieval
from prediction import Eval_modle

csv.field_size_limit(500 * 1024 * 1024)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', choices=['bloom', 'llama', 'baichuan', "bert", 'roberta', 'ernie'], default="bloom", help="The model types")
parser.add_argument("--query_model", default="bigscience/bloomz-7b1-mt", type=str, help="The ann index name")
parser.add_argument("--passage_model", default="bigscience/bloomz-7b1-mt", type=str, help="The ann index name")
parser.add_argument("--query_max_length", default=64, type=int, help="Number of element to retrieve from embedding search")
parser.add_argument("--passage_max_length", default=512, type=int, help="The embedding_dim of index")
parser.add_argument("--evaluate_all", action="store_true", help="Evaluate all checkpoints")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="The checkpoints root directory")

args = parser.parse_args()
# yapf: enable


class PaddleModel:
    def __init__(
        self,
        query_model,
        corpus_model,
        model_type="bloom",
        batch_size=1,
        max_seq_len=512,
        sep=" ",
        pooling_mode="mean_tokens",
        **kwargs,
    ):
        self.query_model = Eval_modle(
            model=query_model,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            model_type=model_type,
        )
        self.sep = sep

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        return self.query_model.run(queries, batch_size=batch_size, max_seq_len=args.query_max_length, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        return self.query_model.run(
            sentences,
            batch_size=batch_size,
            max_seq_len=args.passage_max_length,
            **kwargs,
        )


class T2RRetrieval(AbsTaskRetrieval):
    def __init__(self, num_max_passages: "int | None" = None, **kwargs):
        super().__init__(**kwargs)
        self.num_max_passages = num_max_passages or math.inf

    @property
    def description(self):
        return {
            "name": "T2RankingRetrieval",
            "reference": "https://huggingface.co/datasets/THUIR/T2Ranking",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
        }

    def evaluate(
        self,
        model_query,
        model_corpus,
        model_type="bloom",
        split="test",
        batch_size=32,
        corpus_chunk_size=None,
        target_devices=None,
        score_function="cos_sim",
        **kwargs,
    ):
        from beir.retrieval.evaluation import EvaluateRetrieval

        if not self.data_loaded:
            self.load_data()
        corpus, queries, relevant_docs = (
            self.corpus[split],
            self.queries[split],
            self.relevant_docs[split],
        )

        from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

        model = PaddleModel(model_query, model_corpus, model_type)

        model = DRES(
            model,
            batch_size=batch_size,
            corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 50000,
            **kwargs,
        )
        retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim" or "dot"
        start_time = time.time()
        results = retriever.retrieve(corpus, queries)
        end_time = time.time()
        print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values)
        mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        print(scores)
        return scores

    def load_data(self, **kwargs):
        corpus, queries, qrels = load_t2ranking_for_retraviel(self.num_max_passages)
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        self.corpus["dev"] = corpus
        self.queries["dev"] = queries
        self.relevant_docs["dev"] = qrels
        self.data_loaded = True


def load_t2ranking_for_retraviel(num_max_passages: float):
    collection_dataset = load_dataset("THUIR/T2Ranking", "collection")["train"]  # type: ignore
    dev_queries_dataset = load_dataset("THUIR/T2Ranking", "queries.dev")["train"]  # type: ignore
    dev_rels_dataset = load_dataset("THUIR/T2Ranking", "qrels.dev")["train"]  # type: ignore
    corpus = {}
    for index in range(min(len(collection_dataset), num_max_passages)):
        record = collection_dataset[index]
        record = cast(dict, record)
        pid: int = record["pid"]
        corpus[str(pid)] = {"text": record["text"]}
    queries = {}
    for record in dev_queries_dataset:
        record = cast(dict, record)
        queries[str(record["qid"])] = record["text"]

    all_qrels = defaultdict(dict)
    for record in dev_rels_dataset:
        record = cast(dict, record)
        pid: int = record["pid"]
        if pid > num_max_passages:
            continue
        all_qrels[str(record["qid"])][str(record["pid"])] = record["rel"]
    valid_qrels = {}
    for qid, qrels in all_qrels.items():
        if len(set(list(qrels.values())) - set([0])) >= 1:
            valid_qrels[qid] = qrels
    valid_queries = {}
    for qid, query in queries.items():
        if qid in valid_qrels:
            valid_queries[qid] = query
    print(f"valid qrels: {len(valid_qrels)}")
    return corpus, valid_queries, valid_qrels


if __name__ == "__main__":
    tasks = T2RRetrieval(num_max_passages=10000)
    if args.evaluate_all:
        checkpoints = glob.glob(f"{args.checkpoint_dir}/checkpoint-*")
        checkpoints.sort()
        for checkpoint in checkpoints:
            tasks.evaluate(
                model_query=checkpoint,
                model_corpus=checkpoint,
                model_type=args.model_type,
                split="dev",
            )

    else:
        tasks.evaluate(
            model_query=args.query_model,
            model_corpus=args.passage_model,
            model_type=args.model_type,
            split="dev",
        )
