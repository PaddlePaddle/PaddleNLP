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

import math
from collections import defaultdict
from typing import cast

from datasets import load_dataset
from mteb import MTEB
from mteb.abstasks import AbsTaskRetrieval

from paddlenlp import Taskflow


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

    def load_data(self, **kwargs):
        corpus, queries, qrels = load_t2ranking_for_retraviel(self.num_max_passages)
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        self.corpus["dev"] = corpus
        self.queries["dev"] = queries
        self.relevant_docs["dev"] = qrels
        self.data_loaded = True


def load_t2ranking_for_retraviel(num_max_passages: float):
    import pandas

    collection_dataset = pandas.read_csv("T2Ranking/data/collection.tsv", sep="\t")
    # collection_dataset = load_dataset('THUIR/T2Ranking', 'collection')['train']  # type: ignore
    dev_queries_dataset = load_dataset("THUIR/T2Ranking", "queries.dev")["train"]  # type: ignore
    dev_rels_dataset = load_dataset("THUIR/T2Ranking", "qrels.dev")["train"]  # type: ignore
    corpus = {}
    for index in range(min(len(collection_dataset), num_max_passages)):
        record = collection_dataset.iloc[index, :]
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


class MyModel:
    def __init__(self, batch_size=32, pooling_mode="mean_tokens", max_seq_len=512):
        self.model = Taskflow(
            "feature_extraction",
            model="moka-ai/m3e-base",
            pooling_mode=pooling_mode,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            _static_mode=True,
        )

    def encode(self, sentences, **kwargs):
        embedding = self.model(sentences)["features"].detach().cpu().numpy()
        return embedding


model = MyModel()
evaluation = MTEB(tasks=[T2RRetrieval(num_max_passages=10000)])
evaluation.run(model)
