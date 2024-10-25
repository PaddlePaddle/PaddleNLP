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
import os

from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import MultiModalRetriever
from pipelines.pipelines import Pipeline
from pipelines.utils import convert_files_to_dicts, fetch_archive_from_http

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--index_name", default='wukong_test', type=str, help="The ann index name of ANN.")
parser.add_argument("--embedding_dim", default=768, type=int, help="The embedding_dim of index")
parser.add_argument("--query_embedding_model", default="PaddlePaddle/ernie_vil-2.0-base-zh", type=str, help="The query_embedding_model path")
parser.add_argument("--document_embedding_model", default="PaddlePaddle/ernie_vil-2.0-base-zh", type=str, help="The document_embedding_model path")
args = parser.parse_args()
# yapf: enable


def image_text_retrieval_tutorial():
    faiss_document_store = "faiss_document_store.db"
    if os.path.exists(args.index_name) and os.path.exists(faiss_document_store):
        # Connect to existed FAISS Index
        document_store = FAISSDocumentStore.load(args.index_name)
        retriever_mm = MultiModalRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            query_type="image",
            document_embedding_models={"text": args.document_embedding_model},
        )
    else:
        doc_dir = "data/wukong_text"
        wukong_data = "https://paddlenlp.bj.bcebos.com/applications/wukong_text.zip"
        fetch_archive_from_http(url=wukong_data, output_dir=doc_dir)
        dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True, encoding="utf-8")
        if os.path.exists(args.index_name):
            os.remove(args.index_name)
        if os.path.exists(faiss_document_store):
            os.remove(faiss_document_store)
        document_store = FAISSDocumentStore(embedding_dim=args.embedding_dim, faiss_index_factory_str="Flat")
        retriever_mm = MultiModalRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            query_type="image",
            document_embedding_models={"text": args.document_embedding_model},
        )
        # Update metadata
        document_store.write_documents(dicts)
        # Update Embedding
        document_store.update_embeddings(retriever_mm)
        # Save index
        document_store.save(args.index_name)
    pipe = Pipeline()
    pipe.add_node(component=retriever_mm, name="Retriever", inputs=["Query"])
    result = pipe.run(
        query="data/wukong_test/00001469-0017.jpg", params={"Retriever": {"top_k": 5, "query_type": "image"}}
    )
    print(result)


if __name__ == "__main__":
    image_text_retrieval_tutorial()
