### 城市百科知识智能问答系统
import argparse
import logging
import os

import paddle
from pipelines.document_stores import FAISSDocumentStore
from pipelines.utils import convert_files_to_dicts, fetch_archive_from_http, print_answers
from pipelines.nodes import ErnieReader, ErnieRanker, DensePassageRetriever

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--index_name", default='faiss_index', type=str, help="The ann index name of FAISS.")
parser.add_argument("--max_seq_len_query", default=64, type=int, help="The maximum total length of query after tokenization.")
parser.add_argument("--max_seq_len_passage", default=256, type=int, help="The maximum total length of passage after tokenization.")
parser.add_argument("--retriever_batch_size", default=16, type=int, help="The batch size of retriever to extract passage embedding for building ANN index.")
args = parser.parse_args()
# yapf: enable


def dense_qa_pipeline():

    use_gpu = True if args.device == 'gpu' else False

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
        doc_dir = "data/baike"
        city_data = "https://paddlenlp.bj.bcebos.com/applications/baike.zip"
        fetch_archive_from_http(url=city_data, output_dir=doc_dir)
        dicts = convert_files_to_dicts(dir_path=doc_dir,
                                       split_paragraphs=True,
                                       encoding='utf-8')

        if os.path.exists(args.index_name):
            os.remove(args.index_name)
        if os.path.exists(faiss_document_store):
            os.remove(faiss_document_store)

        document_store = FAISSDocumentStore(embedding_dim=768,
                                            faiss_index_factory_str="Flat")
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

    ### Ranker
    ranker = ErnieRanker(
        model_name_or_path="rocketqa-zh-dureader-cross-encoder",
        use_gpu=use_gpu)

    reader = ErnieReader(
        model_name_or_path="ernie-gram-zh-finetuned-dureader-robust",
        use_gpu=use_gpu,
        num_processes=1)

    # ### Pipeline
    from pipelines.pipelines import ExtractiveQAPipeline

    pipe = ExtractiveQAPipeline(reader, ranker, retriever)

    pipeline_params = {
        "Retriever": {
            "top_k": 50
        },
        "Ranker": {
            "top_k": 1
        },
        "Reader": {
            "top_k": 1
        }
    }

    prediction = pipe.run(query="北京市有多少个行政区？", params=pipeline_params)
    print_answers(prediction, details="minimum")

    prediction = pipe.run(query="上海常住人口有多少？", params=pipeline_params)
    print_answers(prediction, details="minimum")

    prediction = pipe.run(query="广州市总面积多大？", params=pipeline_params)
    print_answers(prediction, details="minimum")

    prediction = pipe.run(query="河北省的省会在哪里？", params=pipeline_params)
    print_answers(prediction, details="minimum")

    prediction = pipe.run(query="安徽省的简称是什么？", params=pipeline_params)
    print_answers(prediction, details="minimum")


if __name__ == "__main__":
    dense_qa_pipeline()
