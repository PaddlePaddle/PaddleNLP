### 城市百科知识智能问答系统

import paddle
import logging
import os
from pipelines.document_stores import FAISSDocumentStore
from pipelines.utils import clean_wiki_text, convert_files_to_dicts, fetch_archive_from_http, print_answers, launch_es
from pipelines.nodes import ErnieReader, ErnieRanker, DensePassageRetriever


def dense_qa_pipeline():
    logger = logging.getLogger(__name__)

    faiss_index_path = "faiss_index"
    faiss_document_store = "faiss_document_store.db"
    if os.path.exists(faiss_index_path) and os.path.exists(
            faiss_document_store):
        # connect to existed FAISS Index
        document_store = FAISSDocumentStore.load(faiss_index_path)
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="rocketqa-zh-dureader-query-encoder",
            passage_embedding_model="rocketqa-zh-dureader-query-encoder",
            max_seq_len_query=64,
            max_seq_len_passage=256,
            batch_size=16,
            use_gpu=True,
            embed_title=False, )
    else:
        doc_dir = "data/baike"
        city_data = "https://paddlenlp.bj.bcebos.com/applications/baike.zip"
        fetch_archive_from_http(url=city_data, output_dir=doc_dir)
        dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True)

        if os.path.exists(faiss_index_path):
            os.remove(faiss_index_path)
        if os.path.exists(faiss_document_store):
            os.remove(faiss_document_store)

        document_store = FAISSDocumentStore(
            embedding_dim=768, faiss_index_factory_str="Flat")
        document_store.write_documents(dicts)

        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="rocketqa-zh-dureader-query-encoder",
            passage_embedding_model="rocketqa-zh-dureader-query-encoder",
            max_seq_len_query=64,
            max_seq_len_passage=256,
            batch_size=16,
            use_gpu=True,
            embed_title=False, )

        # update Embedding
        document_store.update_embeddings(retriever)

        # save index
        document_store.save(faiss_index_path)

    ### Ranker
    ranker = ErnieRanker(
        model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=True)

    reader = ErnieReader(
        model_name_or_path="ernie-gram-zh-finetuned-dureader-robust",
        use_gpu=True,
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
