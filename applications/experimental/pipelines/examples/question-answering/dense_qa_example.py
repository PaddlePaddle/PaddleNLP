# ## Task: Question Answering for Game of Thrones
#
# Question Answering can be used in a variety of use cases. A very common one:  Using it to navigate through complex
# knowledge bases or long documents ("search setting").
#
# A "knowledge base" could for example be your website, an internal wiki or a collection of financial reports.
# In this tutorial we will work on a slightly different domain: "Game of Thrones".
#
# Let's see how we can use a bunch of Wikipedia articles to answer a variety of questions about the
# marvellous seven kingdoms.

import paddle
import logging
import os
from pipelines.document_stores import FAISSDocumentStore
from pipelines.utils import clean_wiki_text, convert_files_to_dicts, fetch_archive_from_http, print_answers, launch_es
from pipelines.nodes import ErnieReader, ErnieRanker, DensePassageRetriever


def dense_qa_pipeline():
    logger = logging.getLogger(__name__)

    # ## Document Store
    #
    # pipelines finds answers to queries within the documents stored in a `DocumentStore`. The current implementations of
    # `DocumentStore` include `ElasticsearchDocumentStore`, `FAISSDocumentStore`, `SQLDocumentStore`, and `InMemoryDocumentStore`.
    #
    # **Here:** We recommended Elasticsearch as it comes preloaded with features like full-text queries, BM25 retrieval,
    # and vector storage for text embeddings.
    # **Alternatives:** If you are unable to setup an Elasticsearch instance, then follow the Tutorial 3
    # for using SQL/InMemory document stores.
    # **Hint**:
    # This tutorial creates a new document store instance with Wikipedia articles on Game of Thrones. However, you can
    # configure pipelines to work with your existing document stores.
    #
    # Start an Elasticsearch server
    # You can start Elasticsearch on your local machine instance using Docker. If Docker is not readily available in
    # your environment (e.g. in Colab notebooks), then you can manually download and execute Elasticsearch from source.

    faiss_index_path = "faiss_index"
    faiss_document_store = "faiss_document_store.db"
    if os.path.exists(faiss_index_path) and os.path.exists(faiss_document_store):
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
            embed_title=False,
        )
    else:
        doc_dir = "data/baike"
        China_cities_data = "https://paddlenlp.bj.bcebos.com/applications/baike.zip"
        fetch_archive_from_http(url=China_cities_data, output_dir=doc_dir)
        dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True)

        if os.path.exists(faiss_index_path):
            os.remove(faiss_index_path)
        if os.path.exists(faiss_document_store):
            os.remove(faiss_document_store)

        document_store = FAISSDocumentStore(embedding_dim=768, faiss_index_factory_str="Flat")
        document_store.write_documents(dicts)

        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="rocketqa-zh-dureader-query-encoder",
            passage_embedding_model="rocketqa-zh-dureader-query-encoder",
            max_seq_len_query=64,
            max_seq_len_passage=256,
            batch_size=16,
            use_gpu=True,
            embed_title=False,
        )

        # update Embedding
        document_store.update_embeddings(retriever)

        # save index
        document_store.save(faiss_index_path)

    # ### Reader
    #
    # A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based
    # on powerful, but slower deep learning models.
    #
    # pipelines currently supports Readers based on the frameworks FARM and Transformers.
    # With both you can either load a local model or one from Hugging Face's model hub (https://huggingface.co/models).
    # **Here:** a medium sized RoBERTa QA model using a Reader based on
    # **Alternatives (Reader):** TransformersReader (leveraging the `pipeline` of the Transformers package)
    # **Alternatives (Models):** e.g. "distilbert-base-uncased-distilled-squad" (fast) or
    # **Hint:** You can adjust the model to return "no answer possible" with the no_ans_boost. Higher values mean
    #           the model prefers "no answer possible"

    ### Ranker
    ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=True)

    reader = ErnieReader(model_name_or_path="ernie-gram-zh-finetuned-dureader-robust", use_gpu=True, num_processes=1)

    # ### Pipeline
    #
    # With a pipelines `Pipeline` you can stick together your building blocks to a search pipeline.
    # Under the hood, `Pipelines` are Directed Acyclic Graphs (DAGs) that you can easily customize for your own use cases.
    # To speed things up, pipelines also comes with a few predefined Pipelines. One of them is the `ExtractiveQAPipeline` that combines a retriever and a reader to answer our questions.
    from pipelines.pipelines import ExtractiveQAPipeline

    pipe = ExtractiveQAPipeline(reader, ranker, retriever)

    ## Voilà! Ask a question!
    pipeline_params = {"Retriever": {"top_k": 50}, "Ranker":{"top_k": 1}, "Reader": {"top_k": 1}}

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
