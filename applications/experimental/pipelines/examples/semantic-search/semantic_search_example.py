import paddle
import os

from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import DensePassageRetriever, ErnieRanker
from pipelines.utils import convert_files_to_dicts, fetch_archive_from_http

# Note: Here import paddle occure core


def semantic_search_tutorial():
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
        doc_dir = "data/dureader_robust_processed"
        dureader_data = "https://paddlenlp.bj.bcebos.com/applications/dureader_robust_processed.zip"
        fetch_archive_from_http(url=dureader_data, output_dir=doc_dir)
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
            batch_size=8,
            use_gpu=True,
            embed_title=False, )

        # update Embedding
        document_store.update_embeddings(retriever)

        # save index
        document_store.save(faiss_index_path)

    ### Retriever
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="rocketqa-zh-dureader-query-encoder",
        passage_embedding_model="rocketqa-zh-dureader-query-encoder",
        max_seq_len_query=64,
        max_seq_len_passage=256,
        batch_size=16,
        use_gpu=True,
        embed_title=False, )

    ### Ranker
    ranker = ErnieRanker(
        model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=True)

    ### Pipeline
    from pipelines.pipelines import SemanticSearchPipeline
    pipe = SemanticSearchPipeline(retriever, ranker)

    ## ask question.
    prediction = pipe.run(
        query="亚马逊河流的介绍",
        params={"Retriever": {
            "top_k": 50
        },
                "Ranker": {
                    "top_k": 5
                }})

    print(prediction)


if __name__ == "__main__":
    semantic_search_tutorial()
