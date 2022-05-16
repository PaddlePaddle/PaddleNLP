import paddle

from pipelines.document_stores import ElasticsearchDocumentStore
from pipelines.nodes import DensePassageRetriever, ErnieRanker


def semantic_search_tutorial():
    document_store = ElasticsearchDocumentStore(
        host="localhost",
        port="9200",
        username="",
        password="",
        index="dureader_robust_query_encoder")

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

    prediction = pipe.run(
        query="北京有多少个行政区？",
        params={"Retriever": {
            "top_k": 50
        },
                "Ranker": {
                    "top_k": 5
                }})

    print(prediction)


if __name__ == "__main__":
    semantic_search_tutorial()
