import argparse

import paddle
from pipelines.utils import convert_files_to_dicts
from pipelines.document_stores import ElasticsearchDocumentStore
from pipelines.nodes import DensePassageRetriever
from pipelines.utils import launch_es

parser = argparse.ArgumentParser()
parser.add_argument("--index_name",
                    default='baike_cities',
                    type=str,
                    help="The index name of the elasticsearch engine")
parser.add_argument("--doc_dir",
                    default='data/baike/',
                    type=str,
                    help="The doc path of the corpus")

args = parser.parse_args()


def offline_ann(index_name, doc_dir):

    launch_es()

    document_store = ElasticsearchDocumentStore(host="127.0.0.1",
                                                port="9200",
                                                username="",
                                                password="",
                                                index=index_name)

    # 将每篇文档按照段落进行切分
    dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True)

    print(dicts[:3])

    # 文档数据写入数据库
    document_store.write_documents(dicts)

    ### 语义索引模型
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

    # 建立索引库
    document_store.update_embeddings(retriever)


if __name__ == "__main__":
    offline_ann(args.index_name, args.doc_dir)
