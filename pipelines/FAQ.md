## FAQ

#### pip安装htbuilder包报错，`UnicodeDecodeError: 'gbk' codec can't decode byte....`

windows的默认字符gbk导致的，可以使用源码进行安装，源码已经进行了修复。

```
git clone https://github.com/tvst/htbuilder.git
cd htbuilder/
python setup install
```

#### 语义检索系统可以跑通，但终端输出字符是乱码怎么解决？

+ 通过如下命令设置操作系统默认编码为 zh_CN.UTF-8
```bash
export LANG=zh_CN.UTF-8
```

#### Linux上安装elasticsearch出现错误 `java.lang.RuntimeException: can not run elasticsearch as root`

elasticsearch 需要在非root环境下运行，可以做如下的操作：

```
adduser est
chown est:est -R ${HOME}/elasticsearch-8.3.2/
cd ${HOME}/elasticsearch-8.3.2/
su est
./bin/elasticsearch
```

#### Mac OS上安装elasticsearch出现错误 `flood stage disk watermark [95%] exceeded on.... all indices on this node will be marked read-only`

elasticsearch默认达到95％就全都设置只读，可以腾出一部分空间出来再启动，或者修改 `config/elasticsearch.pyml`。
```
cluster.routing.allocation.disk.threshold_enabled: false
```

#### nltk_data加载失败的错误 `[nltk_data] Error loading punkt: [Errno 60] Operation timed out`

在命令行里面输入python,然后输入下面的命令进行下载：

```
import nltk
nltk.download('punkt')
```
如果下载还是很慢，可以手动[下载](https://github.com/nltk/nltk_data/tree/gh-pages/packages/tokenizers)，然后放入本地的`~/nltk_data/tokenizers`进行解压即可。

#### 服务端运行报端口占用的错误 `[Errno 48] error while attempting to bind on address ('0.0.0.0',8891): address already in use`

```
lsof -i:8891
kill -9 PID # PID为8891端口的进程
```

#### faiss 安装上了但还是显示找不到faiss怎么办？

推荐您使用anaconda进行单独安装，安装教程请参考[faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

```
# CPU-only version
conda install -c pytorch faiss-cpu

# GPU(+CPU) version
conda install -c pytorch faiss-gpu
```

#### 如何更换pipelines中预置的模型？

更换系统预置的模型以后，由于模型不一样了，需要重新构建索引，并修改相关的配置文件。以语义索引为例，需要修改2个地方，第一个地方是`utils/offline_ann.py`,另一个是`rest_api/pipeline/semantic_search.yaml`，并重新运行：

首先修改`utils/offline_ann.py`：

```
python utils/offline_ann.py --index_name dureader_robust_base_encoder \
                            --doc_dir data/dureader_dev \
                            --query_embedding_model rocketqa-zh-base-query-encoder \
                            --passage_embedding_model rocketqa-zh-base-para-encoder \
                            --embedding_dim 768 \
                            --delete_index
```

然后修改`rest_api/pipeline/semantic_search.yaml`文件：

```
components:    # define all the building-blocks for Pipeline
  - name: DocumentStore
    type: ElasticsearchDocumentStore  # consider using MilvusDocumentStore or WeaviateDocumentStore for scaling to large number of documents
    params:
      host: localhost
      port: 9200
      index: dureader_robust_base_encoder # 修改索引名
      embedding_dim: 768   # 修改向量的维度
  - name: Retriever
    type: DensePassageRetriever
    params:
      document_store: DocumentStore    # params can reference other components defined in the YAML
      top_k: 10
      query_embedding_model: rocketqa-zh-base-query-encoder  # 修改Retriever的query模型名
      passage_embedding_model: rocketqa-zh-base-para-encoder # 修改 Retriever的para模型
      embed_title: False
  - name: Ranker       # custom-name for the component; helpful for visualization & debugging
    type: ErnieRanker    # pipelines Class name for the component
    params:
      model_name_or_path: rocketqa-base-cross-encoder  # 修改 ErnieRanker的模型名
      top_k: 3
```

然后重新运行：

```bash
# 指定语义检索系统的Yaml配置文件
export PIPELINE_YAML_PATH=rest_api/pipeline/semantic_search.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```

#### 运行faiss examples出现了错误：`sqlalchemy.exec.OperationalError: (sqlite3.OperationalError) too many SQL variables`

python 3.7版本引起的错误，修改如下代码：

```
# 增加batch_size参数，传入一个数值即可
document_store.update_embeddings(retriever, batch_size=256)
```

#### 运行后台程序出现了错误：`Exception: Failed loading pipeline component 'DocumentStore': RequestError(400, 'illegal_argument_exception', 'Mapper for [embedding] conflicts with existing mapper:\n\tCannot update parameter [dims] from [312] to [768]')`

以语义检索为例，这是因为模型的维度不对造成的，请检查一下 `elastic search`中的文本的向量的维度和`semantic_search.yaml`里面`DocumentStore`设置的维度`embedding_dim`是否一致，如果不一致，请重新使用`utils/offline_ann.py`构建索引。总之，请确保构建索引所用到的模型和`semantic_search.yaml`设置的模型是一致的。
