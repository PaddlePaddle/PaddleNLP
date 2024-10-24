## FAQ

#### pip 安装 htbuilder 包报错，`UnicodeDecodeError: 'gbk' codec can't decode byte....`

windows 的默认字符 gbk 导致的，可以使用源码进行安装，源码已经进行了修复。

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

#### Linux 上安装 elasticsearch 出现错误 `java.lang.RuntimeException: can not run elasticsearch as root`

elasticsearch 需要在非 root 环境下运行，可以做如下的操作：

```
adduser est
chown est:est -R ${HOME}/elasticsearch-8.3.2/
cd ${HOME}/elasticsearch-8.3.2/
su est
./bin/elasticsearch
```

#### Mac OS 上安装 elasticsearch 出现错误 `flood stage disk watermark [95%] exceeded on.... all indices on this node will be marked read-only`

elasticsearch 默认达到95％就全都设置只读，可以腾出一部分空间出来再启动，或者修改 `config/elasticsearch.pyml`。
```
cluster.routing.allocation.disk.threshold_enabled: false
```

#### nltk_data 加载失败的错误 `[nltk_data] Error loading punkt: [Errno 60] Operation timed out`

在命令行里面输入 python,然后输入下面的命令进行下载：

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

#### faiss 安装上了但还是显示找不到 faiss 怎么办？

推荐您使用 anaconda 进行单独安装，安装教程请参考[faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

```
# CPU-only version
conda install -c pytorch faiss-cpu

# GPU(+CPU) version
conda install -c pytorch faiss-gpu
```

#### 如何更换 pipelines 中预置的模型？

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

#### 运行 faiss examples 出现了错误：`sqlalchemy.exec.OperationalError: (sqlite3.OperationalError) too many SQL variables`

python 3.7版本引起的错误，修改如下代码：

```
# 增加batch_size参数，传入一个数值即可
document_store.update_embeddings(retriever, batch_size=256)
```

#### 运行后台程序出现了错误：`Exception: Failed loading pipeline component 'DocumentStore': RequestError(400, 'illegal_argument_exception', 'Mapper for [embedding] conflicts with existing mapper:\n\tCannot update parameter [dims] from [312] to [768]')`

以语义检索为例，这是因为模型的维度不对造成的，请检查一下 `elastic search`中的文本的向量的维度和`semantic_search.yaml`里面`DocumentStore`设置的维度`embedding_dim`是否一致，如果不一致，请重新使用`utils/offline_ann.py`构建索引。总之，请确保构建索引所用到的模型和`semantic_search.yaml`设置的模型是一致的。

#### 安装后出现错误：`cannot import name '_registerMatType' from 'cv2'`

opencv 版本不匹配的原因，可以对其进行升级到最新版本，保证 opencv 系列的版本一致。

```
pip install opencv-contrib-python --upgrade
pip install opencv-contrib-python-headless --upgrade
pip install opencv-python --upgrade
```

#### 安装运行出现 `RuntimeError: Can't load weights for 'rocketqa-zh-nano-query-encoder'`

rocketqa 模型2.3.7之后才添加，paddlenlp 版本需要升级：
```
pip install paddlenlp --upgrade
```

#### 安装出现问题 `The repository located at mirrors.aliyun.com is not a trusted or secure host and is being ignored.`

设置 pip 源为清华源，然后重新安装，可运行如下命令进行设置：

```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Elastic search 日志显示错误 `exception during geoip databases update`

需要编辑 config/elasticsearch.yml，在末尾添加：

```
ingest.geoip.downloader.enabled: false
```
如果是 Docker 启动，请添加如下的配置，然后运行：

```
docker run \
      -d \
      --name es02 \
      --net elastic \
      -p 9200:9200 \
      -e discovery.type=single-node \
      -e ES_JAVA_OPTS="-Xms256m -Xmx256m"\
      -e xpack.security.enabled=false \
      -e  ingest.geoip.downloader.enabled=false \
      -e cluster.routing.allocation.disk.threshold_enabled=false \
      -it \
      docker.elastic.co/elasticsearch/elasticsearch:8.3.3
```

#### Windows 出现运行前端报错`requests.exceptions.MissingSchema: Invalid URL 'None/query': No scheme supplied. Perhaps you meant http://None/query?`

环境变量没有生效，请检查一下环境变量，确保 PIPELINE_YAML_PATH 和 API_ENDPOINT 生效：

```
$env:PIPELINE_YAML_PATH='rest_api/pipeline/semantic_search.yaml'

$env:API_ENDPOINT='http://127.0.0.1:8891'
```

#### Windows 的 GPU 运行出现错误：`IndexError: index 4616429690595525704 is out of bounds for axis 0 with size 1`

paddle.nozero 算子出现异常，请退回到 PaddlePaddle 2.2.2版本，比如您使用的是 cuda 11.2，可以使用如下的命令：

```
python -m pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

#### 运行应用的时候出现错误 `assert d == self.d`

这是运行多个应用引起的，请在运行其他应用之前，删除现有的 db 文件：

```
rm -rf faiss_document_store.db
```

#### Windows 运行应用的时候出现了下面的错误：`RuntimeError: (NotFound) Cannot open file C:\Users\my_name/.paddleocr/whl\det\ch\ch_PP-OCRv3_det_infer/inference.pdmodel, please confirm whether the file is normal.`

这是 Windows 系统用户命名为中文的原因，详细解决方法参考 issue. [https://github.com/PaddlePaddle/PaddleNLP/issues/3242](https://github.com/PaddlePaddle/PaddleNLP/issues/3242)

#### 怎样从 GPU 切换到 CPU 上运行？

请在对应的所有`sh`文件里面加入下面的环境变量
```
export CUDA_VISIBLE_DEVICES=""
```

#### 运行 streamlit 前端程序出现错误：`AttributeError: module 'click' has no attribute 'get_os_args'`

click 版本过高导致：

```
pip install click==8.0
```

#### 怎么样新增最新的 pytorch 的检索模型

PaddleNLP-Pipelines 提供了可自动将 PyTorch 相关的权重转化为 Paddle 权重的接口，以 BAAI/bge-large-zh-v1.5为例，代码如下：

```python
from paddlenlp.transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5", from_hf_hub=True, convert_from_torch=True)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5', from_hf_hub=True)

model.save_pretrained("BAAI/bge-large-zh-v1.5")
tokenizer.save_pretrained("BAAI/bge-large-zh-v1.5")
```

然后在这里像这样注册一下即可使用：

```
"BAAI/bge-large-zh-v1.5": {
                "task_class": SentenceFeatureExtractionTask,
                "task_flag": "feature_extraction-BAAI/bge-large-zh-v1.5",
                "task_priority_path": "BAAI/bge-large-zh-v1.5",
            },
```

[taskflow 注册地址](https://github.com/PaddlePaddle/PaddleNLP/blob/b6dcb4e19efd85911b13a0fc587fef33578cfebf/paddlenlp/taskflow/taskflow.py#L680)

使用方式示例如下：

```
document_store = FAISSDocumentStore.load(your_index_name)
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="BAAI/bge-large-zh-v1.5",
    passage_embedding_model="BAAI/bge-large-zh-v1.5",
    output_emb_size=None,
    max_seq_len_query=64,
    max_seq_len_passage=256,
    batch_size=16,
    use_gpu=True,
    embed_title=False,
    pooling_mode="mean_tokens",
)
```

**注意** bge-m3的底座模型是 XLMRobertaModel，paddlenlp 没有实现，不推荐使用。
