# WINDOWS环境下搭建端到端智能问答系统

以下的流程都是使用的Anaconda的环境进行的搭建，Anaconda安装好以后，进入 `Anaconda Powershell Prompt`，然后执行下面的流程。

## 1. 快速开始: 城市百科知识问答系统搭建

### 1.1 运行环境和安装说明

a. 依赖安装：
```bash
pip install -r requirements.txt
# 1) 安装 pipelines package
cd ${HOME}/PaddleNLP/applications/experimental/pipelines/
python setup.py install
```
### 1.2 数据说明
问答知识库数据是我们爬取了百度百科上对国内重点城市的百科介绍文档。我们将所有文档中的非结构化文本数据抽取出来， 按照段落切分后作为问答系统知识库的数据，一共包含 365 个城市的百科介绍文档、切分后共 1318 个段落。

### 1.3 一键体验问答系统
我们预置了搭建城市百科知识问答系统的代码示例，您可以通过如下命令快速体验问答系统的效果。


```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
# 设置 1 个空闲的 GPU 卡，此处假设 0 卡为空闲 GPU
export CUDA_VISIBLE_DEVICES=0
python examples/question-answering/dense_qa_example.py --device gpu
# 如果只有 CPU 机器，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
unset CUDA_VISIBLE_DEVICES
python examples/question-answering/dense_qa_example.py --device cpu
```

### 1.4 构建 Web 可视化问答系统

整个 Web 可视化问答系统主要包含 3 大组件: 1. 基于 ElasticSearch 的 ANN 服务 2. 基于 RestAPI 构建模型服务 3. 基于 Streamlit 构建 WebUI。接下来我们依次搭建这 3 个服务并串联构成可视化的问答系统

#### 1.4.1 启动 ANN 服务
1. 参考官方文档下载安装 [elasticsearch-8.3.2](https://www.elastic.co/cn/downloads/elasticsearch) 并解压。
2. 启动 ES 服务
把`xpack.security.enabled` 设置成false，如下：
```
xpack.security.enabled: false
```

然后直接双击bin目录下的elasticsearch.bat即可启动。

3. elasticsearch可视化工具Kibana（可选）
为了更好的对数据进行管理，可以使用Kibana可视化工具进行管理和分析，下载链接为[Kibana](https://www.elastic.co/cn/downloads/kibana)，下载完后解压，直接双击运行 `bin\kibana.bat`即可。

#### 1.4.2 文档数据写入 ANN 索引库
```
# 以百科城市数据为例建立 ANN 索引库
python utils/offline_ann.py --index_name baike_cities --doc_dir data/baike
```
运行成功后会输出如下的日志：
```
INFO - pipelines.utils.logger -  Logged parameters:
 {'processor': 'TextSimilarityProcessor', 'tokenizer': 'NoneType', 'max_seq_len': '0', 'dev_split': '0.1'}
INFO - pipelines.document_stores.elasticsearch -  Updating embeddings for all 1318 docs ...
Updating embeddings: 10000 Docs [00:16, 617.76 Docs/s]
```
运行结束后，可使用Kibana查看数据

#### 1.4.3 启动 RestAPI 模型服务
```bash
# 指定智能问答系统的Yaml配置文件
$env:PIPELINE_YAML_PATH='rest_api/pipeline/dense_qa.yaml'
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```

#### 1.4.4 启动 WebUI
```bash
# 配置模型服务地址
$env:API_ENDPOINT='http://127.0.0.1:8891'
# 在指定端口 8502 启动 WebUI
python -m streamlit run ui/webapp_question_answering.py --server.port 8502
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验城市百科知识问答系统服务了。

## FAQ

#### pip安装htbuilder包报错，`UnicodeDecodeError: 'gbk' codec can't decode byte....`

windows的默认字符gbk导致的，可以使用源码进行安装，源码已经进行了修复。

```
git clone https://github.com/tvst/htbuilder.git
cd htbuilder/
python setup install
```
