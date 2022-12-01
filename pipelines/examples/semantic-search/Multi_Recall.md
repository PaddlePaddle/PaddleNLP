# 端到端两路召回语义检索系统

## 1. 概述

多路召回是指采用不同的策略、特征或者简单的模型，分别召回一部分候选集合，然后把这些候选集混合在一起供后续的排序模型进行重排，也可以定制自己的重排序的规则等等。本项目使用关键字和语义检索两路召回的检索系统，系统的架构如下，用户输入的Query会分别通过关键字召回BMRetriever（Okapi BM 25算法，Elasticsearch默认使用的相关度评分算法，是基于词频和文档频率和文档长度相关性来计算相关度），语义向量检索召回DenseRetriever（使用RocketQA抽取向量，然后比较向量之间相似度）后得到候选集，然后通过JoinResults进行结果聚合，最后通过通用的Ranker模块得到重排序的结果返回给用户。

<div align="center">
    <img src="https://user-images.githubusercontent.com/12107462/204423532-90f62781-5f81-4b6d-9f94-741416ae3fcb.png" width="500px">
</div>

## 2. 产品功能介绍

本项目提供了低成本搭建端到端两路召回语义检索系统的能力。用户只需要处理好自己的业务数据，就可以使用本项目预置的两路召回语义检索系统模型(召回模型、排序模型)快速搭建一个针对自己业务数据的检索系统，并可以提供 Web 化产品服务。

<div align="center">
    <img src="https://user-images.githubusercontent.com/12107462/204435911-0ba1cb9f-cb56-4bcd-9f64-63ff173826d6.png" width="500px">
</div>

## 3. 快速开始: 快速搭建两路召回语义检索系统

### 3.1 运行环境和安装说明

本实验采用了以下的运行环境进行，详细说明如下，用户也可以在自己 GPU 硬件环境进行：

a. 软件环境：
- python >= 3.7.0
- paddlenlp >= 2.4.3
- paddlepaddle-gpu >=2.3
- CUDA Version: 10.2
- NVIDIA Driver Version: 440.64.00
- Ubuntu 16.04.6 LTS (Docker)

b. 硬件环境：

- NVIDIA Tesla V100 16GB x4卡
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

c. 依赖安装：
首先需要安装PaddlePaddle，PaddlePaddle的安装请参考文档[官方安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)，然后安装下面的依赖：
```bash
# pip 一键安装
pip install --upgrade paddle-pipelines -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或者源码进行安装最新版本
cd ${HOME}/PaddleNLP/pipelines/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py install
```

【注意】

- Windows的安装复杂一点，教程请参考：[Windows视频安装教程](https://www.bilibili.com/video/BV1DY4y1M7HE/?zw)
- 以下的所有的流程都只需要在`pipelines`根目录下进行，不需要跳转目录

### 3.2 数据说明

语义检索数据库的数据来自于[DuReader-Robust数据集](https://github.com/baidu/DuReader/tree/master/DuReader-Robust)，共包含 46972 个段落文本，并选取了其中验证集1417条段落文本来搭建语义检索系统。

### 3.3 一键体验语义检索系统

#### 3.3.1 启动 ANN 服务
1. 参考官方文档下载安装 [elasticsearch-8.3.2](https://www.elastic.co/cn/downloads/elasticsearch) 并解压。
2. 启动 ES 服务
首先修改`config/elasticsearch.yml`的配置：
```
xpack.security.enabled: false
```
然后启动：
```bash
./bin/elasticsearch
```
3. 检查确保 ES 服务启动成功
```bash
curl http://localhost:9200/_aliases?pretty=true
```
备注：ES 服务默认开启端口为 9200

#### 3.3.2 快速一键启动

我们预置了基于[DuReader-Robust数据集](https://github.com/baidu/DuReader/tree/master/DuReader-Robust)搭建语义检索系统的代码示例，您可以通过如下命令快速体验语义检索系统的效果
```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
# 设置 1 个空闲的 GPU 卡，此处假设 0 卡为空闲 GPU
export CUDA_VISIBLE_DEVICES=0
python examples/semantic-search/multi_recall_semantic_search_example.py --device gpu \
                                                          --search_engine elastic
# 如果只有 CPU 机器，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
unset CUDA_VISIBLE_DEVICES
python examples/semantic-search/multi_recall_semantic_search_example.py --device cpu \
                                                          --search_engine elastic
```
`multi_recall_semantic_search_example.py`中`DensePassageRetriever`和`ErnieRanker`的模型介绍请参考[API介绍](../../API.md)

参数含义说明
* `device`: 设备名称，cpu/gpu，默认为gpu
* `index_name`: 索引的名称
* `search_engine`: 选择的近似索引引擎elastic，milvus，默认elastic
* `max_seq_len_query`: query的最大长度，默认是64
* `max_seq_len_passage`: passage的最大长度，默认是384
* `retriever_batch_size`: 召回模型一次处理的数据的数量
* `query_embedding_model`: query模型的名称，默认为rocketqa-zh-nano-query-encoder
* `passage_embedding_model`: 段落模型的名称，默认为rocketqa-zh-nano-para-encoder
* `params_path`: Neural Search的召回模型的名称，默认为
* `embedding_dim`: 模型抽取的向量的维度,默认为312，为rocketqa-zh-nano-query-encoder的向量维度
* `host`: ANN索引引擎的IP地址
* `port`: ANN索引引擎的端口号
* `bm_topk`: 关键字召回节点BM25Retriever的召回数量
* `dense_topk`: 语义向量召回节点DensePassageRetriever的召回数量
* `rank_topk`: 排序模型节点ErnieRanker的排序过滤数量

### 3.4 构建 Web 可视化语义检索系统

整个 Web 可视化语义检索系统主要包含 3 大组件: 1. 基于 ElasticSearch 的 ANN 服务 2. 基于 RestAPI 构建模型服务 3. 基于 Streamlit 构建 WebUI，搭建ANN服务请参考1.3.1节，接下来我们依次搭建后台和前端两个服务。

#### 3.4.1 文档数据写入 ANN 索引库
```
# 以DuReader-Robust 数据集为例建立 ANN 索引库
python utils/offline_ann.py --index_name dureader_nano_query_encoder \
                            --doc_dir data/dureader_dev \
                            --search_engine elastic \
                            --delete_index
```
可以使用下面的命令来查看数据：

```
# 打印几条数据
curl http://localhost:9200/dureader_nano_query_encoder/_search
```

参数含义说明
* `index_name`: 索引的名称
* `doc_dir`: txt文本数据的路径
* `host`: ANN索引引擎的IP地址
* `port`: ANN索引引擎的端口号
* `search_engine`: 选择的近似索引引擎elastic，milvus，默认elastic
* `delete_index`: 是否删除现有的索引和数据，用于清空es的数据，默认为false

#### 3.4.2 启动 RestAPI 模型服务
```bash
# 指定语义检索系统的Yaml配置文件
export PIPELINE_YAML_PATH=rest_api/pipeline/multi_recall_semantic_search.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```
启动后可以使用curl命令验证是否成功运行：

```
curl -X POST -k http://localhost:8891/query -H 'Content-Type: application/json' -d '{"query": "衡量酒水的价格的因素有哪些?","params": {"BMRetriever": {"top_k": 10}, "DenseRetriever": {"top_k": 10}, "Ranker":{"top_k": 3}}}'
```
#### 3.4.3 启动 WebUI
```bash
# 配置模型服务地址
export API_ENDPOINT=http://127.0.0.1:8891
# 在指定端口 8502 启动 WebUI
python -m streamlit run ui/webapp_multi_recall_semantic_search.py --server.port 8502
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验语义检索系统服务了。

#### 3.4.4 数据更新

数据更新的方法有两种，第一种使用前面的 `utils/offline_ann.py`进行数据更新，第二种是使用前端界面的文件上传（在界面的左侧）进行数据更新。对于第一种使用脚本的方式，可以使用多种文件更新数据，示例的文件更新建索引的命令如下，里面包含了图片（目前仅支持把图中所有的文字合并建立索引），docx（支持图文，需要按照空行进行划分段落），txt（需要按照空行划分段落）三种格式的文件建索引：

```
python utils/offline_ann.py --index_name dureader_robust_query_encoder \
                            --doc_dir data/file_example \
                            --port 9200 \
                            --search_engine elastic \
                            --delete_index
```

对于第二种使用界面的方式，支持txt，pdf，image，word的格式，以txt格式的文件为例，每段文本需要使用空行隔开，程序会根据空行进行分段建立索引，示例数据如下(demo.txt)：

```
兴证策略认为，最恐慌的时候已经过去，未来一个月市场迎来阶段性修复窗口。

从海外市场表现看，
对俄乌冲突的恐慌情绪已显著释放，
海外权益市场也从单边下跌转入双向波动。

长期，继续聚焦科技创新的五大方向。1)新能源(新能源汽车、光伏、风电、特高压等)，2)新一代信息通信技术(人工智能、大数据、云计算、5G等)，3)高端制造(智能数控机床、机器人、先进轨交装备等)，4)生物医药(创新药、CXO、医疗器械和诊断设备等)，5)军工(导弹设备、军工电子元器件、空间站、航天飞机等)。
```
如果安装遇见问题可以查看[FAQ文档](../../FAQ.md)

## Reference
[1]Y. Sun et al., “[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/pdf/2107.02137.pdf),” arXiv:2107.02137 [cs], Jul. 2021, Accessed: Jan. 17, 2022. [Online]. Available: http://arxiv.org/abs/2107.02137

[2]Y. Qu et al., “[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2010.08191),” arXiv:2010.08191 [cs], May 2021, Accessed: Aug. 16, 2021. [Online]. Available: http://arxiv.org/abs/2010.08191

[3]H. Tang, H. Li, J. Liu, Y. Hong, H. Wu, and H. Wang, “[DuReader_robust: A Chinese Dataset Towards Evaluating Robustness and Generalization of Machine Reading Comprehension in Real-World Applications](https://arxiv.org/pdf/2004.11142.pdf).” arXiv, Jul. 21, 2021. Accessed: May 15, 2022. [Online]. Available: http://arxiv.org/abs/2004.11142

## Acknowledge

我们借鉴了 Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) 优秀的框架设计，在此对[Haystack](https://github.com/deepset-ai/haystack)作者及其开源社区表示感谢。

We learn form the excellent framework design of Deepset.ai [Haystack](https://github.com/deepset-ai/haystack), and we would like to express our thanks to the authors of Haystack and their open source community.
