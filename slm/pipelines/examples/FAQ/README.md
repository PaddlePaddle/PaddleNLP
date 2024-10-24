# 端到端 FAQ 智能问答系统

## 1. 场景概述

智能问答是获取信息和只是的更直接、更高效的方式之一，传统的信息检索方法只能找到相关的文档，而智能问答能够直接找到精准的答案，极大的节省了人们获取信息的时间。问答技按照技术划分分为基于阅读理解抽取式的问答和检索式的问答，本项目属于检索式问答，即检索匹配库里面的高频的问题，然后把高频问题对应的答案返回给用户。问答的领域用途很广，比如搜索引擎，小度音响等智能硬件，政府，金融，银行，电信，电商领域的智能客服，聊天机器人等。

## 2. 产品功能介绍

本项目提供了低成本搭建端到端 FAQ 智能问答的能力。用户只需要处理好自己的业务数据，就可以使用本项目预置的检索系统模型(召回模型、排序模型)快速搭建一个针对自己业务数据的问答系统，并可以提供 Web 化产品服务。以下是使用预置模型的教程，如果用户想训练并接入自己训练的模型，模型训练可以参考[FAQ Finance](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/applications/question_answering/supervised_qa/faq_finance)，模型的接入流程参考 Pipelines 语义检索中 Neural Search 模型接入流程即可。

<div align="center">
    <img src="https://user-images.githubusercontent.com/12107462/190307449-38135678-f259-4483-ac0f-2fa3ae4be97f.gif" width="500px">
</div>

### 2.1 系统特色

+ 端到端
    + 提供包括数据建库、模型服务部署、WebUI 可视化一整套端到端 FAQ 智能问答系统能力
    + 多源数据支持: 支持对 Txt、Word、PDF、Image 多源数据进行解析、识别并写入 ANN 数据库
+ 效果好
    + 依托百度领先的 NLP 技术，包括[ERNIE](https://github.com/PaddlePaddle/ERNIE)语义理解技术与[RocketQA](https://github.com/PaddlePaddle/RocketQA)开放域问答技术
    + 预置领先的深度学习模型

## 3. 快速开始: 快速搭建 FAQ 智能问答系统

以下是针对 mac 和 linux 的安装流程

### 3.1 运行环境和安装说明

本实验采用了以下的运行环境进行，详细说明如下，用户也可以在自己 GPU 硬件环境进行：

a. 软件环境：
- python >= 3.7.3
- paddlenlp >= 2.4.0
- paddlepaddle-gpu >=2.3
- CUDA Version: 10.2
- NVIDIA Driver Version: 440.64.00
- Ubuntu 16.04.6 LTS (Docker)

b. 硬件环境：

- NVIDIA Tesla V100 16GB x4卡
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

c. 依赖安装：
首先需要安装 PaddlePaddle，PaddlePaddle 的安装请参考文档[官方安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)，然后安装下面的依赖：
```bash
# pip 一键安装
pip install --upgrade paddle-pipelines -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或者源码进行安装最新版本
cd ${HOME}/PaddleNLP/pipelines/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py install
```

```
# 下载pipelines源代码
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines
```

【注意】以下的所有的流程都只需要在`pipelines`根目录下进行，不需要跳转目录

### 3.2 数据说明
FAQ 智能问答数据库的数据来自于[8000 多条保险行业问答数据](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/baoxianzhidao/intro.ipynb)，共包含 8000 多个问答对，并过滤选取了其中3788条问答对来搭建 FAQ 智能问答系统。

### 3.3 一键体验 FAQ 智能问答系统

#### 3.3.1 快速一键启动

我们预置了基于[ 8000 多条保险行业问答数据](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/baoxianzhidao/intro.ipynb)搭建保险 FAQ 智能问答的代码示例，您可以通过如下命令快速体验智能问答的效果
```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
# 设置 1 个空闲的 GPU 卡，此处假设 0 卡为空闲 GPU
export CUDA_VISIBLE_DEVICES=0
python examples/FAQ/dense_faq_example.py --device gpu
# 如果只有 CPU 机器，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
unset CUDA_VISIBLE_DEVICES
python examples/FAQ/dense_faq_example.py --device cpu
```
`dense_faq_example.py`中`DensePassageRetriever`和`ErnieRanker`的模型介绍请参考[API 介绍](../../API.md)

### 3.4 构建 Web 可视化 FAQ 智能问答

整个 Web 可视化 FAQ 智能问答主要包含 3 大组件: 1. 基于 ElasticSearch 的 ANN 服务 2. 基于 RestAPI 构建模型服务 3. 基于 Streamlit 构建 WebUI，接下来我们依次搭建这 3 个服务并最终形成可视化的 FAQ 智能问答。

#### 3.4.1 启动 ANN 服务
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
如果 elasticsearch 里面没有数据，结果会输出为空：

```
{ }
```

备注：ES 服务默认开启端口为 9200

#### 3.4.2 文档数据写入 ANN 索引库
```
# 以保险数据集为例建立 ANN 索引库
python utils/offline_ann.py --index_name insurance \
                            --doc_dir data/insurance \
                            --split_answers \
                            --delete_index \
                            --query_embedding_model rocketqa-zh-nano-query-encoder \
                            --passage_embedding_model rocketqa-zh-nano-para-encoder \
                            --embedding_dim 312

```
参数含义说明
* `index_name`: 索引的名称
* `doc_dir`: txt 文本数据的路径
* `host`: Elasticsearch 的 IP 地址
* `port`: Elasticsearch 的端口号
* `split_answers`: 是否切分每一行的数据为 query 和 answer 两部分
* `delete_index`: 是否删除现有的索引和数据，用于清空 es 的数据，默认为 false

```
# 打印几条数据
curl http://localhost:9200/insurance/_search
```
会输出如下的示例结果：

```
{"took":2,"timed_out":false,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0},"hits":{"total":{"value":3776,"relation":"eq"},"max_score":1.0,"hits":[{"_index":"insurance","_id":"5bfb94d6da02e52ce5b778bc4876f91f","_score":1.0,"_source":{"content":"如果你想和你最好的朋友一起去席，推荐一个旅游保险","content_type":"text","name":"qa_pair.txt","answer":"您好，去西*旅游保险比较重要。慧择网推出huts保险，包括境内游险种......
```

#### 3.4.3 启动 RestAPI 模型服务

**注意** dense_faq.yaml 里面的检索模型需要与前面使用 offline_ann.py 建库的时候使用的检索模型一致

```bash
# 指定FAQ智能问答系统的Yaml配置文件
export PIPELINE_YAML_PATH=rest_api/pipeline/dense_faq.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/FAQ/run_faq_server.sh
```
启动后可以使用 curl 命令验证是否成功运行：

```
curl -X POST -k http://localhost:8891/query -H 'Content-Type: application/json' -d '{"query": "企业如何办理养老保险?","params": {"Retriever": {"top_k": 5}, "Ranker":{"top_k": 5}}}'
```
如果成功运行，则会返回结果。

更多 API 接口文档及其调用方式请参考链接[http://127.0.0.1:8891/docs](http://127.0.0.1:8891/docs)

#### 3.4.4 启动 WebUI
```bash
pip install streamlit==1.11.1
# 配置模型服务地址
export API_ENDPOINT=http://127.0.0.1:8891
# 在指定端口 8502 启动 WebUI
python -m streamlit run ui/webapp_faq.py --server.port 8502
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/FAQ/run_faq_web.sh
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验 FAQ 智能问答系统服务了。

#### 3.4.5 数据更新

数据更新有两种，第一种是使用界面的文件上传，支持 txt，word，必须是 Question 和 Answer 两列，用\t 进行分隔，另外 word 格式的数据，数据间用空行分隔开，txt 格式按正常回车键分隔即可。第二种是使用前面的 `utils/offline_ann.py`进行数据更新，示例数据如下(demo.txt)：

```
我想买保险，可以买哪些？    人身保障的保险，主要可以分为四大险种——即意外险、重疾险、医疗险和寿险。意外险——像过马路被车撞、被开水烫伤等等意外，意外险皆可赔付。医疗险——花多少钱报销多少钱，一般建议买百万医疗险。重疾险——得了重疾，按比例一次性赔付你约定保额。寿险——身故即赔。
选保险产品时，保险公司很重要吗？    重要，但不是第一重要，也不是最重要。产品应该是优先于公司的，毕竟产品的保障才是最直接和我们的利益挂钩的。在保险产品的保障差不多的情况下，知名度更高的保险公司会更好。
```
word 示例数据：

```
我想买保险，可以买哪些？    可以买哪些？人身保障的保险，主要可以分为四大险种——即意外险、重疾险、医疗险和寿险。意外险——像过马路被车撞、被开水烫伤等等意外，意外险皆可赔付。医疗险——花多少钱报销多少钱，一般建议买百万医疗险。重疾险——得了重疾，按比例一次性赔付你约定保额。寿险——身故即赔。

选保险产品时，保险公司很重要吗？    重要，但不是第一重要，也不是最重要。产品应该是优先于公司的，毕竟产品的保障才是最直接和我们的利益挂钩的。在保险产品的保障差不多的情况下，知名度更高的保险公司会更好。

```


如果安装遇见问题可以查看[FAQ 文档](../../FAQ.md)

## Reference
[1]Y. Sun et al., “[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/pdf/2107.02137.pdf),” arXiv:2107.02137 [cs], Jul. 2021, Accessed: Jan. 17, 2022. [Online]. Available: http://arxiv.org/abs/2107.02137

[2]Y. Qu et al., “[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2010.08191),” arXiv:2010.08191 [cs], May 2021, Accessed: Aug. 16, 2021. [Online]. Available: http://arxiv.org/abs/2010.08191

[3]H. Tang, H. Li, J. Liu, Y. Hong, H. Wu, and H. Wang, “[DuReader_robust: A Chinese Dataset Towards Evaluating Robustness and Generalization of Machine Reading Comprehension in Real-World Applications](https://arxiv.org/pdf/2004.11142.pdf).” arXiv, Jul. 21, 2021. Accessed: May 15, 2022. [Online]. Available: http://arxiv.org/abs/2004.11142

## Acknowledge

我们借鉴了 Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) 优秀的框架设计，在此对[Haystack](https://github.com/deepset-ai/haystack)作者及其开源社区表示感谢。

We learn form the excellent framework design of Deepset.ai [Haystack](https://github.com/deepset-ai/haystack), and we would like to express our thanks to the authors of Haystack and their open source community.
