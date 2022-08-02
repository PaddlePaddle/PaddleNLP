# 端到端智能问答系统

## 1. 场景概述

问答系统是信息检索系统的一种高级形式，通过对用户输入的问题进行理解，然后从知识库中寻找答案，并直接反馈给用户。在很多具体场景下我们都会对问答系统有需求。

在日常生活中，用户会经常碰到很多复杂的规章制度、规则条款。例如：火车提前多久可以免费退票？、在北京工作几年可以办理居住证？

在平时工作中，员工也会面对公司多种多样的政策。比如：商业保险理赔需要什么材料？打车报销的具体流程是什么？

这些情况下，传统做法需要仔细阅读政策文件、规章制度、或者咨询相关工作人员才能得到答案，费时费力。现在我们则可以针对这类常见的业务场景快速搭建一套智能问答系统，高效地回答用户的常见问题，提升用户体验的同时，也降低了客服人员的工作负荷及企业成本。

## 2. 产品功能介绍

本项目提供了低成本搭建端到端问答系统的能力。用户只需要处理好自己的业务数据，就可以使用本项目预置的问答系统模型(召回模型、排序模型、阅读理解模型)快速搭建一个针对自己业务数据的问答系统，并可以提供基于[Streamlit](https://streamlit.io/) 的 Web 可视化服务。

### 2.1 系统特色

+ 端到端
    + 提供包括数据建库、模型服务部署、WebUI 可视化一整套端到端问答系统能力
    + 多源数据支持: 支持对 Txt、Word、PDF、Image 多源数据进行解析、识别并写入 ANN 数据库
+ 效果好
    + 依托百度领先的NLP技术，包括[ERNIE](https://github.com/PaddlePaddle/ERNIE)语义理解技术与[RocketQA](https://github.com/PaddlePaddle/RocketQA)开放域问答技术
    + 预置领先的深度学习模型

## 3. 快速开始: 城市百科知识问答系统搭建

以下是针对mac和linux的安装流程，windows的安装和使用流程请参考[windows](./Install_windows.md)

### 3.1 运行环境和安装说明

本实验采用了以下的运行环境进行，详细说明如下，用户也可以在自己 GPU 硬件环境进行：

a. 软件环境：
- python >= 3.7.0
- paddlenlp >= 2.2.1
- paddlepaddle-gpu >=2.3
- CUDA Version: 10.2
- NVIDIA Driver Version: 440.64.00
- Ubuntu 16.04.6 LTS (Docker)

b. 硬件环境：

- NVIDIA Tesla V100 16GB x4卡
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

c. 依赖安装：
```bash
pip install -r requirements.txt
# 1) 安装 pipelines package
cd ${HOME}/PaddleNLP/applications/experimental/pipelines/
python setup.py install
# 2) 安装 RestAPI 相关依赖
python ./rest_api/setup.py install
# 3) 安装 Streamlit WebUI 相关依赖
python ./ui/setup.py install
```
### 3.2 数据说明
问答知识库数据是我们爬取了百度百科上对国内重点城市的百科介绍文档。我们将所有文档中的非结构化文本数据抽取出来， 按照段落切分后作为问答系统知识库的数据，一共包含 365 个城市的百科介绍文档、切分后共 1318 个段落。

### 3.3 一键体验问答系统
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

### 3.4 构建 Web 可视化问答系统

整个 Web 可视化问答系统主要包含 3 大组件: 1. 基于 ElasticSearch 的 ANN 服务 2. 基于 RestAPI 构建模型服务 3. 基于 Streamlit 构建 WebUI。接下来我们依次搭建这 3 个服务并串联构成可视化的问答系统

#### 3.4.1 启动 ANN 服务
1. 参考官方文档下载安装 [elasticsearch-8.1.2](https://www.elastic.co/cn/downloads/elasticsearch) 并解压。
2. 启动 ES 服务
```bash
./bin/elasticsearch
```
3. 检查确保 ES 服务启动成功
```bash
curl http://localhost:9200/_aliases?pretty=true
```
备注：ES 服务默认开启端口为 9200

#### 3.4.2 文档数据写入 ANN 索引库
```
# 以百科城市数据为例建立 ANN 索引库
python utils/offline_ann.py --index_name baike_cities \
                            --doc_dir data/baike \
                            --delete_index
```

参数含义说明
* `index_name`: 索引的名称
* `doc_dir`: txt文本数据的路径
* `delete_index`: 是否删除现有的索引和数据，用于清空es的数据，默认为false

运行成功后会输出如下的日志：
```
INFO - pipelines.utils.logger -  Logged parameters:
 {'processor': 'TextSimilarityProcessor', 'tokenizer': 'NoneType', 'max_seq_len': '0', 'dev_split': '0.1'}
INFO - pipelines.document_stores.elasticsearch -  Updating embeddings for all 1318 docs ...
Updating embeddings: 10000 Docs [00:16, 617.76 Docs/s]
```
使用如下的命令可以查看是否插入成功：
```
curl -XGET http://localhost:9200/baike_cities/_count
```

运行结束后会有如下的输出：
```
{"count":1318,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0}}
```

#### 3.4.3 启动 RestAPI 模型服务
```bash
# 指定智能问答系统的Yaml配置文件
export PIPELINE_YAML_PATH=rest_api/pipeline/dense_qa.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh scripts/run_qa_server.sh
```
启动后可以使用curl命令验证是否成功运行：

```
curl -X POST -k http://localhost:8891/query -H 'Content-Type: application/json' -d '{"query": "北京市有多少个行政区？","params": {"Retriever": {"top_k": 5}, "Ranker":{"top_k": 5}}}'
```

#### 3.4.4 启动 WebUI
```bash
# 配置模型服务地址
export API_ENDPOINT=http://127.0.0.1:8891
# 在指定端口 8502 启动 WebUI
python -m streamlit run ui/webapp_question_answering.py --server.port 8502
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh scripts/run_qa_web.sh
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验城市百科知识问答系统服务了。

#### 3.4.5 数据更新

数据更新的方法有两种，第一种使用前面的 `utils/offline_ann.py`进行数据更新，另一种是使用前端界面的文件上传进行数据更新，支持txt，pdf，image，word的格式，以txt格式的文件为例，每段文本需要使用空行隔开，程序会根据空行进行分段建立索引，示例数据如下(demo.txt)：

```
兴证策略认为，最恐慌的时候已经过去，未来一个月市场迎来阶段性修复窗口。

从海外市场表现看，
对俄乌冲突的恐慌情绪已显著释放，
海外权益市场也从单边下跌转入双向波动。

长期，继续聚焦科技创新的五大方向。1)新能源(新能源汽车、光伏、风电、特高压等)，2)新一代信息通信技术(人工智能、大数据、云计算、5G等)，3)高端制造(智能数控机床、机器人、先进轨交装备等)，4)生物医药(创新药、CXO、医疗器械和诊断设备等)，5)军工(导弹设备、军工电子元器件、空间站、航天飞机等)。
```

## Reference
[1]Y. Sun et al., “[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/pdf/2107.02137.pdf),” arXiv:2107.02137 [cs], Jul. 2021, Accessed: Jan. 17, 2022. [Online]. Available: http://arxiv.org/abs/2107.02137

[2]Y. Qu et al., “[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2010.08191),” arXiv:2010.08191 [cs], May 2021, Accessed: Aug. 16, 2021. [Online]. Available: http://arxiv.org/abs/2010.08191

[3]H. Tang, H. Li, J. Liu, Y. Hong, H. Wu, and H. Wang, “[DuReader_robust: A Chinese Dataset Towards Evaluating Robustness and Generalization of Machine Reading Comprehension in Real-World Applications](https://arxiv.org/pdf/2004.11142.pdf).” arXiv, Jul. 21, 2021. Accessed: May 15, 2022. [Online]. Available: http://arxiv.org/abs/2004.11142

## Acknowledge

我们借鉴了 Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) 优秀的框架设计，在此对[Haystack](https://github.com/deepset-ai/haystack)作者及其开源社区表示感谢。

We learn form the excellent framework design of Deepset.ai [Haystack](https://github.com/deepset-ai/haystack), and we would like to express our thanks to the authors of Haystack and their open source community.
