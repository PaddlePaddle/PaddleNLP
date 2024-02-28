# 无监督智能检索问答系统

## 1. 场景概述

智能问答（QA）是获取信息和只是的更直接、更高效的方式之一，传统的信息检索系统只能找到相关的文档，而问答系统能够直接找到精准的答案，极大的节省了人们获取信息的时间。问答系统中最关键的挑战之一是标记数据的稀缺性，这是因为对目标领域获取问答对或常见问答对（FAQ）的成本很高，需要消耗大量的人力和时间。由于上述制约，这导致检索式问答系统落地困难，解决此问题的一种方法是依据上下文或大量非结构化文本自动生成的QA问答对。

本项目，即无监督智能检索问答(问答对自动生成智能检索式问答)，基于PaddleNLP问题生成、UIE、检索式问答，支持以非结构化文本形式为上下文自动生成QA问答对，生成的问答对语料可以通过无监督的方式构建检索式问答系统。

<div align="center">
    <img src="https://user-images.githubusercontent.com/20476674/199488926-c64d3f4e-8117-475f-afe6-b02088105d09.gif" >
</div>

若开发者已有FAQ语料，请参考FAQ检索式问答。
## 2. 产品功能介绍

本项目提供了低成本搭建问答对自动生成智能检索问答系统的能力。开发者只需要提供非结构化的纯文本，就可以使用本项目预制的问答对生成模块生成大量的问答对，并基于此快速搭建一个针对自己业务的检索问答系统，并可以提供Web可视化产品服务。Web可视化产品服务支持问答检索、在线问答对生成，在线文件上传和解析，在线索引库更新等功能，用户也可根据需要自行调整。

### 2.1 系统特色
+ 低成本
    + 可通过自动生成的方式快速大量合成QA语料，大大降低人力成本
    + 可控性好，合成语料和语义检索解耦合，可以人工筛查和删除合成的问答对，也可以添加人工标注的问答对
+ 端到端
    + 提供包括问答语料生成、索引库构建、模型服务部署、WebUI可视化一整套端到端智能问答系统能力
    + 支持对Txt、Word、PDF、Image多源数据上传，同时支持离线、在线QA语料生成和ANN数据库更新
+ 效果好
    + 可通过自动问答对生成提升问答对语料覆盖度，缓解中长尾问题覆盖较少的问题
    + 依托百度领先的NLP技术，预置效果领先的深度学习模型

## 3. 快速开始: 快速搭建无监督智能检索问答系统

以下是针对mac和linux的搭建流程。

### 3.1 运行环境和安装说明

本项目采用了以下的运行环境进行，详细说明如下，用户也可以在自己的GPU硬件环境进行：

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
首先需要安装PaddlePaddle，PaddlePaddle的安装请参考文档[官方安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。然后需要安装paddle-pipelines依赖，使用pip安装命令如下：
```bash
# pip一键安装
pip install --upgrade paddle-pipelines -i https://pypi.tuna.tsinghua.edu.cn/simple
```
或者进入pipelines目录下，针对源码进行安装：
```bash
# 源码进行安装
cd PaddleNLP/pipelines/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py install
```
**【注意】** 以下的所有的流程都只需要在`pipelines`根目录下进行，不需要跳转目录

### 3.2 数据说明
我们以提供的纯文本文件[source_file.txt](https://paddlenlp.bj.bcebos.com/applications/unsupervised_qa/source_file.txt)为例，系统将每一条都视为一个上下文并基于此生成多个问答对，并基于此构建索引库，该文件可直接下载放入./data，开发者也可以使用自己的文件。


### 3.3 一键体验无监督智能检索问答系统

开发者可以通过如下命令快速体验无监督智能检索问答系统的效果，系统将自动根据提供的纯文本文件构建问答对语料库，并基于生成的问答对语料库构造检索数据库。
我们建议在GPU环境下运行本示例，运行速度较快，运行命令如下：
```bash
# GPU环境下运行示例
# 设置1个空闲的GPU卡，此处假设0卡为空闲GPU
export CUDA_VISIBLE_DEVICES=0
python examples/unsupervised-question-answering/unsupervised_question_answering_example.py --device gpu --source_file data/source_file.txt --doc_dir data/my_data --index_name faiss_index --retriever_batch_size 16
```
关键参数释义如下：
- `device`: 使用的设备，默认为'gpu'，可选择['cpu', 'gpu']。
- `source_file`: 源文件路径，指定该路径将自动为其生成问答对至`doc_dir`。
- `doc_dir`: 生成的问答对语料保存的位置，系统将根据该位置自动构建检索数据库，默认为'data/my_data'。
- `index_name`: FAISS的ANN索引名称，默认为'faiss_index'。
- `retriever_batch_size`: 构建ANN索引时的批量大小，默认为16。

如果只有CPU机器，可以通过--device参数指定cpu即可, 运行耗时较长，运行命令如下：
```bash
# CPU环境下运行示例
unset CUDA_VISIBLE_DEVICES
python examples/unsupervised-question-answering/unsupervised_question_answering_example.py --device cpu --source_file data/source_file.txt --doc_dir data/my_data
```
**【注意】**  `unsupervised_question_answering_example.py`中`DensePassageRetriever`和`ErnieRanker`的模型介绍请参考[API介绍](../../API.md)

### 3.4 构建Web可视化无监督智能检索问答系统

整个Web可视化无监督智能检索问答系统主要包含3大组件:
1. 基于ElasticSearch的ANN服务搭建在线索引库
2. 基于RestAPI构建模型后端服务
3. 基于Streamlit构建前端WebUI

接下来我们依次搭建这些个服务，得到可视化、可交互的无监督智能检索问答系统。


#### 3.4.1 离线生成问答对语料
执行以下命令将自动根据提供的纯文本文件离线构建问答对语料库：
```bash
# GPU环境下运行示例
# 设置1个空闲的GPU卡，此处假设0卡为空闲GPU
export CUDA_VISIBLE_DEVICES=0
python examples/unsupervised-question-answering/offline_question_answer_pairs_generation.py --device gpu --source_file data/source_file.txt --doc_dir data/my_data
```
关键参数释义如下：
- `device`: 使用的设备，默认为'gpu'，可选择['cpu', 'gpu']。
- `source_file`: 源文件路径，指定该路径将自动为其生成问答对至`doc_dir`。
- `doc_dir`: 生成的问答对语料保存的位置，系统将根据该位置自动构建检索数据库，默认为'data/my_data'。


如果只有CPU机器，可以通过--device参数指定cpu即可, 运行耗时较长，运行命令如下：
```bash
# CPU环境下运行示例
unset CUDA_VISIBLE_DEVICES
python examples/unsupervised-question-answering/offline_question_answer_pairs_generation.py --device cpu --source_file data/source_file.txt --doc_dir data/my_data
```

#### 3.4.2 启动ElasticSearch ANN服务
1. 参考官方文档下载安装 [elasticsearch-8.3.2](https://www.elastic.co/cn/downloads/elasticsearch) 并解压。
2. 启动ElasticSearch服务。

首先修改`config/elasticsearch.yml`的配置：
```
xpack.security.enabled: false
```
然后启动elasticsearch：
```bash
./bin/elasticsearch
```
3. 检查确保ElasticSearch服务启动成功。

执行以下命令，如果ElasticSearch里面没有数据，结果会输出为空，即{ }。
```bash
curl http://localhost:9200/_aliases?pretty=true
```

备注：ElasticSearch服务默认开启端口为 9200

#### 3.4.3 ANN索引库构建
执行以下命令建立ANN索引库：
```
python utils/offline_ann.py --index_name my_data \
                            --doc_dir data/my_data \
                            --split_answers \
                            --delete_index
```
参数含义说明
* `index_name`: 索引的名称
* `doc_dir`: txt文本数据的路径
* `host`: Elasticsearch的IP地址
* `port`: Elasticsearch的端口号
* `split_answers`: 是否切分每一行的数据为query和answer两部分
* `delete_index`: 是否删除现有的索引和数据，用于清空es的数据，默认为false

执行以下命令打印几条数据，检测ANN索引库是否构建成功：
```
curl http://localhost:9200/my_data/_search
```
如果索引库正常会输出类似如下的结果：
```
{"took":1,"timed_out":false,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0},"hits":{"total":{"value":5,"relation":"eq"},"max_score":1.0,"hits":[{"_index":"my_data","_id":"fb308738f2767626d72282f5a35402e5","_score":1.0,"_source":{"content":......
```

#### 3.4.4 启动RestAPI模型后端
```bash
export CUDA_VISIBLE_DEVICES=0
# 指定无监督智能检索问答系统的Yaml配置文件
export PIPELINE_YAML_PATH=rest_api/pipeline/unsupervised_qa.yaml
# 使用端口号8896启动模型服务
python rest_api/application.py 8896
```
Linux 用户推荐采用Shell脚本来启动服务：

```bash
sh examples/unsupervised-question-answering/run_unsupervised_question_answering_server.sh
```
启动后可以使用curl命令验证是否成功运行：
```
curl -X POST -k http://localhost:8896/query -H 'Content-Type: application/json' -d '{"query": "企业如何办理养老保险?","params": {"Retriever": {"top_k": 5}, "Ranker":{"top_k": 5}}}'
```
如果成功运行，则会返回结果。

#### 3.4.5 启动Streamlit WebUI前端
```bash
pip install streamlit==1.11.1
# 配置模型服务地址
export API_ENDPOINT=http://127.0.0.1:8896
# 在指定端口 8502 启动 WebUI
python -m streamlit run ui/webapp_unsupervised_question_answering.py --server.port 8508
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/unsupervised-question-answering/run_unsupervised_question_answering_web.sh
```

到这里您就可以打开浏览器访问地址 http://127.0.0.1:8508 体验无监督智能检索问答系统服务了。



**【注意】** 如果安装遇见问题可以查看[FAQ文档](../../FAQ.md)

## Reference
[1]Y. Sun et al., “[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/pdf/2107.02137.pdf),” arXiv:2107.02137 [cs], Jul. 2021, Accessed: Jan. 17, 2022. [Online]. Available: http://arxiv.org/abs/2107.02137

[2]Y. Qu et al., “[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2010.08191),” arXiv:2010.08191 [cs], May 2021, Accessed: Aug. 16, 2021. [Online]. Available: http://arxiv.org/abs/2010.08191

[3]H. Tang, H. Li, J. Liu, Y. Hong, H. Wu, and H. Wang, “[DuReader_robust: A Chinese Dataset Towards Evaluating Robustness and Generalization of Machine Reading Comprehension in Real-World Applications](https://arxiv.org/pdf/2004.11142.pdf).” arXiv, Jul. 21, 2021. Accessed: May 15, 2022. [Online]. Available: http://arxiv.org/abs/2004.11142

[4]Li, Wei, et al. "Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning." arXiv preprint arXiv:2012.15409 (2020).

## Acknowledge

我们借鉴了 Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) 优秀的框架设计，在此对[Haystack](https://github.com/deepset-ai/haystack)作者及其开源社区表示感谢。

We learn form the excellent framework design of Deepset.ai [Haystack](https://github.com/deepset-ai/haystack), and we would like to express our thanks to the authors of Haystack and their open source community.
