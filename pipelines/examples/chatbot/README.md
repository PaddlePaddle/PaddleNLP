# ChatFile

## 1. 场景概述

Chatfile的目的是让用户上传文件并与它进行交互，从而实现与文件的对话。它具备丰富落地场景，包括 1）内置翻译系统的Chatfile实现国内外商人之间的无障碍交流；2）基于本地知识库的Chatfile，实现智能客服问答功能，与客服人员实现良好互补；3）通过分析历史报告文件和新数据，Chatflie可以生成新的报告文件。

ChatFile带有一个内置的聊天机器人，它使用 文心一言 ErnieBot 技术。这使得它能够理解用户的输入并以自然的方式进行响应。此外，用户可以上传多个文件，ChatFile将文件存储在一个索引中。这样，用户就可以轻松地与多个文件进行交互和对话。ChatFile的使用非常简单，用户只需将文件上传到应用程序中，然后即可开始与它进行聊天。当用户发送消息时，聊天机器人会自动解析消息并做出相应的回答。

## 2. 产品功能介绍

本项目提供了低成本搭建端到端聊天机器人系统的能力。用户只需要处理好自己的业务数据，就可以使用本项目预置的聊天机器人系统模型(召回模型、排序模型、文心一言 ErnieBot)快速搭建一个针对自己业务数据的问答系统，并可以提供 Web 化产品服务。

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleNLP/assets/137043369/978b6d88-355f-4e63-91e0-874e9bcb8012" width="1000px">
</div>


### 2.1 系统特色

+ 端到端
    + 提供包括数据建库、模型服务部署、WebUI 可视化一整套端到端聊天机器人系统能力
    + 多源数据支持: 支持对 Txt、Word、PDF、Image、Markdown 多源数据进行解析、识别并写入 ANN 数据库
+ 效果好
    + 依托百度领先的NLP技术
    + 预置领先的深度学习模型

## 3. 快速开始: 快速搭建聊天机器人系统

以下是针对mac和linux的安装流程

### 3.1 运行环境和安装说明

本实验采用了以下的运行环境进行，详细说明如下，用户也可以在自己 GPU 硬件环境进行：

a. 软件环境：
- python >= 3.7.3
- paddlenlp >= 2.6
- paddlepaddle-gpu >=2.5
- CUDA Version: 11.2
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

```
# 下载pipelines源代码
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines
```
【注意】以下的所有的流程都只需要在`pipelines`根目录下进行，不需要跳转目录

### 3.2 数据说明
聊天机器人数据库的数据来自于[enterprise_responsibility_report](https://paddlenlp.bj.bcebos.com/applications/enterprise_responsibility_report.zip)，共包含 3个pdf文件和3个docx文件。如果有版权问题，请第一时间联系，并删除数据。

### 3.3 一键体验聊天机器人系统

#### 3.3.1 快速一键启动

我们预置了基于[enterprise_responsibility_report](https://paddlenlp.bj.bcebos.com/applications/enterprise_responsibility_report.zip)搭建聊天机器人的代码示例，您可以通过如下命令快速体验聊天机器人系统的效果

```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
# 设置 1 个空闲的 GPU 卡，此处假设 0 卡为空闲 GPU
export CUDA_VISIBLE_DEVICES=0
python examples/chatbot/chat_markdown_example.py --device gpu \
                                                 --search_engine faiss
# 如果只有 CPU 机器，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
unset CUDA_VISIBLE_DEVICES
python examples/chatbot/chat_markdown_example.py --device cpu \
                                                 --search_engine faiss
```
`chat_markdown_example.py`中`DensePassageRetriever`和`ErnieRanker`的模型介绍请参考[API介绍](../../API.md)


### 3.4 构建 Web 可视化聊天机器人系统

整个 Web 可视化聊天机器人系统主要包含 3 大组件: 1. 基于 ElasticSearch 的 ANN 服务 2. 基于 RestAPI 构建模型服务 3. 基于 Gradio 构建 WebUI，接下来我们依次搭建这 3 个服务并最终形成可视化的聊天机器人系统。

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
备注：ES 服务默认开启端口为 9200

#### 3.4.2 文档数据写入 ANN 索引库
```
# 以enterprise_responsibility_report数据集为例建立 ANN 索引库
python utils/offline_ann.py --index_name esg_example \
                            --doc_dir data/enterprise_responsibility_report \
                            --search_engine elastic \
                            --embed_title True \
                            --use_splitter  True \
                            --delete_index
```
可以使用下面的命令来查看数据：

```
# 打印几条数据
curl http://localhost:9200/esg_example/_search
```

参数含义说明
* `index_name`: 索引的名称
* `doc_dir`: txt文本数据的路径
* `host`: ANN索引引擎的IP地址
* `port`: ANN索引引擎的端口号
* `search_engine`: 选择的近似索引引擎elastic，milvus，默认elastic
* `delete_index`: 是否删除现有的索引和数据，用于清空es的数据，默认为false
* `embed_title`: 是否需要对标题建索引，默认为false，标题默认为文件名

删除索引也可以使用下面的命令：

```
curl -XDELETE http://localhost:9200/esg_example
```

#### 3.4.3 启动 RestAPI 模型服务
Note: chatfile.yaml中的api_key和secret_key需要自己补充
```bash
# 指定聊天机器人系统的Yaml配置文件
export PIPELINE_YAML_PATH=rest_api/pipeline/chatfile.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/chatbot/run_chatfile_server.sh
```
启动后可以使用curl命令验证是否成功运行：

```
curl -X POST -k http://localhost:8891/query -H 'Content-Type: application/json' -d '{"query": "光大证券的社会企业责任 ","params": {"Retriever": {"top_k": 5}, "Ranker":{"top_k": 5}}}'
```

更多API接口文档及其调用方式请参考链接[http://127.0.0.1:8891/docs](http://127.0.0.1:8891/docs)

#### 3.4.4 启动 WebUI
```bash
pip install gradio
# 配置模型服务地址
export API_ENDPOINT=http://127.0.0.1:8891
# 在指定端口 8502 启动 WebUI
python ui/webapp_chatfile_gradio.py --server.port 8502
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/chatbot/run_chatfile_wb.sh
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验聊天机器人服务了。

#### 3.4.5 数据更新

数据更新的方法有两种，第一种使用前面的 `utils/offline_ann_chatfile.py`进行数据更新，第二种是使用前端界面的文件上传（在界面的左侧）进行数据更新。对于第一种使用脚本的方式，可以使用多种文件更新数据，示例的文件更新建索引的命令如下，里面包含了图片（目前仅支持把图中所有的文字合并建立索引），docx（纯文本），txt，pdf，markdown五种格式的文件建索引：

```
python utils/offline_ann_chatfile.py --index_name esg_example \
                            --doc_dir data/enterprise_responsibility_report \
                            --port 9200 \
                            --search_engine elastic \
                            --delete_index
```

对于第二种使用界面的方式，支持txt，pdf，image，word，markdown的格式，以markdown格式的文件为例，程序会根据标题分段建立索引，示例数据如下(demo.md)：

```
**目录**

 1. [教学内容](#教学内容)
     1.1  [添加课节](#添加课节)
     1.2  [项目-添加项目](#项目-添加项目)
     1.3  [项目-编辑项目](#项目-编辑项目)
     1.4  [项目-发布项目](#项目-发布项目)
     1.5  [文档](#文档)
     1.6  [视频](#视频)
     1.7  [调序](#调序)
     1.8  [设置试看](#设置试看)
 3. [学习跟踪](#学习跟踪)
 4. [教学大纲](#教学大纲)
 ```
如果安装遇见问题可以查看[FAQ文档](../../FAQ.md)

## Reference
[1]Y. Sun et al., “[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/pdf/2107.02137.pdf),” arXiv:2107.02137 [cs], Jul. 2021, Accessed: Jan. 17, 2022. [Online]. Available: http://arxiv.org/abs/2107.02137

[2]Y. Qu et al., “[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2010.08191),” arXiv:2010.08191 [cs], May 2021, Accessed: Aug. 16, 2021. [Online]. Available: http://arxiv.org/abs/2010.08191


## Acknowledge

我们借鉴了 Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) 和 langchain(https://github.com/hwchase17/langchain)优秀的框架设计，在此对[Haystack](https://github.com/deepset-ai/haystack)和langchain(https://github.com/hwchase17/langchain)作者及其开源社区表示感谢。

We learn form the excellent framework design of Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) and langchain(https://github.com/hwchase17/langchain), and we would like to express our thanks to the authors of Haystack and langchain and their open source community.
