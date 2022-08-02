# 端到端语义检索系统

## 1. 场景概述

检索系统存在于我们日常使用的很多产品中，比如商品搜索系统、学术文献检索系等等，本方案提供了检索系统完整实现。限定场景是用户通过输入检索词 Query，快速在海量数据中查找相似文档。

所谓语义检索（也称基于向量的检索），是指检索系统不再拘泥于用户 Query 字面本身，而是能精准捕捉到用户 Query 后面的真正意图并以此来搜索，从而更准确地向用户返回最符合的结果。通过使用最先进的语义索引模型找到文本的向量表示，在高维向量空间中对它们进行索引，并度量查询向量与索引文档的相似程度，从而解决了关键词索引带来的缺陷。

例如下面两组文本 Pair，如果基于关键词去计算相似度，两组的相似度是相同的。而从实际语义上看，第一组相似度高于第二组。

```
车头如何放置车牌    前牌照怎么装
车头如何放置车牌    后牌照怎么装
```

语义检索系统的关键就在于，采用语义而非关键词方式进行召回，达到更精准、更广泛得召回相似结果的目的。

## 2. 产品功能介绍

本项目提供了低成本搭建端到端语义检索系统的能力。用户只需要处理好自己的业务数据，就可以使用本项目预置的语义检索系统模型(召回模型、排序模型)快速搭建一个针对自己业务数据的问答系统，并可以提供 Web 化产品服务。

### 2.1 系统特色

+ 端到端
    + 提供包括数据建库、模型服务部署、WebUI 可视化一整套端到端语义检索系统能力
    + 多源数据支持: 支持对 Txt、Word、PDF、Image 多源数据进行解析、识别并写入 ANN 数据库
+ 效果好
    + 依托百度领先的NLP技术，包括[ERNIE](https://github.com/PaddlePaddle/ERNIE)语义理解技术与[RocketQA](https://github.com/PaddlePaddle/RocketQA)开放域问答技术
    + 预置领先的深度学习模型

## 3. 快速开始: 快速搭建语义检索系统

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
语义检索数据库的数据来自于[DuReader-Robust数据集](https://github.com/baidu/DuReader/tree/master/DuReader-Robust)，共包含 46972 个段落文本，并选取了其中验证集1417条段落文本来搭建语义检索系统。

### 3.3 一键体验语义检索系统
我们预置了基于[DuReader-Robust数据集](https://github.com/baidu/DuReader/tree/master/DuReader-Robust)搭建语义检索系统的代码示例，您可以通过如下命令快速体验语义检索系统的效果
```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
# 设置 1 个空闲的 GPU 卡，此处假设 0 卡为空闲 GPU
export CUDA_VISIBLE_DEVICES=0
python examples/semantic-search/semantic_search_example.py --device gpu
# 如果只有 CPU 机器，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
unset CUDA_VISIBLE_DEVICES
python examples/semantic-search/semantic_search_example.py --device cpu
```

### 3.4 构建 Web 可视化语义检索系统

整个 Web 可视化语义检索系统主要包含 3 大组件: 1. 基于 ElasticSearch 的 ANN 服务 2. 基于 RestAPI 构建模型服务 3. 基于 Streamlit 构建 WebUI，接下来我们依次搭建这 3 个服务并最终形成可视化的语义检索系统。

#### 3.4.1 启动 ANN 服务
1. 参考官方文档下载安装 [elasticsearch-8.3.2](https://www.elastic.co/cn/downloads/elasticsearch) 并解压。
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
# 以DuReader-Robust 数据集为例建立 ANN 索引库
python utils/offline_ann.py --index_name dureader_robust_query_encoder \
                            --doc_dir data/dureader_dev
```

参数含义说明
* `index_name`: 索引的名称
* `doc_dir`: txt文本数据的路径
* `delete_index`: 是否删除现有的索引和数据，用于清空es的数据，默认为false

#### 3.4.3 启动 RestAPI 模型服务
```bash
# 指定语义检索系统的Yaml配置文件
export PIPELINE_YAML_PATH=rest_api/pipeline/semantic_search.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```
Linux 用户推荐采用 Shell 脚本来启动服务：：

```bash
sh scripts/run_search_server.sh
```

#### 3.4.4 启动 WebUI
```bash
# 配置模型服务地址
export API_ENDPOINT=http://127.0.0.1:8891
# 在指定端口 8502 启动 WebUI
python -m streamlit run ui/webapp_semantic_search.py --server.port 8502
```
Linux 用户推荐采用 Shell 脚本来启动服务：：

```bash
sh scripts/run_search_web.sh
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验语义检索系统服务了。

#### 3.4.5 数据更新

数据更新的方法有两种，第一种使用前面的 `utils/offline_ann.py`进行数据更新，另一种是使用前端界面的文件上传进行数据更新，支持txt，pdf，image，word的格式，以txt格式的文件为例，每段文本需要使用空行隔开，程序会根据空行进行分段建立索引，示例数据如下(demo.txt)：

```
兴证策略认为，最恐慌的时候已经过去，未来一个月市场迎来阶段性修复窗口。

从海外市场表现看，
对俄乌冲突的恐慌情绪已显著释放，
海外权益市场也从单边下跌转入双向波动。

长期，继续聚焦科技创新的五大方向。1)新能源(新能源汽车、光伏、风电、特高压等)，2)新一代信息通信技术(人工智能、大数据、云计算、5G等)，3)高端制造(智能数控机床、机器人、先进轨交装备等)，4)生物医药(创新药、CXO、医疗器械和诊断设备等)，5)军工(导弹设备、军工电子元器件、空间站、航天飞机等)。
```

## FAQ

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

## Reference
[1]Y. Sun et al., “[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/pdf/2107.02137.pdf),” arXiv:2107.02137 [cs], Jul. 2021, Accessed: Jan. 17, 2022. [Online]. Available: http://arxiv.org/abs/2107.02137

[2]Y. Qu et al., “[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2010.08191),” arXiv:2010.08191 [cs], May 2021, Accessed: Aug. 16, 2021. [Online]. Available: http://arxiv.org/abs/2010.08191

[3]H. Tang, H. Li, J. Liu, Y. Hong, H. Wu, and H. Wang, “[DuReader_robust: A Chinese Dataset Towards Evaluating Robustness and Generalization of Machine Reading Comprehension in Real-World Applications](https://arxiv.org/pdf/2004.11142.pdf).” arXiv, Jul. 21, 2021. Accessed: May 15, 2022. [Online]. Available: http://arxiv.org/abs/2004.11142

## Acknowledge

我们借鉴了 Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) 优秀的框架设计，在此对[Haystack](https://github.com/deepset-ai/haystack)作者及其开源社区表示感谢。

We learn form the excellent framework design of Deepset.ai [Haystack](https://github.com/deepset-ai/haystack), and we would like to express our thanks to the authors of Haystack and their open source community.
