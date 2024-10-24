# 端到端文图跨模态检索系统

## 1. 场景概述

文图跨模态检索系统目的是通过文字找到最符合描述的图片。传统的方案是用标签和图片的关键字进行匹配，而跨模态检索真正的实现了文本语义和图片语义内容的匹配，这种检索方式更符合人类的逻辑判断，是一种真正意义上的端到端人工智能。文图应用目前可以广泛应用于电商搜索，安防视频，图像检索，抖音等小视频，旅游 app 应用搜索。有助于提升效率和搜索体验。另外还有一些潜在的领域，比如司法的互联网调查取证，侵权检测，数据增强，文案匹配，各种互联网 logo，肖像，风景，海报等图片网站的检索，医药等专业领域的文图搜索等。

## 2. 产品功能介绍

本项目提供了低成本搭建端到端文图跨模态检索系统的能力。用户只需要处理好自己的业务数据，就可以使用本项目预置的文图跨模态检索系统模型快速搭建一个针对自己业务数据的跨模态检索系统，并可以提供 Web 化产品服务。

<div align="center">
    <img src="https://user-images.githubusercontent.com/12107462/216578818-f194cf9f-3d7f-4139-a173-dc9366e52e97.png" width="500px">
</div>

以下是文搜图的系统搭建流程，如果用户需要图搜文的应用，请参考[图搜文系统搭建流程](./IMAGE_TO_TEXT_SEARCH.md)

### 2.1 系统特色

+ 端到端
    + 提供包括数据建库、模型服务部署、WebUI 可视化一整套端到端文图跨模态检索系统能力
    + 依托百度领先的 NLP 技术，包括[ERNIE](https://github.com/PaddlePaddle/ERNIE)语义理解技术，[ERNIE-ViL 2.0](https://arxiv.org/abs/2209.15270)跨模态检索能力
    + 预置领先的深度学习模型

## 3. 快速开始: 快速搭建文图跨模态检索系统


### 3.1 运行环境和安装说明

本实验采用了以下的运行环境进行，详细说明如下，用户也可以在自己 GPU 硬件环境进行：

a. 软件环境：
- python >= 3.7.3
- paddlenlp >= 2.5.0
- paddlepaddle-gpu >=2.4.1
- CUDA Version: 11.2
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
文图跨模态检索数据库的数据来自于[Noah-Wukong 数据集](https://wukong-dataset.github.io/wukong-dataset/index.html)，并选取了测试集中3056张图片来搭建文图跨模态检索系统。

### 3.3 一键体验文图跨模态检索系统

#### 3.3.1 快速一键启动

我们预置了基于[Noah-Wukong 数据集](https://wukong-dataset.github.io/wukong-dataset/index.html)搭建文图跨模态检索系统的代码示例，您可以通过如下命令快速体验文图跨模态检索系统的效果
```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
# 设置 1 个空闲的 GPU 卡，此处假设 0 卡为空闲 GPU
export CUDA_VISIBLE_DEVICES=0
python examples/image_text_retrieval/text_to_image_retrieval_example.py --device gpu
# 如果只有 CPU 机器，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
unset CUDA_VISIBLE_DEVICES
python examples/image_text_retrieval/text_to_image_retrieval_example.py --device cpu
```


### 3.4 构建 Web 可视化文图跨模态检索系统

整个 Web 可视化文图跨模态检索系统主要包含 3 大组件: 1. 基于 ElasticSearch 的 ANN 服务 2. 基于 RestAPI 构建模型服务 3. 基于 Streamlit 构建 WebUI，接下来我们依次搭建这 3 个服务并最终形成可视化的文图跨模态检索系统。

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
# 以DuReader-Robust 数据集为例建立 ANN 索引库
python utils/offline_ann_mm.py --index_name wukong_test \
                            --doc_dir data/wukong_test \
                            --search_engine elastic \
                            --delete_index
```
可以使用下面的命令来查看数据：

```
# 打印几条数据
curl http://localhost:9200/wukong_test/_search
```

参数含义说明
* `index_name`: 索引的名称
* `doc_dir`: txt 文本数据的路径
* `host`: ANN 索引引擎的 IP 地址
* `port`: ANN 索引引擎的端口号
* `search_engine`: 选择的近似索引引擎 elastic，milvus，默认 elastic
* `delete_index`: 是否删除现有的索引和数据，用于清空 es 的数据，默认为 false

删除索引也可以使用下面的命令：

```
curl -XDELETE http://localhost:9200/wukong_test
```

#### 3.4.3 启动 RestAPI 模型服务
```bash
# 指定文图跨模态检索系统的Yaml配置文件
export PIPELINE_YAML_PATH=rest_api/pipeline/text_to_image_retrieval.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/image_text_retrieval/run_search_server.sh
```
启动后可以使用 curl 命令验证是否成功运行：

```
curl -X POST -k http://localhost:8891/query -H 'Content-Type: application/json' -d '{"query": "云南普者黑现纯白色⒌蒂莲","params": {"Retriever": {"top_k": 5}}}'
```

更多 API 接口文档及其调用方式请参考链接[http://127.0.0.1:8891/docs](http://127.0.0.1:8891/docs)

#### 3.4.4 启动 WebUI
```bash
pip install gradio
# 配置模型服务地址
export API_ENDPOINT=http://127.0.0.1:8891
# 在指定端口 8502 启动 WebUI
python ui/webapp_text_to_image_retrieval.py --server.port 8502
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/image_text_retrieval/run_search_web.sh
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验文图跨模态检索系统服务了。

#### 3.4.5 数据更新

数据更新使用前面的 `utils/offline_ann_mm.py`进行数据更新，把图片放在特定目录，然后传入该目录即可：

```
python utils/offline_ann_mm.py --index_name wukong_test \
                            --doc_dir data/wukong_test \
                            --port 9200 \
                            --search_engine elastic \
                            --delete_index
```


如果安装遇见问题可以查看[FAQ 文档](../../FAQ.md)

## Reference
[1]Y. Sun et al., “[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/pdf/2107.02137.pdf),” arXiv:2107.02137 [cs], Jul. 2021, Accessed: Jan. 17, 2022. [Online]. Available: http://arxiv.org/abs/2107.02137

[2]Shan, Bin, et al. "[ERNIE-ViL 2.0: Multi-view Contrastive Learning for Image-Text Pre-training](https://arxiv.org/abs/2209.15270)." arXiv preprint arXiv:2209.15270 (2022).

## Acknowledge

我们借鉴了 Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) 优秀的框架设计，在此对[Haystack](https://github.com/deepset-ai/haystack)作者及其开源社区表示感谢。

We learn form the excellent framework design of Deepset.ai [Haystack](https://github.com/deepset-ai/haystack), and we would like to express our thanks to the authors of Haystack and their open source community.
