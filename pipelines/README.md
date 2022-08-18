## PaddleNLP Pipelines：智能文本产线

PaddleNLP Pipelines 是一个端到端智能文本产线框架，面向 NLP **全场景**为用户提供**低门槛**构建强大**产品级系统**的能力。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168514868-1babe981-c675-4f89-9168-dd0a3eede315.gif" width="500">
</div>

## 智能文本产线特色
* **全场景支持**：依托灵活的插拔式组件产线化设计，支持各类 NLP 场景任务，包括：信息抽取、情感倾向分析、阅读理解、检索系统、问答系统、文本分类、文本生成等。

* **低门槛开发**：依托丰富的预置组件，像搭积木一样快速构建产品级系统，预置组件覆盖文档解析、数据处理、模型组网、预测部署、Web 服务、UI 界面等全流程系统功能。

* **高精度预测**：基于前沿的预训练模型、成熟的系统方案，可构建效果领先的产品级系统，如[智能文本产线库](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines#智能文本产线库)中预置的语义检索系统、阅读理解式智能问答系统等。

* **灵活可定制**：除深度兼容 PaddleNLP 模型组件外，还可嵌入飞桨生态下任意模型、[AI 开放平台算子](https://ai.baidu.com/)、其它开源项目如 Elasticsearch 等作为基础组件，快速扩展，从而实现任意复杂系统的灵活定制开发。

## 智能文本产线库

PaddleNLP Pipelines 智能文本产线库针对 NLP 部分高频场景开源了经过充分打磨的产品级系统，并会不断开放其它场景的产品级系统，用户可以基于智能文本产线库提供的系统能力快速开发出适配业务数据的产品。

* 快速搭建产品级[**语义检索**](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/semantic-search)系统：使用自然语言文本通过语义进行智能文档查询，而不是关键字匹配
* 快速搭建产品级[**智能问答**](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/question-answering)系统：用自然语言提问，即可获得精准答案片段
* 快速搭建产品级**多模态信息抽取**系统（即将开放，敬请期待）
* 快速搭建产品级 **FAQ 问答**系统（即将开放，敬请期待）

|  |  |
|-|-|
| :floppy_disk: [快速安装](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines#floppy_disk-安装) |安装 PaddleNLP Pipelines|
| :beginner: [快速体验](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines#beginner-快速体验) |基于 Pipelines 快速搭建语义检索/智能问答等产品系统|
| :man_office_worker: [用户案例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines#man_office_worker-用户案例) |各行业用户基于PaddleNLP Pipelinse 构建的产品案例|
| :mortar_board: [Tutorials](https://github.com/deepset-ai/haystack#mortar_board-tutorials) |像搭积木一样一步步构建产品级系统教程|
| :bar_chart: [Benchmarks](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/benchmarks) |针对各场景模型的性能、精度评测指标|
| :telescope: [Roadmap](https://github.com/PaddlePaddle/PaddleNLP) | PaddleNLP Pipelines 产品路线图|
| :newspaper: [技术博客](https://github.com/PaddlePaddle/PaddleNLP) | 阅读 PaddleNLP Pipelines 系列技术文章|
| :vulcan_salute: [社区交流](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines#vulcan_salute-社区交流) | [官方微信群](https://github.com/PaddlePaddle/PaddleNLP#社区交流), [GitHub Discussions](https://github.com/PaddlePaddle/PaddleNLP/discussions) |

## :floppy_disk: 安装
Note: 因为 pipelines 依赖较多, 安装耗时大概 10 分钟左右，安装过程中请请耐心等待。
### 环境依赖
- python >= 3.7.0
- paddlenlp >= 2.2.1
- paddlepaddle >=2.3
- CUDA Version: 10.2
- NVIDIA Driver Version: 440.64.00
### pip 安装
```
pip install --upgrade pipelines
```

### 源码安装
```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines
python setup.py install
```

## :beginner: 快速体验

您可以基于我们发布的 Docker 镜像一键搭建产品级语义检索系统进行快速体验。

### CPU 镜像
```
docker pull paddlepaddle/paddlenlp:2.4.0
docker run -d --name paddlenlp_pipelines --net host -ti paddlepaddle/paddlenlp:2.4.0
```
CPU 镜像下载大概耗时 20 分钟左右，容器启动成功后，通过浏览器访问 [http://127.0.0.1:8502](http://127.0.0.1:8502) 快速体验产品级语义检索服务。

### GPU 镜像
```
docker pull paddlepaddle/paddlenlp:2.4.0-gpu-cuda10.2-cudnn7
nvidia-docker run -d --name paddlenlp_pipelines_gpu --net host -ti paddlepaddle/paddlenlp:2.4.0
```
GPU 镜像下载大概耗时 3 分钟左右，容器启动成功后，通过浏览器访问 [http://127.0.0.1:8502](http://127.0.0.1:8502) 快速体验产品级语义检索服务。

## :man_office_worker: 用户案例

### 案例1: [寻规-工程规范搜索引擎](https://xungui365.com/)

[寻规](https://xungui365.com/)，是一款基于飞桨 PaddleNLP Pipelines 构建的建筑工程规范搜索引擎。大幅提升了工程设计人员工作效率。

#### 查询效率提升 36~60 倍

相比市面当前的工程规范查询系统/网站，平均查询到一条规范条款要 3\~5 分钟，而基于 PaddleNLP Pipelines 构建的[寻规](https://xungui365.com/)检索系统，平均查询到一条规范条款仅需 5 秒左右，搜索时间大幅缩短，仅规范查询效率方面就提升**36\~60** 倍！

#### 查询精度大幅提升

市面已有的工程规范查询系统解决方案一直延续着传统关键字词匹配的方式，依赖用户对对查询结果进行自行排序、筛选，甚至要再次人工查阅工程规范文件后，才能最终确认是否为想要查询的规范条款。传统规范查询系统至少需要进行 3~5 次查询才能找到用户想要的规范条款，而寻规系统是基于强大预训练模型构建起来的语义检索系统，针对 80% 的规范查询需求仅 **1 次查询** 就能精确命中查询意图，并返回查询条款的结果！

## :mortar_board: Tutorials
- Tutorial 1 - 语义检索 Pipeline: [AIStudio notebook]() | [Python](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/semantic-search/semantic_search_example.py)
- Tutorial 2 - 智能问答 Pipeline: [AIStudio notebook]() | [Python](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/question-answering/dense_qa_example.py)
## :vulcan_salute: 社区交流
微信扫描二维码并填写问卷之后，加入交流群与来自各行各业的小伙伴交流学习吧~
  <div align="center">
  <img src="https://user-images.githubusercontent.com/11793384/168411900-d9f3d777-99ab-4b5c-8cdc-ef747a48b864.jpg" width="150" height="150" />
  </div>


## :heart: Acknowledge
我们借鉴了 Deepset.ai [Haystack](https://github.com/deepset-ai/haystack) 优秀的框架设计，在此对[Haystack](https://github.com/deepset-ai/haystack)作者及其开源社区表示感谢。

We learn form the excellent framework design of Deepset.ai [Haystack](https://github.com/deepset-ai/haystack), and we would like to express our thanks to the authors of Haystack and their open source community.
