# 汽车说明书跨模态智能问答

## 1. 项目说明

**跨模态文档问答** 是跨模态的文档抽取任务，要求文档智能模型在文档中抽取能够回答文档相关问题的答案，需要模型在抽取和理解文档中文本信息的同时，还能充分利用文档的布局、字体、颜色等视觉信息，这比单一模态的信息抽取任务更具挑战性。

这种基于跨模态文档阅读理解技术的智能问答能力，可以深度解析非结构化文档中排版复杂的图文/图表内容，直接定位问题答案。

本项目将基于跨模态文档问答技术实现**汽车说明书问答系统**，该系统能够对用户提出的问题，自动从汽车说明书中寻找答案并进行回答。

如下图所示， 用户提出问题："如何更换前风窗玻璃的刮水片"，跨模态文档问答引擎将从库中寻找相关的文档，然后通过跨模态阅读理解模型抽取出相应的答案，并进行了高亮展示。

<center><img width="883" alt="image" src="https://user-images.githubusercontent.com/35913314/169781111-0734729d-3c7b-400d-8e92-e56548bb7dc5.png"></center>

通过使用汽车说明书问答系统，能够极大地解决传统汽车售后的压力：
- 用户：用户没有耐心查阅说明书，打客服电话需要等待
- 售后客服：需要配置大量客服人员，且客服专业知识培训周期长
- 构建问题库：需要投入大量人力整理常见问题库，并且固定的问题库难以覆盖灵活多变的提问

对于用户来说，汽车说明书问答系统能够支持通过车机助手/APP/小程序为用户提供即问即答的的功能。对于常见问题，用户不再需要查阅说明书，也无需打客服电话，从而缓解了人工客服的压力。

对于客服来讲，汽车说明书问答系统帮助客服人员快速定位答案，高效查阅文档，提高客服的专业水平，同时也能够缩短客服的培训周期。

## 2. 安装说明

#### 环境要求

- paddlepaddle >= 2.2.0
- paddlenlp >= 2.3.1
- paddleocr >= 2.5

安装相关问题可参考[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)和[PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/get_started/installation.html)文档。


## 3. 整体流程

汽车说明书问答系统针对用户提出的汽车使用相关问题，智能化地在汽车说明书中找出对应答案，并返回给用户。本项目提供的汽车说明书问答系统的使用流程如下图所示。本项目提供的汽车说明书问答系统主要包括 3 个模块：OCR处理模块、排序模块和跨模态阅读理解模块。

在使用汽车说明书问答模型回答问题之前，需要先使用PaddleOCR对离线提供的汽车说明书文档进解析，并将解析结果保存下来，以备后续排序模块使用。

对于用户提问的问题，首先会被传入排序模块，排序模块会针对该问题对解析的文档进行排序打分，其结果将会被传入跨模态阅读理解模块。阅读理解模块将从分数最高的说明书文档中，抽取用户问题的答案，并返回给用户。

<center><img width="864" alt="image" src="https://user-images.githubusercontent.com/35913314/170222662-c438ff2a-a1df-44e5-8a83-f14dc0814b9d.png"></center>

下面将具体介绍各个模块的功能。

## 4. OCR处理模块

本项目提供了包含10张图片的汽车说明书，为方便后续处理，首先需要通过 PaddleOCR 对汽车说明书进行识别，记录汽车说明书上的文字和文字布局信息， 以方便后续使用计算机视觉和自然语言处理方面的技术进行问答任务。

本项目提供的汽车说明书图片可点击[这里](https://paddlenlp.bj.bcebos.com/images/applications/automobile.tar.gz)进行下载，下载后解压放至 `./OCR_process/demo_pics` 目录下，然后通过如下命令，使用 PaddleOCR 对图片进行解析。

```shell
cd OCR_process/
python3 ocr_process.py
cd ..
```

解析后的结果存放至 `./OCR_process/demo_ocr_res.json` 中。

## 5. 排序模块
对于用户提出的问题，如果从所有的汽车说明书图片中去寻找答案会比较耗时且耗费资源。因此这里使用了一个基于[RocketQA](https://arxiv.org/pdf/2010.08191.pdf)的排序模块，该模块将根据用户提出的问题对汽车说明书的不同图片进行打分排序，这样便可以获取和问题最相关的图片，并使用跨模态阅读理解模块在该问题上进行抽取答案。

本项目提供了140条汽车说明书相关的训练样本，用于排序模型的训练， 同时也提供了一个基于RocketQA的预先训练好的基线模型 base_model。 本模块可以使用 base_model 在汽车说明书训练样本上进一步微调。

其中，汽车说明书的训练集可点击[这里](https://paddlenlp.bj.bcebos.com/data/automobile_rerank_train.tsv) 进行下载，下载后将其重命名为 `train.tsv` ，存放至 `./Rerank/data/` 目录下。

同时，base_model 是 [Dureader retrieval](https://arxiv.org/abs/2203.10232) 数据集训练的排序模型， 可点击[这里](https://paddlenlp.bj.bcebos.com/models/base_ranker.tar.gz) 进行下载，解压后可获得包含模型的目录 `base_model`，将其放至 `./Rerank/checkpoints` 目录下。

可使用如下代码进行训练：

```shell
cd Rerank
bash run_train.sh ./data/train.tsv ./checkpoints/base_model 50 1
cd ..
```
其中，参数依次为训练数据地址，base_model 地址，训练轮次，节点数。

在模型训练完成后，可将模型重命名为 `ranker` 存放至 `./checkpoints/` 目录下，接下来便可以使用如下命令，根据给定的汽车说明书相关问题，对汽车说明书的图片进行打分。代码如下：

```shell
cd Rerank
bash run_test.sh 后备箱怎么开
cd ..
```

其中，后一项为用户问题，命令执行完成后，分数文件将会保存至 `./Rerank/data/demo.score` 中。


## 6. 跨模态阅读理解模块
本项目首先获取排序模块输出的结果中评分最高的图片，然后将会使用跨模态的语言模型 LayoutXLM 从该图片中去抽取用户提问的答案。在获取答案后，将会对答案在该图片中进行高亮显示并返回用户。

本项目提供了28条汽车说明书相关的训练样本，用于跨模态阅读理解模型的训练， 同时也提供了一个预先训练好的基线模型 base_model。 本模块可以使用 base_model 在汽车说明书训练样本上进一步微调，增强模型对汽车说明书领域的理解。

其中，汽车说明书的阅读理解训练集可点击[这里](https://paddlenlp.bj.bcebos.com/data/automobile_mrc_train.json) 进行下载，下载后将其重命名为 `train.json`，存放至 `./Extraction/data/` 目录下。

同时，base_model 是 [Dureader VIS](https://aclanthology.org/2022.findings-acl.105.pdf) 数据集训练的跨模态阅读理解模型， 可点击[这里](https://paddlenlp.bj.bcebos.com/models/base_mrc.tar.gz) 进行下载，解压后可获得包含模型的目录 `base_model`，将其放至 `./Extraction/checkpoints` 目录下。

可使用如下代码进行训练：

```shell
cd Extraction
bash run_train.sh
cd ..
```

在模型训练完成后，可将模型重命名为 `layoutxlm` 存放至 `./checkpoints/` 目录下，接下来便可以使用如下命令，根据给定的汽车说明书相关问题，从得分最高的汽车说明书图片中抽取答案。代码如下：

```shell
cd Extraction
bash run_test.sh 后备箱怎么开
cd ..
```

其中，后一项为用户问题，命令执行完成后，最终结果将会保存至 `./answer.png` 中。

## 7. 全流程预测
本项目提供了全流程预测的功能，可通过如下命令进行一键式预测：

```shell
bash run_test.sh 后备箱怎么开
```

其中，后一项参数为用户问题，最终结果将会保存至 `./answer.png` 中。

**备注**：在运行命令前，请确保已使用第4节介绍的命令对原始汽车说明书图片完成了文档解析。


下图展示了用户提问的三个问题："后备箱怎么开"，"钥匙怎么充电" 和 "NFC解锁注意事项"， 可以看到，本项目的汽车说明书问答系统能够精准地找到答案并进行高亮显示。

<center><img src="https://user-images.githubusercontent.com/35913314/169012902-1a42bd14-976f-4da8-b5b5-d8e7352b68df.png"/></center>
