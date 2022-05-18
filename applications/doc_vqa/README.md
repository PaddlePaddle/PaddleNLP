# 文档视觉问答 (Document Visual Question Answering)

## 1. 项目说明

**文档视觉问答**是在问答领域的一项新的探索，与传统文本问答不同的是，可深度解析非结构化文档中排版复杂的图文/图表内容，直接定位问题答案。。 

本项目将针对**汽车说明书问答**实现视觉问答，即在用户提供汽车说明书后，针对用户提出的一些进行问题，从汽车说明书中寻找答案并进行回答。

#### 场景痛点
- 构建问题库方面：需投入大量人力整理常见问题库；固定的问题库难以覆盖灵活多变的提问；
- 售后客服方面： 需要配置大量客服人员；客服专业知识培训周期长；
- 用户方面： 没有耐心查阅说明书；打客服电话需要等待；

通过视觉文档问答能力，能够及时解答日常用车问题，亦赋能售后客服人员。


## 2. 安装说明

#### 环境要求

- paddlepaddle >= 2.2.0
- paddlenlp

安装相关问题可参考[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)和[PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/get_started/installation.html)文档。


## 3. 实验流程

简而言之，**汽车说明书问答**是针对用户提出的汽车相关问题，通过模型计算在汽车说明书中找出对应的答案，并返回给用户。


- **OCR模块**   
本项目提供了包含10张图片的汽车说明书，为方便后续处理，首先需要通过 PaddleOCR 对汽车说明书进行识别，记录汽车说明书上的文字和文字布局信息， 以方便后续使用计算机视觉和自然语言处理方面的技术进行问答任务。

- **排序模块**  
对于用户提出的问题，如果从所有的汽车说明书图片中去寻找答案会比较耗时且耗费资源。因此这里使用了一个排序模块，该模块将根据用户提出的问题对汽车说明书的不同图片进行打分排序，这样便可以获取和问题最相关的图片，并使用阅读理解模块在该问题上进行抽取答案。

- **阅读理解模块**  
获取排序模块输出的结果中评分最高的图片，然后将会使用 LayoutXLM 模型从该图片中去抽取用户提问的答案。在获取答案后，将会对答案在该图片中进行高亮显示并返回用户。

下面将具体介绍各个模块的训练和测试功能。

## 4. OCR模块

本实验提供的汽车说明书图片可点击[这里](https://paddlenlp.bj.bcebos.com/images/applications/automobile.tar.gz)进行下载，下载后解压放至 `./OCR_process/demo_pics` 目录下，然后通过如下命令，使用 PaddleOCR 进行图片文档解析。

```shell
cd OCR_process/
python3 ocr_process.py
cd ..
```

解析后的结果存放至 `./OCR_process/demo_ocr_res.json` 中。

## 5. 排序模块
本模块将基于使用 Dureader retrieval 数据集训练的排序模型 base_model 基础上，然后使用本项目提供的140条汽车说明书相关的训练样本进一步微调。

其中，base_model可点击[这里](https://paddlenlp.bj.bcebos.com/models/base_ranker.tar.gz)下载，将下载后的模型文件解压后可获得模型目录 `base_model`，将其放至 `./Rerank/checkpoints` 目录下；和汽车说明书相关的训练数据可点击[这里](https://paddlenlp.bj.bcebos.com/data/automobile_rerank_train.tsv)下载，下载后将其重命名为 `train.tsv` ，存放至 `./Rerank/data/` 目录下。

排序模型可使用如下代码进行训练：

```shell
cd Rerank
bash run_train.sh ./data/train.tsv ./checkpoints/base_model 50 1
cd ..
```
其中，参数依次为训练数据地址，base_model 地址，训练轮次，节点数。

在模型训练完成后，可将模型重命名为 `ranker` 存放至 `./checkpoints/` 目录下，接下来便可以使用如下命令，根据给定的汽车说明书相关问题，对汽车说明书的图片进行打分。代码如下：

```shell
cd Rerank
bash run_test.sh NFC咋开门
cd ..
```

其中，后一项为用户问题，命令执行完成后，分数文件将会保存至 `./Rerank/data/demo.score` 中。


## 5. 阅读理解模块
本模块将使用PaddleNLP提供的LayoutXLM模型进行视觉问答任务，主要是从排序模型输出评分最高的图片中，结合用户问题，抽取相应的答案。

本模块将基于使用Dureader VIS数据集训练的阅读理解模型base_model基础上，然后使用本项目提供的28条汽车说明书相关的训练样本进一步微调。

其中，base_model可点击[这里](https://paddlenlp.bj.bcebos.com/models/base_mrc.tar.gz)下载，将下载后的模型文件解压后可获得模型目录 `base_model`，将其放至 `./Extraction/checkpoints` 目录下；和汽车说明书相关的训练数据可点击[这里](https://paddlenlp.bj.bcebos.com/data/automobile_mrc_train.json)下载，下载后将其重命名为 `train.json`，存放至 `./Extraction/data/` 目录下。

排序模型可使用如下代码进行训练：

```shell
cd Rerank
bash run_train.sh
cd ..
```

在模型训练完成后，可将模型重命名为 `layoutxlm` 存放至 `./checkpoints/` 目录下，接下来便可以使用如下命令，根据给定的汽车说明书相关问题，从得分最高的汽车说明书图片中抽取答案。代码如下：

```shell
cd Rerank
bash run_test.sh NFC咋开门
cd ..
```

其中，后一项为用户问题，命令执行完成后，最终结果将会保存至 `./answer.png` 中。 

## 6. 全流程预测方式
本项目提供了全流程预测的功能，可通过如下命令进行一键式预测：

```shell
bash run_test.sh NFC咋开门
```
其中，后一项参数为用户问题，最终结果将会保存至 `./answer.png` 中。 

**备注**：在运行命令前，请确保已使用第3节介绍的命令对将原始的汽车说明书图片进行文档解析。