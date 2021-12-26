# 情感分析

## 1. 场景概述

情感分析旨在对带有情感色彩的主观性文本进行分析、处理、归纳和推理，其广泛应用于消费决策、舆情分析、个性化推荐等领域，具有很高的商业价值。

依托百度领先的情感分析技术，食行生鲜自动生成菜品评论标签辅助用户购买，并指导运营采购部门调整选品和促销策略；房天下向购房者和开发商直观展示楼盘的用户口碑情况，并对好评楼盘置顶推荐；国美搭建服务智能化评分系统，客服运营成本减少40%，负面反馈处理率100%。

情感分析相关的任务有语句级情感分析、评论对象抽取、观点抽取等等。一般来讲，被人们所熟知的情感分析任务是语句级别的情感分析，该任务是在宏观上去分析整句话的感情色彩，其粒度可能相对比较粗。

因为在人们进行评论的时候，往往针对某一对象进行多个维度的评论，对每个维度的评论可能也会褒贬不一，因此针对维度级别的情感分析在真实的场景中会更加实用，同时更能给到企业用户或商家更加具体的建议。例如这句关于薯片的评论。

> 这个薯片味道真的太好了，口感很脆，只是包装很一般。

可以看到，顾客在口感、包装和味道 三个维度上对薯片进行了评价，顾客在味道和口感两个方面给出了好评，但是在包装上给出了负面的评价。只有通过这种比较细粒度的分析，商家才能更有针对性的发现问题，进而改进自己的产品或服务。

本项目基于这样的考量，提供出一套完整的细粒度情感分析解决方案，期望能够在评论语句中评论维度的粒度级别进行情感分析。

## 2. 产品功能介绍

### 2.1 系统特色
为了降低技术门槛，方便开发者共享效果领先的情感分析技术，PaddleNLP本次开源的情感分析系统，具备三大亮点：

- 覆盖任务全
    - 集成句子级情感分类、评论观点抽取、属性级情感分类等多种情感分析能力，并开源模型，且打通模型训练、评估、预测部署全流程。
- 效果领先
    - 集成百度研发的基于情感知识增强的预训练模型SKEP，为各类情感分析任务提供统一且强大的情感语义表示能力。
- 预测性能强
    - 针对预训练模型预测效率低的问题，开源小模型PP-MiniLM，量化优化策略，预测性能大幅提升。

### 2.2 架构&功能

针对以上提到的细粒度情感分析，我们提出的解决方案如下图所示。整个情感分析的过程包含两个阶段，依次是评论维度和观点抽取模型，细粒度情感分类模型。对于给定的一段文本，首先基于前者抽取出文本语句中潜在的评论维度以及该维度相应的评论观点，然后将评论维度、观点以及原始文本进行拼接，传给细粒度情感分类模型以识别出该评论维度的情感色彩。

这里需要提到的是，由于目前市面上的大多数模型是基于通用语料训练出来的，这些模型可能并不会对情感信息那么敏感。基于这样的考量，本项目使用了百度自研的 SKEP 预训练模型，其在预训练阶段便设计了多种情感信息相关的预训练目标进行训练。作为一种情感专属的模型，其更适合用来做上边提到的评论维度和观点抽取任务，以及细粒度情感分类任务。

另外，本项目使用的是 Large 版的 SKEP 模型，考虑到企业用户在线上部署时会考虑到模型预测效率，所以本项目专门提供了一个通用版的小模型 [PP-MiniLM](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm) 以及一套量化策略，用户可以使用相应情感数据集对 PP-MiniLM 进行微调，然后进行量化，以达到更快的使用效率。

<div align="center">
    <img src="./imgs/sentiment_system.png" />
    <p>图1 情感分析系统图<p/>
</div>

## 3. 细粒度情感分析实践

以下是本项目运行的完整目录结构以及说明：

```shell
.
├── extraction                         # 评价维度和观点抽取模型包
├── classification                     # 细粒度情感分类模型包
├── speedup                           # PP-MiniLM特色小模型包
├── imgs                               # 图片目录
├── dynamic_predict.py                 # 全流程动态图单条预测脚本
├── dynamic_predict_by_batch.py        # 全流程动态图批量预测脚本
├── export_model.py                    # 动转静模型导出脚本
├── static_predict.py                  # 全流程静态图单条预测脚本
├── utils.py                           # 工具函数脚本
├── run_dynamic_predict.sh             # 全流程动态图单条预测命令
├── run_dynamic_predict_by_batch.sh    # 全流程动态图批量预测命令
├── run_export_model.sh                # 动转静模型导出命令
├── run_static_predict.sh              # 全流程静态图单条预测命令
├── requirements.txt                   # 环境依赖
└── README.md
```

### 3.1 运行环境和依赖安装
(1) 运行环境  
除非特殊说明，本实验默认是在以下配置环境研发运行的：
```shell
python version: 3.8
CUDA Version: 10.2
NVIDIA Driver Version: 440.64.00
GPU： Tesla V100
linux：CentOS Linux release 7.9.2009 (Core)
```
(2) 环境依赖  
```
python >= 3.6
paddlenlp >= 2.2.1
paddlepaddle-gpu >= 2.2.1
```

可以通过以下命令进行一键式软件环境安装：
```shell
pip install -r requirements.txt
```
(3) 运行环境准备  
在运行之前，请在本目录下新建文件夹 `data` 和 `checkpoints`，分别用于存放数据和保存模型。

### 3.2 数据说明
本项目需要训练两个阶段的模型：评论维度和观点抽取模型，细粒度情感分类模型。本次针对这抽取和分类模型，我们分别开源了 Demo 数据： [ext_data](https://bj.bcebos.com/v1/paddlenlp/data/ext_data.tar.gz)和[cls_data](https://bj.bcebos.com/v1/paddlenlp/data/cls_data.tar.gz)。

用户可分别点击下载，解压后将相应的数据文件依次放入 `./data/ext_data` 和 `./data/cls_data` 目录下即可。

### 3.3 评论维度和观点抽取模型
在抽取模型训练过程中，总共训练了10轮，并选择了评估F1得分最高的 best 模型，下表展示了训练过程中使用的训练参数。我们同时开源了相应的模型，可点击下表的 `ext_model` 进行下载，下载后将模型重命名为 `best.pdparams`，然后放入目录 `./checkpoints/ext_checkpoints` 中。
|Model|训练参数配置|MD5|
| ------------ | ------------ |-----------|
|[ext_model](https://bj.bcebos.com/paddlenlp/models/best_ext.pdparams)|<div style="width: 150pt"> learning_rate: 5e-5, batch_size: 8, max_seq_len:512, epochs：10 </div> |e3358632165aa0338225e175b57cb304|

我们基于训练过程中的 best 模型在验证集 `dev` 和测试集 `test` 上进行了评估测试，模型效果如下表所示:
|Model|数据集|precision|Recall|F1|
| ------------ | ------------ | ------------ |-----------|------------ |
|SKEP-Large|dev|0.87095|0.90056|0.88551|
|SKEP-Large|test|0.87125|0.89944|0.88512|

**备注**：以上数据是基于全量数据训练和测试结果，并非 Demo 数据集。关于评论维度和观点抽取模型的原理和使用方式，请参考[这里](extraction/README.md)。

### 3.4 细粒度情感分类模型
在分类模型训练过程中，总共训练了10轮，并选择了评估 F1 得分最高的 best 模型，下表展示了训练过程中使用的训练参数。我们同时开源了相应的模型，可点击下表的 `cls_model` 进行下载，下载后将模型重命名为 `best.pdparams`，然后放入目录 `./checkpoints/cls_checkpoints` 中。
|Model|训练参数配置|MD5|
| ------------ | ------------ |-----------|
|[cls_model](https://bj.bcebos.com/paddlenlp/models/best_cls.pdparams)|<div style="width: 150pt"> learning_rate: 3e-5, batch_size: 16, max_seq_len:256, epochs：10 </div>|3de6ddf581e665d9b1d035c29b49778a|

我们基于训练过程中的 best 模型在验证集 `dev` 和测试集 `test` 上进行了评估测试，模型效果如下表所示:
|Model|数据集|precision|Recall|F1|
| ------------ | ------------ | ------------ |-----------|------------ |
|SKEP-Large|dev|0.98758|0.99251|0.99004|
|SKEP-Large|test|0.98497|0.99139|0.98817|

**备注**： 以上数据是基于全量数据训练和测试结果，并非 Demo 数据集。
关于细粒度情感分类模型的原理和使用方式，请参考[这里](classification/README.md)。


### 3.5 全流程细粒度情感分析推理
在训练完成评论维度和观点模型，细粒度情感分类模型后，默认会将训练过程中最好的模型保存在 `./checkpoints/ext_checkpoints` 和 `./checkpoints/cls_checkpoints` 目录下。接下来，便可以根据保存好的模型进行全流程的模型推理：给定一句评论文本，首先使用抽取模型进行抽取评论维度和观点，然后使用细粒度情感分类模型以评论维度级别进行情感极性分类。

本项目将提供两套全流程预测方案：动态图预测和静态图高性能预测，两者默认均支持单条文本预测，同时考虑用户有批量输出处理的需求，本项目还支持了动态图批量预测。

其中，在单条预测时，用户需要根据运行相应命令后的提示传入待分析的文本，然后模型便会给出相应的分析结果，如下所示：

```
input_text: 蛋糕味道不错，很好吃，店家很耐心，服务也很好，很棒
aspect: 蛋糕味道, opinions: ['不错', '好吃'], sentiment_polarity: 正向
aspect: 店家, opinions: ['耐心'], sentiment_polarity: 正向
aspect: 服务, opinions: ['好', '棒'], sentiment_polarity: 正向
```

动态图批量预测时需要传入测试集文件路径，可将测试集文件放入本目录的 `./data` 文件夹下，模型在预测后会将结果以文件的形式存入测试集的同目录下。需要注意的是，测试集文件每行均为一个待预测的语句，如下所示。
```
蛋糕味道不错，很好吃，店家很耐心，服务也很好，很棒
酒店干净整洁，性价比很高
酒店环境不错，非常安静，性价比还可以
房间很大，环境不错
```

#### 3.5.1 全流程动态图预测
通过运行以下命令进行全流程动态图单条预测：
```shell
sh run_dynamic_predict.sh
```

通过运行以下命令进行动态图批量预测：
```shell
sh run_dynamic_predict_by_batch.sh
```

#### 3.5.2 静态图高性能预测
在基于静态图进行高性能预测过程中，首先需要将动态图模型转换为静态图模型，然后基于 Paddle Inference 高性能推理引擎进行预测。

通过以下命令分别将抽取模型和分类模型，从动态图转为静态图：
```shell
sh run_export_model.sh extraction
sh run_export_model.sh classification
```

基于Paddle Inference 进行动态图高性能预测：
```shell
sh run_static_predict.sh
```

### 3.6 小模型优化策略
本项目提供了一套基于 [PP-MiniLM](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm) 中文特色小模型的细粒度情感分类解决方案。PP-MiniLM 提供了一套完整的小模型优化方案：首先使用 Task-agnostic 的方式进行模型蒸馏、然后依托于 [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 进行模型裁剪、模型量化等模型压缩技术，有效减小了模型的规模，加快了模型运行速度。

本项目基于 PP-MiniLM 中文特色小模型进行 fine-tune 细粒度情感分类模型，然后使用 PaddleSlim 对训练好的模型进行量化操作。

在实验进行后，我们将 SKEP-Large、PP-MiniLM、量化PP-MiniLM 三个模型在性能和效果方面进行了对比，如下表所示。可以看到，三者在本任务数据集上的评估指标几乎相等，但是 PP-MiniLM 小模型运行速度较 SKEP-Large 提高了4倍，量化后的 PP-MiniLM 运行速度较 SKEP-Large 提高了近8倍。更多的详细信息请参考[这里](./speedup/README.md)。

|Model|运行时间(s)|precision|Recall|F1|
| ------------ | ------------ | ------------ |-----------|------------ |
|SKEP-Large|1.00x|0.98497|0.99139|0.98817|
|PP-MiniLM|4.95x|0.98379|0.98859|0.98618|
|量化 PP-MiniLM|8.93x|0.98312|0.98953|0.98631|

## 4. 引用

[1] H. Tian et al., “SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis,” arXiv:2005.05635 [cs], May 2020, Accessed: Nov. 11, 2021.
