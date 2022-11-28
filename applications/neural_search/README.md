# 手把手搭建一个语义检索系统

## 1. 场景概述

检索系统存在于我们日常使用的很多产品中，比如商品搜索系统、学术文献检索系等等，本方案提供了检索系统完整实现。限定场景是用户通过输入检索词 Query，快速在海量数据中查找相似文档。
<div align="center">
    <img src="https://user-images.githubusercontent.com/12107462/191490721-90a8f526-ad64-4f2b-b9b4-34ab06c6749b.png" width="500px">
</div>

所谓语义检索（也称基于向量的检索，如上图所示），是指检索系统不再拘泥于用户 Query 字面本身，而是能精准捕捉到用户 Query 后面的真正意图并以此来搜索，从而更准确地向用户返回最符合的结果。通过使用最先进的语义索引模型找到文本的向量表示，在高维向量空间中对它们进行索引，并度量查询向量与索引文档的相似程度，从而解决了关键词索引带来的缺陷。

例如下面两组文本 Pair，如果基于关键词去计算相似度，两组的相似度是相同的。而从实际语义上看，第一组相似度高于第二组。

```
车头如何放置车牌    前牌照怎么装
车头如何放置车牌    后牌照怎么装
```

语义检索系统的关键就在于，采用语义而非关键词方式进行召回，达到更精准、更广泛得召回相似结果的目的。想快速体验搜索的效果，请参考[Pipelines的语义检索实现](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/semantic-search)

<div align="center">
    <img src="https://user-images.githubusercontent.com/12107462/190302765-663ba441-9dd3-470a-8fee-f7a6f81da615.gif" width="500px">
</div>


## 2. 产品功能介绍

通常检索业务的数据都比较庞大，都会分为召回（索引）、排序两个环节。召回阶段主要是从至少千万级别的候选集合里面，筛选出相关的文档，这样候选集合的数目就会大大降低，在之后的排序阶段就可以使用一些复杂的模型做精细化或者个性化的排序。一般采用多路召回策略（例如关键词召回、热点召回、语义召回结合等），多路召回结果聚合后，经过统一的打分以后选出最优的 TopK 的结果。

### 2.1 系统特色

+ 低门槛
    + 手把手搭建起检索系统
    + 无需标注数据也能构建检索系统
    + 提供 训练、预测、ANN 引擎一站式能力
    + Pipelines 快速实现语义检索系统

+ 效果好
    + 针对多种数据场景的专业方案
        + 仅有无监督数据: SimCSE
        + 仅有有监督数据: InBatchNegative
        + 兼具无监督数据 和 有监督数据：融合模型
    + 进一步优化方案: 面向领域的预训练 Domain-adaptive Pretraining
+ 性能快
    + Paddle Inference 快速抽取向量
    + Milvus 快速查询和高性能建库
    + Paddle Serving服务化部署

###  2.2 功能架构

索引环节有两类方法：基于字面的关键词索引；语义索引。语义索引能够较好地表征语义信息，解决字面不相似但语义相似的情形。本系统给出的是语义索引方案，实际业务中可融合其他方案使用。下面就详细介绍整个方案的架构和功能。

#### 2.2.1 整体介绍


<div align="center">
    <img src="https://user-images.githubusercontent.com/12107462/191469309-42a54a67-a3a3-4e43-b6b1-b12be81ddf3d.png" width="800px">
</div>

以上是nerual_search的系统流程图，其中左侧为召回环节，核心是语义向量抽取模块；右侧是排序环节，核心是排序模型。召回环节需要用户通过自己的语料构建向量索引库，用户发起query了之后，就可以检索出相似度最高的向量，然后找出该向量对应的文本；排序环节主要是对召回的文本进行重新排序。下面我们分别介绍召回中的语义向量抽取模块，以及排序模型。


#### 2.2.2 召回模块

召回模块需要从千万量级数据中快速召回候选数据。首先需要抽取语料库中文本的 Embedding，然后借助向量搜索引擎实现高效 ANN，从而实现候选集召回。

我们针对不同的数据情况推出三种语义索引方案，如下图所示，您可以参照此方案，快速建立语义索引：

|  ⭐️ 无监督数据 |  ⭐️ 有监督数据 | **召回方案** |
| ------------ | ------------ | ------------ |
|  多 |  无 | SimCSE |
|  无 |  多 | In-batch Negatives|
|  有 | 有  | SimCSE+ In-batch Negatives |

最基本的情况是只有无监督数据，我们推荐您使用 SimCSE 进行无监督训练；另一种方案是只有有监督数据，我们推荐您使用 In-batch Negatives 的方法进行有监督训练。

如果想进一步提升模型效果：还可以使用大规模业务数据，对预训练模型进行 Domain-adaptive Pretraining，训练完以后得到预训练模型，再进行无监督的 SimCSE。

此外，如果您同时拥有监督数据和无监督数据，我们推荐将两种方案结合使用，这样能训练出更加强大的语义索引模型。

#### 2.2.3 排序模块

召回模型负责从海量（千万级）候选文本中快速（毫秒级）筛选出与 Query 相关性较高的 TopK Doc，排序模型会在召回模型筛选出的 TopK Doc 结果基础之上针对每一个 (Query, Doc) Pair 对进行两两匹配计算相关性，排序效果更精准。

排序模块有2种选择，第一种基于前沿的预训练模型 ERNIE，训练 Pair-wise 语义匹配模型；第二种是基于RocketQA模型训练的Cross Encoder模形。第一种是Pair-wise的排序算法，基本思路是对样本构建偏序文档对，两两比较，从比较中学习顺序，第二种是Poinet-Wise的算法，只考虑当前Query和每个文档的绝对相关度，并没有考虑其他文档与Query的相关度，但是建模方式比较简单。第一种Pair-wise模型可以说是第二种point-wise模型的改进版本，但对于噪声数据更为敏感，即一个错误的标注会导致多个pair对的错误，用户可以先使用基于Point-wise的Cross Encoder构建一个基础模型，需要进一步优化可以使用Pair-wise的方法优化。

## 3. 文献检索实践

### 3.1 技术方案和评估指标

#### 3.1.1 技术方案

**语义索引**：由于我们既有无监督数据，又有有监督数据，所以结合 SimCSE 和 In-batch Negatives 方案，并采取 Domain-adaptive Pretraining 优化模型效果。

首先是利用 ERNIE模型进行 Domain-adaptive Pretraining，在得到的预训练模型基础上，进行无监督的 SimCSE 训练，最后利用 In-batch Negatives 方法进行微调，得到最终的语义索引模型，把建库的文本放入模型中抽取特征向量，然后把抽取后的向量放到语义索引引擎 milvus 中，利用 milvus 就可以很方便得实现召回了。

**排序**：使用 ERNIE-Gram 的单塔结构/RocketQA的Cross Encoder对召回后的数据精排序。

#### 3.1.2 评估指标

**模型效果指标**
* 在语义索引召回阶段使用的指标是 Recall@K，表示的是预测的前topK（从最后的按得分排序的召回列表中返回前K个结果）结果和语料库中真实的前 K 个相关结果的重叠率，衡量的是检索系统的查全率。

* 在排序阶段使用的指标为AUC，AUC反映的是分类器对样本的排序能力，如果完全随机得对样本分类，那么AUC应该接近0.5。分类器越可能把真正的正样本排在前面，AUC越大，分类性能越好。

**性能指标**
* 基于 Paddle Inference 快速抽取向量

* 建库性能和 ANN 查询性能快

### 3.2 预置数据说明

数据集来源于某文献检索系统，既有大量无监督数据，又有有监督数据。

（1）采用文献的 query, title,keywords,abstract 四个字段内容，构建无标签数据集进行 Domain-adaptive Pretraining；

（2）采用文献的 query,title,keywords 三个字段内容，构造无标签数据集，进行无监督召回训练SimCSE；

（3）使用文献的的query, title, keywords，构造带正标签的数据集，不包含负标签样本，基于 In-batch Negatives 策略进行训练；

（4）在排序阶段，使用点击（作为正样本）和展现未点击（作为负样本）数据构造排序阶段的训练集，进行精排训练。

|  阶段 |模型 |   训练集 | 评估集（用于评估模型效果） | 召回库 |测试集 |
| ------------ | ------------ |------------ | ------------ | ------------ | ------------ |
|  召回 |  Domain-adaptive Pretraining  |  2kw | - | - | - |
|  召回 |  无监督预训练 - SimCSE |  798w  | 20000 |  300000| 1000 |
|  召回 |  有监督训练 - In-batch Negatives | 3998  | 20000 |300000  | 1000 |
|  排序 |  有监督训练 - ERNIE-Gram单塔 Pairwise/RocketQA Cross Encoder| 1973538   | 57811 | - | 1000 |

我们将除 Domain-adaptive Pretraining 之外的其他数据集全部开源，下载地址：

- [literature_search_data](https://bj.bcebos.com/v1/paddlenlp/data/literature_search_data.zip)
- [literature_search_rank](https://paddlenlp.bj.bcebos.com/applications/literature_search_rank.zip)

```
├── milvus # milvus建库数据集
    ├── milvus_data.csv.  # 构建召回库的数据（模拟实际业务线上的语料库，实际语料库远大于这里的规模），用于直观演示相关文献召回效果
├── recall  # 召回阶段数据集
    ├── train_unsupervised.csv # 无监督训练集，用于训练 SimCSE
    ├── train.csv  # 有监督训练集，用于训练 In-batch Negative
    ├── dev.csv  # 召回阶段验证集，用于评估召回模型的效果，SimCSE 和 In-batch Negative 共用
    ├── corpus.csv # 构建召回库的数据（模拟实际业务线上的语料库，实际语料库远大于这里的规模），用于评估召回阶段模型效果，SimCSE 和 In-batch Negative 共用
    ├── test.csv # 召回阶段测试数据，预测文本之间的相似度，SimCSE 和 In-batch Negative 共用
├── data # RocketQA排序数据集
    ├── test.csv   # 测试集
    ├── dev_pairwise.csv    # 验证集
    └── train.csv  # 训练集
├── sort # 排序阶段数据集
    ├── train_pairwise.csv  # 排序训练集
    ├── dev_pairwise.csv    # 排序验证集
    └── test_pairwise.csv   # 排序测试集
```


### 3.3 数据格式

1. 对于无监督SimCSE的训练方法，格式参考`train_unsupervised.csv`,即一行条文本即可，无需任何标注。对于召回模型训练需要规定格式的本地数据集，需要准备训练集文件`train.csv`，验证集`dev.csv`，召回集文件`corpus.csv`。


训练数据集`train.csv`的格式如下：

```
query1 \t 用户点击的title1
query2 \t 用户点击的title2
```
训练集合`train.csv`的文件样例：
```
从《唐律疏义》看唐代封爵贵族的法律特权  从《唐律疏义》看唐代封爵贵族的法律特权《唐律疏义》,封爵贵族,法律特权
宁夏社区图书馆服务体系布局现状分析      宁夏社区图书馆服务体系布局现状分析社区图书馆,社区图书馆服务,社区图书馆服务体系
人口老龄化对京津冀经济  京津冀人口老龄化对区域经济增长的影响京津冀,人口老龄化,区域经济增长,固定效应模型
英语广告中的模糊语      模糊语在英语广告中的应用及其功能模糊语,英语广告,表现形式,语用功能
甘氨酸二肽的合成        甘氨酸二肽合成中缩合剂的选择甘氨酸,缩合剂,二肽
......
```

验证集`dev.csv`的格式如下：

```
query1 \t 用户点击的title1
query2 \t 用户点击的title2
```

验证集合`train.csv`的文件样例：
```
试论我国海岸带经济开发的问题与前景      试论我国海岸带经济开发的问题与前景海岸带,经济开发,问题,前景
外语阅读焦虑与英语成绩及性别的关系      外语阅读焦虑与英语成绩及性别的关系外语阅读焦虑,外语课堂焦虑,英语成绩,性别
加油站风险分级管控      加油站工作危害风险分级研究加油站,工作危害分析(JHA),风险分级管控
```
召回集合`corpus.csv`主要作用是检验测试集合的句子对能否被正确召回，它的构造主要是提取验证集的第二列的句子，然后加入很多无关的句子，用来检验模型能够正确的从这些文本中找出测试集合对应的第二列的句子，格式如下：

```
2002-2017年我国法定传染病发病率和死亡率时间变化趋势传染病,发病率,死亡率,病死率
陕西省贫困地区城乡青春期少女生长发育调查青春期,生长发育,贫困地区
五丈岩水库溢洪道加固工程中的新材料应用碳纤维布,粘钢加固技术,超细水泥,灌浆技术
......
```

2. 对于排序模型的训练，排序模型目前提供了2种，第一种是Pairwise训练的方式，第二种是RocketQA的排序模型，对于第一种排序模型，需要准备训练集`train_pairwise.csv`,验证集`dev_pairwise.csv`两个文件,除此之外还可以准备测试集文件`test.csv`或者`test_pairwise.csv`。

训练数据集`train_pairwise.csv`的格式如下：

```
query1 \t 用户点击的title1 \t 用户未点击的title2
query2 \t 用户点击的title3 \t 用户未点击的title4
```

训练数据集`train_pairwise.csv`的示例如下：

```
英语委婉语引起的跨文化交际障碍  英语委婉语引起的跨文化交际障碍及其翻译策略研究英语委婉语,跨文化交际障碍,翻译策略        委婉语在英语和汉语中的文化差异委婉语,文化,跨文化交际
范迪慧 嘉兴市中医院     滋阴疏肝汤联合八穴隔姜灸治疗肾虚肝郁型卵巢功能低下的临床疗效滋阴疏肝汤,八穴隔姜灸,肾虚肝郁型卵巢功能低下,性脉甾类激素,妊娠      温针灸、中药薰蒸在半月板损伤术后康复中的疗效分析膝损伤,半月板,胫骨,中医康复,温针疗法,薰洗
......
```

验证数据集`dev_pairwise.csv`的格式如下：

```
query1 \t title1 \t label
query2 \t title2 \t label
```
验证数据集`dev_pairwise.csv`的示例如下：

```
作者单位:南州中学       浅谈初中教学管理如何体现人文关怀初中教育,教学管理,人文关怀      1
作者单位:南州中学       高中美术课堂教学中藏区本土民间艺术的融入路径藏区,传统民间艺术,美术课堂  0
作者单位:南州中学       列宁关于资产阶级民主革命向 社会主义革命过渡的理论列宁,直接过渡,间接过渡,资产阶级民主革命,社会主义革命   0
DAA髋关节置换   DAA前侧入路和后外侧入路髋关节置换疗效对比髋关节置换术;直接前侧入路;后外侧入路;髋关节功能;疼痛;并发症    1
DAA髋关节置换   DAA全髋关节置换术治疗髋关节病变对患者髋关节运动功能的影响直接前侧入路全髋关节置换术,髋关节病变,髋关节运动功能   0
DAA髋关节置换   护患沟通技巧在急诊输液护理中的应用分析急诊科,输液护理,护理沟通技巧,应用 0
.......
```
训练数据集`test_pairwise.csv`的格式如下，其中这个score得分是召回算出来的相似度或者距离，仅供参考，可以忽略：

```
query1 \t title1 \t score
query2 \t title2 \t score
```
训练数据集`test_pairwise.csv`的示例如下：

```
中西方语言与文化的差异  中西方文化差异以及语言体现中西方文化,差异,语言体现      0.43203747272491455
中西方语言与文化的差异  论中西方文化差异在非言语交际中的体现中西方文化,差异,非言语交际  0.4644506871700287
中西方语言与文化的差异  中西方体态语文化差异跨文化,体态语,非语言交际,差异       0.4917311668395996
中西方语言与文化的差异  由此便可以发现两种语言以及两种文化的差异。      0.5039259195327759
.......
```

对于第二种基于RocketQA的排序模型。

训练数据集`train.csv`,验证集`dev_pairwise.csv`的格式如下：

```
query1 \t title1 \t label
query2 \t title2 \t label
```
训练数据集`train.csv`,验证集`dev_pairwise.csv`的示例如下：

```
(小学数学教材比较) 关键词:新加坡        新加坡与中国数学教材的特色比较数学教材,教材比较,问题解决        0
徐慧新疆肿瘤医院        头颈部非霍奇金淋巴瘤扩散加权成像ADC值与Ki-67表达相关性分析淋巴瘤,非霍奇金,头颈部肿瘤,磁共振成像 1
抗生素关性腹泻  鼠李糖乳杆菌GG防治消化系统疾病的研究进展鼠李糖乳杆菌,腹泻,功能性胃肠病,肝脏疾病,幽门螺杆菌      0
德州市图书馆    图书馆智慧化建设与融合创新服务研究图书馆;智慧化;阅读服务;融合创新       1
维生素c 综述    维生素C防治2型糖尿病研究进展维生素C;2型糖尿病;氧化应激;自由基;抗氧化剂  0
.......
```

训练数据集`test.csv`的格式如下，其中这个score得分是召回算出来的相似度或者距离，仅供参考，可以忽略：

```
query1 \t title1 \t score
query2 \t title2 \t score
```
训练数据集`test.csv`的示例如下：

```
加强科研项目管理有效促进医学科研工作    科研项目管理策略科研项目,项目管理,实施,必要性,策略      0.32163668
加强科研项目管理有效促进医学科研工作    关于推进我院科研发展进程的相关问题研究医院科研,主体,环境,信息化 0.32922596
加强科研项目管理有效促进医学科研工作    深圳科技计划对高校科研项目资助现状分析与思考基础研究,高校,科技计划,科技创新     0.36869502
加强科研项目管理有效促进医学科研工作    普通高校科研管理模式的优化与创新普通高校,科研,科研管理  0.3688045
.......
```


### 3.4 运行环境和安装说明


（1）运行环境

本实验采用了以下的运行环境进行，详细说明如下，用户也可以在自己 GPU 硬件环境进行：

a. 软件环境：


- python >= 3.6
- paddlenlp >= 2.2.1
- paddlepaddle-gpu >=2.2
- CUDA Version: 10.2
- NVIDIA Driver Version: 440.64.00
- Ubuntu 16.04.6 LTS (Docker)


b. 硬件环境：


- NVIDIA Tesla V100 16GB x4卡
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz


c. 依赖安装:

```
pip install -r requirements.txt
```

## 4. Neural Search 快速体验实践

PaddleNLP已经基于ERNIE 1.0训练了一个基线模型，如果想快速搭建Neural Search的完整系统，有两种方法，第一种是请参考下面的实现，包含了服务化的完整流程，另一种是使用Pipelines加载，Pipelines已经支持Neural Search训练的模型的载入，可以使用Pipelines的快速的基于Neural Search模型实现检索系统，详情请参考文档[Pipelines-Neural-Search](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/semantic-search/Neural_Search.md)。

### 4.1. 召回

- 召回向量抽取服务的搭建请参考：[In-batch Negatives](./recall/in_batch_negative/)， 只需要下载基于ERNIE 1.0的预训练模型，导出成Paddle Serving的格式，然后启动Pipeline Server服务即可

- 召回向量检索服务的搭建请参考：[Milvus](./recall/milvus/)， 需要搭建Milvus并且插入检索数据的向量

【注意】如果使用Neural Search训练好的模型，由于该模型是基于ERNIE 1.0训练的，所以需要把 `model_name_or_path`指定为`ernie 1.0`，向量抽取结果才能正常。


### 4.2. 排序

排序服务的搭建请参考 [ernie_matching](./ranking/ernie_matching/)，只需要下载基于ERNIE Gram的预训练模型，导出成Paddle Serving的格式，最后需要启动 Pipeline Serving服务

【注意】如果使用Neural Search训练好的模型，由于该模型是基于ERNIE Gram训练的，所以需要把 `model_name_or_path`指定为`ernie-gram-zh`，向量抽取结果才能正常。

### 4.3. 系统运行

以上召回和排序模型都经过Paddle Serving服务化以后，就可以直接使用下面的命令运行体验：

```
python3 run_system.py
```
输出的结果为：

```
PipelineClient::predict pack_data time:1656991375.5521955
PipelineClient::predict before time:1656991375.5529568
Extract feature time to cost :0.0161135196685791 seconds
Search milvus time cost is 0.8139839172363281 seconds
PipelineClient::predict pack_data time:1656991376.3981335
PipelineClient::predict before time:1656991376.3983877
time to cost :0.05616641044616699 seconds
```
会输出2个文件 `recall_result.csv` 是召回检索的结果，`rank_result.csv` 是排序的结果。csv的示例输出下。

召回的结果：

```
中西方语言与文化的差异,港台文化对内地中小学生的负面影响,0.055068351328372955
中西方语言与文化的差异,外来文化在越南的传播与融合,0.05621318891644478
中西方语言与文化的差异,临终关怀中的“仪式”,0.05705389380455017
中西方语言与文化的差异,历史的真实与艺术加工,0.05745899677276611
......
```

排序的结果：

```
中西方语言与文化的差异,论中西方教育差异,0.870943009853363
中西方语言与文化的差异,浅析中西方问候语的差异,0.8468159437179565
中西方语言与文化的差异,文化认同及其根源,0.8288694620132446
中西方语言与文化的差异,从历史文化角度分析中西方学校教育的差异,0.8209370970726013
中西方语言与文化的差异,中西医思维方式的差异,0.8150948882102966
中西方语言与文化的差异,浅析中韩餐桌文化差异,0.7751647233963013
......
```



## 5. 从头开始搭建自己的检索系统

这里展示了能够从头至尾跑通的完整代码，您使用自己的业务数据，照着跑，能搭建出一个给定 Query，返回 topK 相关文档的小型检索系统。您可以参照我们给出的效果和性能数据来检查自己的运行过程是否正确。

### 5.1 召回阶段

**召回模型训练**

我们进行了多组实践，用来对比说明召回阶段各方案的效果：

|  模型 |  Recall@1 | Recall@5 |Recall@10 |Recall@20 |Recall@50 |策略简要说明|
| ------------ | ------------ | ------------ |--------- |--------- |--------- |--------- |
|  有监督训练 Baseline | 30.077| 43.513| 48.633 | 53.448 |59.632| 标准 pair-wise 训练范式，通过随机采样产生负样本|
|  有监督训练 In-batch Negatives |  51.301 | 65.309| 69.878| 73.996|78.881| In-batch Negatives 有监督训练|
|  无监督训练 SimCSE |  42.374 | 57.505| 62.641| 67.09|72.331| SimCSE 无监督训练|
|  无监督 + 有监督训练 SimCSE + In-batch Negatives |  55.976 | 71.849| 76.363| 80.49|84.809| SimCSE无监督训练，In-batch Negatives 有监督训练|
|  Domain-adaptive Pretraining + SimCSE |  51.031 | 66.648| 71.338 | 75.676 |80.144| ERNIE 预训练，SimCSE 无监督训练|
|  Domain-adaptive Pretraining + SimCSE + In-batch Negatives|  **58.248** | **75.099**| **79.813**| **83.801**|**87.733**| ERNIE 预训练，SimCSE 无监督训训练，In-batch Negatives 有监督训练|

从上述表格可以看出，首先利用 ERNIE 3.0 做 Domain-adaptive Pretraining ，然后把训练好的模型加载到 SimCSE 上进行无监督训练，最后利用 In-batch Negatives 在有监督数据上进行训练能够获得最佳的性能。[模型下载](https://paddlenlp.bj.bcebos.com/models/inbatch_model_best.zip)，模型的使用方式参考[In-batch Negatives](./recall/in_batch_negative/) 。


这里采用 Domain-adaptive Pretraining + SimCSE + In-batch Negatives 方案：

第一步：无监督训练 Domain-adaptive Pretraining

训练用时 16hour55min，可参考：[ERNIE 1.0](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-1.0)

第二步：无监督训练 SimCSE

训练用时 16hour53min，可参考：[SimCSE](./recall/simcse/)

第三步：有监督训练

几分钟内训练完成，可参考 [In-batch Negatives](./recall/in_batch_negative/)


**召回系统搭建**

召回系统使用索引引擎 Milvus，可参考 [milvus_system](./recall/milvus/)。
我们展示一下系统的效果，输入的文本如下：

```
中西方语言与文化的差异

```
下面是召回的部分结果，第一个是召回的title，第二个数字是计算的相似度距离

```
跨文化中的文化习俗对翻译的影响翻译,跨文化,文化习俗	0.615584135055542
试论翻译过程中的文化差异与语言空缺翻译过程,文化差异,语言空缺,文化对比	0.6155391931533813
中英文化差异及习语翻译习语,文化差异,翻译	0.6153547763824463
英语中的中国文化元素英语,中国文化,语言交流	0.6151996850967407
跨文化交际中的文化误读研究文化误读,影响,中华文化,西方文明	0.6137217283248901
在语言学习中了解中法文化差异文化差异,对话交际,语言	0.6134252548217773
从翻译视角看文化差异影响下的中式英语的应对策略文化差异;中式英语现;汉英翻译;动态对等理论	0.6127341389656067
归化与异化在跨文化传播中的动态平衡归化,异化,翻译策略,跨文化传播,文化外译	0.6127211451530457
浅谈中西言语交际行为中的文化差异交际用语,文化差异,中国,西方	0.6125463843345642
翻译中的文化因素--异化与归化文化翻译,文化因素,异化与归化	0.6111845970153809
历史与文化差异对翻译影响的分析研究历史与文化差异,法汉翻译,翻译方法	0.6107486486434937
从中、韩、美看跨文化交际中的东西方文化差异跨文化交际,东西方,文化差异	0.6091923713684082
试论文化差异对翻译工作的影响文化差异,翻译工作,影响	0.6084284782409668
从归化与异化看翻译中的文化冲突现象翻译,文化冲突,归化与异化,跨文化交际	0.6063553690910339
中西方问候语的文化差异问候语,文化差异,文化背景	0.6054259538650513
中英思维方式的差异对翻译的影响中英文化的差异,中英思维方式的差异,翻译	0.6026732921600342
略论中西方语言文字的特性与差异语言,会意,确意,特性,差异	0.6009351015090942
......

```


### 5.2 排序阶段

排序阶段有2种方案，第一种是[ernie_matching](./ranking/ernie_matching/)使用的模型是 ERNIE-3.0-Medium-zh，用时 20h；第二种是基于RocketQA的排序模型[cross_encoder](./ranking/cross_encoder/)，训练用时也是20h左右。


排序阶段的效果评估：

|  模型 |  AUC |
| ------------ | ------------ |
|  Baseline: In-batch Negatives |  0.582 |
| pairwise ERNIE-Gram  |0.801 |
|  CrossEncoder：rocketqa-base-cross-encoder  |**0.835** |


同样输入文本：

```
中西方语言与文化的差异
```
排序阶段的结果展示如下，第一个是 Title ，第二个数字是计算的概率，显然经排序阶段筛选的文档与 Query 更相关：

```
中西方文化差异以及语言体现中西方文化,差异,语言体现	0.999848484992981
论中西方语言与文化差异的历史渊源中西方语言,中西方文化,差异,历史渊源	0.9998375177383423
从日常生活比较中西方语言与文化的差异中西方,语言,文化,比较	0.9985846281051636
试论中西方语言文化教育的差异比较与融合中西方,语言文化教育,差异	0.9972485899925232
中西方文化差异对英语学习的影响中西方文化,差异,英语,学习	0.9831035137176514
跨文化视域下的中西文化差异研究跨文化,中西,文化差异	0.9781349897384644
中西方文化差异对跨文化交际的影响分析文化差异,跨文化交际,影响	0.9735479354858398
探析跨文化交际中的中西方语言差异跨文化交际,中西方,语言差异	0.9668175578117371
中西方文化差异解读中英文差异表达中西文化,差异表达,跨文化交际	0.9629314541816711
中西方文化差异对英语翻译的影响中西方文化差异,英语翻译,翻译策略,影响	0.9538986086845398
论跨文化交际中的中西方文化冲突跨文化交际,冲突,文化差异,交际策略,全球化	0.9493677616119385
中西方文化差异对英汉翻译的影响中西方文化,文化差异,英汉翻译,影响	0.9430705904960632
中西方文化差异与翻译中西方,文化差异,翻译影响,策略方法,译者素质	0.9401137828826904
外语教学中的中西文化差异外语教学,文化,差异	0.9397934675216675
浅析西语国家和中国的文化差异-以西班牙为例跨文化交际,西语国家,文化差异	0.9373322129249573
中英文化差异在语言应用中的体现中英文化,汉语言,语言应用,语言差异	0.9359155297279358
....
```


## Reference

[1] Tianyu Gao, Xingcheng Yao, Danqi Chen: [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821). EMNLP (1) 2021: 6894-6910

[2] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906). Preprint 2020.

[3] Dongling Xiao, Yu-Kun Li, Han Zhang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang:
[ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding](https://arxiv.org/abs/2010.12148). NAACL-HLT 2021: 1702-1715

[4] Yu Sun, Shuohuan Wang, Yu-Kun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, Hua Wu:
[ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223). CoRR abs/1904.09223 (2019)
