 **目录**

* [背景介绍](#背景介绍)
* [Milvus召回](#Milvus召回)
    * [1. 技术方案和评估指标](#技术方案)
    * [2. 环境依赖](#环境依赖)
    * [3. 代码结构](#代码结构)
    * [4. 数据准备](#数据准备)
    * [5. 向量检索](#向量检索)


<a name="背景介绍"></a>

# 背景介绍

基于某检索平台开源的数据集构造生成了面向语义索引的召回库。

<a name="Milvus召回"></a>

# Milvus召回

<a name="技术方案"></a>

## 1. 技术方案和评估指标

### 技术方案

使用 Milvus 搭建召回系统，然后使用训练好的语义索引模型，抽取向量，插入到 Milvus 中，然后进行检索。

<a name="环境依赖"></a>

## 2. 环境依赖和安装说明

**环境依赖**
* python >= 3.6.2
* paddlepaddle >= 2.2
* paddlenlp >= 2.2
* milvus >= 2.1.0
* pymilvus >= 2.1.0

<a name="代码结构"></a>

## 3. 代码结构

## 代码结构：

```
|—— scripts
    |—— feature_extract.sh  提取特征向量的bash脚本
    |—— search.sh  插入向量和向量检索bash脚本
├── base_model.py # 语义索引模型基类
├── config.py  # milvus配置文件
├── data.py # 数据处理函数
├── milvus_ann_search.py # 向量插入和检索的脚本
├── inference.py # 动态图模型向量抽取脚本
├── feature_extract.py # 批量抽取向量脚本
├── milvus_util.py # milvus的工具类
└── README.md
```
<a name="数据准备"></a>

## 4. 数据准备

数据集的样例如下，有两种，第一种是 title+keywords 进行拼接；第二种是一句话。

```
煤矸石-污泥基活性炭介导强化污水厌氧消化煤矸石,污泥,复合基活性炭,厌氧消化,直接种间电子传递
睡眠障碍与常见神经系统疾病的关系睡眠觉醒障碍,神经系统疾病,睡眠,快速眼运动,细胞增殖,阿尔茨海默病
城市道路交通流中观仿真研究智能运输系统;城市交通管理;计算机仿真;城市道路;交通流;路径选择
....
```

### 数据集下载


- [literature_search_data](https://bj.bcebos.com/v1/paddlenlp/data/literature_search_data.zip)

```
├── milvus # milvus建库数据集
    ├── milvus_data.csv.  # 构建召回库的数据
├── recall  # 召回（语义索引）数据集
    ├── corpus.csv # 用于测试的召回库
    ├── dev.csv  # 召回验证集
    ├── test.csv # 召回测试集
    ├── train.csv  # 召回训练集
    ├── train_unsupervised.csv # 无监督训练集
├── sort # 排序数据集
    ├── test_pairwise.csv   # 排序测试集
    ├── dev_pairwise.csv    # 排序验证集
    └── train_pairwise.csv  # 排序训练集

```

<a name="向量检索"></a>

## 5. 向量检索

### 5.1 基于Milvus的向量检索系统搭建

数据准备结束以后，我们开始搭建 Milvus 的语义检索引擎，用于语义向量的快速检索，我们使用[Milvus](https://milvus.io/)开源工具进行召回，Milvus 的搭建教程请参考官方教程  [Milvus官方安装教程](https://milvus.io/docs/v2.1.x/install_standalone-docker.md)本案例使用的是 Milvus 的2.1版本，建议使用官方的 Docker 安装方式，简单快捷。

Milvus 搭建完系统以后就可以插入和检索向量了，首先生成 embedding 向量，每个样本生成256维度的向量，使用的是32GB的V100的卡进行的提取：

```
CUDA_VISIBLE_DEVICES=0 python feature_extract.py \
        --model_dir=./output \
        --corpus_file "data/milvus_data.csv"
```
其中 output 目录下存放的是召回的 Paddle Inference 静态图模型。

|  数据量 |  时间 |
| ------------ | ------------ |
|1000万条|3hour40min39s|

运行结束后会生成 corpus_embedding.npy

生成了向量后，需要把数据插入到 Milvus 库中，首先修改配置：

修改 config.py 的配置 ip 和端口，本项目使用的是8530端口，而 Milvus 默认的是19530，需要根据情况进行修改：

```
MILVUS_HOST='your milvus ip'
MILVUS_PORT = 8530
```

然后运行下面的命令把向量插入到Milvus库中：

```
python milvus_ann_search.py --data_path milvus/milvus_data.csv \
                            --embedding_path corpus_embedding.npy \
                            --batch_size 100000 \
                            --index 0 \
                            --insert
```
参数含义说明

* `data_path`: 数据的路径
* `embedding_path`: 数据对应向量的路径
* `index`: 选择检索向量的索引，用于向量检索
* `insert`: 是否插入向量
* `search`: 是否检索向量
* `batch_size`: 表示的是一次性插入的向量的数量


|  数据量 |  时间 |
| ------------ | ------------ |
|1000万条|21min12s|

另外，Milvus提供了可视化的管理界面，可以很方便的查看数据，安装地址为[Attu](https://github.com/zilliztech/attu).

![](../../img/attu.png)


运行召回脚本：

```
python milvus_ann_search.py --data_path milvus/milvus_data.csv \
                            --embedding_path corpus_embedding.npy \
                            --batch_size 100000 \
                            --index 18 \
                            --search
```

运行以后的结果的输出为：

```
hit: (distance: 0.0, id: 18), text field: 吉林铁合金集团资产管理现状分析及对策资产管理;资金控制;应收帐款风险;造价控制;集中化财务控制
hit: (distance: 0.6585227251052856, id: 99102), text field: 资产结构管理的重点，在于确定一个既能维持企业正常开展经营活动，又能在减少或不增加风险的前提下，给企业带来更多利润的流动资金水平。
hit: (distance: 0.8119696974754333, id: 34124), text field: 关于电厂企业计划管理中固定资产管理的重要性电厂企业,计划管理,固定资产管理,重要性
hit: (distance: 0.8282783627510071, id: 70874), text field: 《商业银行资产负债优化管理:数理建模与应用》内容简介：资产负债管理是一种总体风险控制与资源配给方法，是把资产与负债组合视为有机整体，调整资产负债在总量上平衡、结构上对称、质量上优化，以实现利润最大化的方法。
...
```
返回的是向量的距离，向量的id，以及对应的文本。

也可以一键执行上述的过程：

```
sh scripts/search.sh
```

### 5.2 文本检索

首先修改代码的模型路径和样本：

```
params_path='checkpoints/model_40/model_state.pdparams'
id2corpus={0:'国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据'}
```

运行命令

```
python3 inference.py

```
运行的输出为，分别是抽取的向量和召回的结果：

```
[1, 256]
[[ 0.06374735 -0.08051944  0.05118101 -0.05855767 -0.06969483  0.05318566
   0.079629    0.02667932 -0.04501902 -0.01187392  0.09590752 -0.05831281
   ....
5677638 国有股权参股对家族企业创新投入的影响混合所有制改革,国有股权,家族企业,创新投入 0.5417419672012329
1321645 高管政治联系对民营企业创新绩效的影响——董事会治理行为的非线性中介效应高管政治联系,创新绩效,民营上市公司,董事会治理行为,中介效应 0.5445536375045776
1340319 国有控股上市公司资产并购重组风险探讨国有控股上市公司,并购重组,防范对策 0.5515031218528748
....
```
## FAQ

#### 抽取文本语义向量后，利用 Milvus 进行 ANN 检索查询到了完全相同的文本，但是计算出的距离为什么不是 0？

使用的是近似索引，详情请参考Milvus官方文档，[索引创建机制](https://milvus.io/cn/docs/v2.0.x/index.md)
