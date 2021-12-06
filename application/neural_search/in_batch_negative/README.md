# In-batch negatives

 **目录**

* [背景介绍](#背景介绍)
* [In-batch negatives](#In-batch)
    * [1. 技术方案和评估指标](#技术方案)
    * [2. 环境依赖](#环境依赖)  
    * [3. 代码结构](#代码结构)
    * [4. 数据准备](#数据准备)
    * [5. 模型训练](#模型训练)
    * [6. 评估](#开始评估)
    * [7. 预测](#预测)
    * [8. 部署](#部署)

<a name="背景介绍"></a>

# 背景介绍

语义索引（可通俗理解为向量索引）技术是搜索引擎、推荐系统、广告系统在召回阶段的核心技术之一。语义索引模型的目标是：给定输入文本，模型可以从海量候选召回库中**快速、准确**地召回一批语义相关文本。语义索引模型的效果直接决定了语义相关的物料能否被成功召回进入系统参与上层排序，从基础层面影响整个系统的效果。

在召回阶段，最常见的方式是通过双塔模型，学习Document(简写为Doc)的向量表示，对Doc端建立索引，用ANN召回。我们在这种方式的基础上，引入语义索引策略 [In-batch negatives](https://arxiv.org/abs/2004.04906)，以如下Batch size=4的训练数据为例：


```
煤矸石-污泥基活性炭介导强化污水厌氧消化 煤矸石-污泥基活性炭介导强化污水厌氧消化煤矸石,污泥,复合基活性炭,厌氧消化,直接种间电子传递
. 睡眠障碍与常见神经系统疾病的关系      睡眠障碍与常见神经系统疾病的关系睡眠觉醒障碍,神经系统疾病,睡眠,快速眼运动,细胞增殖,阿尔茨海默病
城市道路交通流中观仿真研究      城市道路交通流中观仿真研究智能运输系统;城市交通管理;计算机仿真;城市道路;交通流;路径选择
网络健康可信性研究      网络健康可信性研究网络健康信息;可信性;评估模式
脑瘫患儿家庭复原力的影响因素及干预模式雏形 研究 脑瘫患儿家庭复原力的影响因素及干预模式雏形研究脑瘫患儿;家庭功能;干预模式
地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较      地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较
```

In-batch negatives 策略的训练数据为语义相似的 Pair 对，策略核心是在 1 个 Batch 内同时基于 N 个负例进行梯度更新，将Batch 内除自身之外其它所有 Source Text 的相似文本 Target Text 作为负例，例如: 上例中“我手机丢了，我想换个手机” 有 1 个正例(”我想买个新手机，求推荐“)，3 个负例(1.求秋色之空全集漫画，2.手机学日语的软件，3.侠盗飞车罪恶都市怎么改车)。


<a name="In-batch"></a>

# In-batch negatives 

<a name="技术方案"></a>

## 1. 技术方案和评估指标

### 技术方案
双塔模型，采用ERNIE1.0热启
在...阶段引入In-batch negatives 策略...


### 评估指标

（1）采用 Recall@1，Recall@5 ，Recall@10 ，Recall@20  和 Recall@50 指标来评估语义索引模型的召回效果。

**效果评估**

|  模型 |  Recall@1 | Recall@5 |Recall@10 |Recall@20 |Recall@50 |策略简要说明|
| ------------ | ------------ | ------------ |--------- |--------- |--------- |--------- |
|  In-batch negatives |  51.301 | 65.309| 69.878| 73.996|78.881| Inbatch-negative有监督训练|



<a name="环境依赖"></a>

## 2. 环境依赖和安装说明

**环境依赖**
* python >= 3.x
* paddlepaddle-gpu >= 2.1.3
* paddlenlp >= 2.1
* hnswlib >=0.5.2

- [hnswlib](https://github.com/nmslib/hnswlib)

<a name="代码结构"></a>

## 3. 代码结构

```
|—— train_batch_neg.py # In-batch negatives 策略的训练主脚本
|—— batch_negative
    |—— model.py # In-batch negatives 策略核心网络结构
|—— deploy
    |—— python
        |—— predict.py # PaddleInference
|—— ann_util.py # Ann 建索引库相关函数
|—— base_model.py # 语义索引模型基类
|—— data.py # 数据读取、数据转换等预处理逻辑
|—— evaluate.py # 根据召回结果和评估集计算评估指标
|—— predict.py # 给定输入文件，计算文本 pair 的相似度
|—— recall.py # 基于训练好的语义索引模型，从召回库中召回给定文本的相似文本
```


<a name="数据准备"></a>

## 4. 数据准备

### 数据集说明

**构造训练集**


我们基于语义匹配数据集构造生成了面向语义索引的训练集、评估集、召回库。

样例数据如下:
```
煤矸石-污泥基活性炭介导强化污水厌氧消化 煤矸石-污泥基活性炭介导强化污水厌氧消化煤矸石,污泥,复合基活性炭,厌氧消化,直接种间电子传递
. 睡眠障碍与常见神经系统疾病的关系      睡眠障碍与常见神经系统疾病的关系睡眠觉醒障碍,神经系统疾病,睡眠,快速眼运动,细胞增殖,阿尔茨海默病
城市道路交通流中观仿真研究      城市道路交通流中观仿真研究智能运输系统;城市交通管理;计算机仿真;城市道路;交通流;路径选择
网络健康可信性研究      网络健康可信性研究网络健康信息;可信性;评估模式
脑瘫患儿家庭复原力的影响因素及干预模式雏形 研究 脑瘫患儿家庭复原力的影响因素及干预模式雏形研究脑瘫患儿;家庭功能;干预模式
地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较      地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较
```


#### 构造数据集

用下面的脚本划分训练集和测试集

```
python split_data.py
```

#### 构造召回库

```
python generate_recall.py
```

### 数据集下载


- [literature_search_data](https://bj.bcebos.com/v1/paddlenlp/data/literature_search_data.zip)


<a name="模型训练"></a>

## 5. 模型训练

**语义索引预训练模型下载链接：**

以下模型结构参数为: `TrasformerLayer:12, Hidden:768, Heads:12, OutputEmbSize: 256`

|Model|训练参数配置|硬件|MD5|
| ------------ | ------------ | ------------ |-----------|
|[batch_neg_v1.0](https://paddlenlp.bj.bcebos.com/models/semantic_index/batch_neg_v1.0.tar)|<div style="width: 150pt">margin:0.2 scale:30 epoch:3 lr:5E-5 bs:128 max_len:64 </div>|<div style="width: 100pt">单卡v100-16g</div>|da1bb1487bd3fd6a53b8ef95c278f3e6|

### 训练环境说明
说明系统（Linux or Windows，cpu or gpu）硬件情况

### 单机单卡训练/单机多卡训练
默认采用单机单卡训练，如果单机单卡跑不动，需要说明。
这里采用单机多卡方式进行训练，通过如下命令，指定 GPU 0,1,2,3 卡, 基于 In-batch negatives 策略训练模型，用时xx天/小时/分钟。

```
root_path=train_0.001
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    train_batch_neg.py \
    --device gpu \
    --save_dir ./checkpoints/${root_path} \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 3 \
    --output_emb_size 256 \
    --save_steps 1 \
    --max_seq_length 64 \
    --margin 0.2 \
    --train_set_file data/${root_path}/train.csv 
```
参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `save_dir`: 模型存储路径
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `save_steps`： 模型存储 checkpoint 的间隔 steps 个数
* `margin`: 正样本相似度与负样本之间的目标 Gap
* `train_set_file`: 训练集文件


也可以使用bash脚本：

```
sh train_batch_neg.sh
```



<a name="评估"></a>

## 6. 评估

效果评估分为 4 个步骤:

a. 获取Doc端Embedding
基于语义索引模型抽取出Doc样本库的文本向量，

b. 采用hnswlib对Doc端Embedding建库
使用 ANN 引擎构建索引库(这里基于 [hnswlib](https://github.com/nmslib/hnswlib) 进行 ANN 索引)

c. 获取Query的Embedding并查询相似结果
基于语义索引模型抽取出评估集 *Source Text* 的文本向量，在第 2 步中建立的索引库中进行 ANN 查询，召回 Top50 最相似的 *Target Text*, 产出评估集中 *Source Text* 的召回结果 `recall_result` 文件

d. 评估
基于评估集 `same_semantic.tsv` 和召回结果 `recall_result` 计算评估指标 R@10 和 R@50

运行如下命令进行 ANN 建库、召回，产出召回结果数据 `recall_result`

```
root_dir="checkpoints/train_0.001" 
corpus_num=280000
python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_10/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus_${corpus_num}.csv" 
```
参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `recall_result_dir`: 召回结果存储目录
* `recall_result_file`: 召回结果的文件名
* `params_path`： 待评估模型的参数文件名
* `hnsw_m`: hnsw 算法相关参数，保持默认即可
* `hnsw_ef`: hnsw 算法相关参数，保持默认即可
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `recall_num`: 对 1 个文本召回的相似文本数量
* `similar_text_pair`: 由相似文本对构成的评估集 semantic_similar_pair.tsv
* `corpus_file`: 召回库数据 corpus_file

也可以使用下面的bash脚本：

```
sh run_build_index.sh
```



成功运行结束后，会在 `./recall_result_dir/` 目录下产出 `recall_result.txt` 文件

接下来，运行如下命令进行效果评估，产出Recall@1, Recall@5, Recall@10, Recall@20 和 Recall@50 指标:
```
python -u evaluate.py \
        --similar_text_pair "data/test.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50
```
也可以使用下面的bash脚本：

```
bash evaluate.sh
```

参数含义说明
* `similar_text_pair`: 由相似文本对构成的评估集 semantic_similar_pair.tsv
* `recall_result_file`: 针对评估集中第一列文本 *Source Text* 的召回结果
* `recall_num`: 对 1 个文本召回的相似文本数量

<a name="预测"></a>

## 7. 预测

## 开始预测
我们可以基于语义索引模型抽取文本的语义向量或者计算文本 Pair 的语义相似度，我们以计算文本 Pair 的语义相似度为例:

### 准备预测数据
待预测数据为 tab 分隔的 tsv 文件，每一行为 1 个文本 Pair，部分示例如下:
```
西安下雪了？是不是很冷啊?       西安的天气怎么样啊？还在下雪吗？
第一次去见女朋友父母该如何表现？   第一次去见家长该怎么做
猪的护心肉怎么切            猪的护心肉怎么吃
显卡驱动安装不了，为什么？      显卡驱动安装不了怎么回事
```

### 开始预测
以上述 demo 数据为例，运行如下命令基于我们开源的 [In-batch negatives](https://arxiv.org/abs/2004.04906) 策略语义索引模型开始计算文本 Pair 的语义相似度:
```
python -u -m paddle.distributed.launch --gpus "0" \
    predict.py \
    --device gpu \
    --params_path "./checkpoints/batch_neg_v1.0.0/model_state.pdparams" \
    --output_emb_size 256
    --batch_size 128 \
    --max_seq_length 64 \
    --text_pair_file ${your_input_file}
```

参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `params_path`： 预训练模型的参数文件名
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `text_pair_file`: 由文本 Pair 构成的待预测数据集

<a name="部署"></a>

## 8. 部署

### 动转静导出

首先把动态图模型转换为静态图：

```
python export_model.py --params_path checkpoints/train_0.001/model_40/model_state.pdparams --output_path=./output
```
也可以运行下面的bash脚本：

```
sh export.sh
```

### Python服务


然后使用PaddleInference

```
python deploy/python/predict.py --model_dir=./output
```
也可以运行下面的bash脚本：

```
sh deploy.sh
```

## Reference

[1] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, Dense Passage Retrieval for Open-Domain Question Answering, Preprint 2020.
