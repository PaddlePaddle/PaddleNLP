
 **目录**

* [背景介绍](#背景介绍)
* [SimCSE](#SimCSE)
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

在召回阶段，最常见的方式是通过双塔模型，学习Document(简写为Doc)的向量表示，对Doc端建立索引，用ANN召回。我们在这种方式的基础上，引入无监督预训练策略，以如下训练数据为例：


```
我手机丢了，我想换个手机     我想买个新手机，求推荐
求秋色之空漫画全集          求秋色之空全集漫画
学日语软件手机上的          手机学日语的软件
侠盗飞车罪恶都市怎样改车     侠盗飞车罪恶都市怎么改车
```

SimCSE 模型适合缺乏监督数据，但是又有大量无监督数据的匹配和检索场景。


<a name="SimCSE"></a>

# SimCSE

<a name="技术方案"></a>

## 1. 技术方案和评估指标

### 技术方案

双塔模型，采用ERNIE1.0热启，在召回阶段引入 SimCSE 策略。


### 评估指标

（1）采用 Recall@1，Recall@5 ，Recall@10 ，Recall@20  和 Recall@50 指标来评估语义索引模型的召回效果。

**效果评估**

|  模型 |  Recall@1 | Recall@5 |Recall@10 |Recall@20 |Recall@50 |策略简要说明|
| ------------ | ------------ | ------------ |--------- |--------- |--------- |--------- |
|  SimCSE |  42.374 | 57.505| 62.641| 67.09|72.331| SimCSE无监督训练|


<a name="环境依赖"></a>

## 2. 环境依赖和安装说明

**环境依赖**
* python >= 3.6
* paddlepaddle >= 2.1.3
* paddlenlp >= 2.2
* [hnswlib](https://github.com/nmslib/hnswlib) >= 0.5.2
* visualdl >= 2.2.2



<a name="代码结构"></a>

## 3. 代码结构

以下是本项目主要代码结构及说明：

```
simcse/
├── model.py # SimCSE 模型组网代码
|—— deploy
    |—— python
        |—— predict.py # PaddleInference
        ├── deploy.sh # Paddle Inference的bash脚本
|—— scripts
    ├── export_model.sh # 动态图转静态图bash脚本
    ├── predict.sh # 预测的bash脚本
    ├── evaluate.sh # 召回评估bash脚本
    ├── run_build_index.sh  # 索引的构建脚本
    ├── train.sh # 训练的bash脚本
|—— ann_util.py # Ann 建索引库相关函数
├── data.py # 无监督语义匹配训练数据、测试数据的读取逻辑
├── export_model.py # 动态图转静态图
├── predict.py # 基于训练好的无监督语义匹配模型计算文本 Pair 相似度
├── evaluate.py # 根据召回结果和评估集计算评估指标
|—— inference.py # 动态图抽取向量
|—— recall.py # 基于训练好的语义索引模型，从召回库中召回给定文本的相似文本
└── train.py # SimCSE 模型训练、评估逻辑

```

<a name="数据准备"></a>

## 4. 数据准备

### 数据集说明

我们基于开源的语义匹配数据集构造生成了面向语义索引的训练集、评估集、召回库。

样例数据如下:
```
睡眠障碍与常见神经系统疾病的关系睡眠觉醒障碍,神经系统疾病,睡眠,快速眼运动,细胞增殖,阿尔茨海默病
城市道路交通流中观仿真研究
城市道路交通流中观仿真研究智能运输系统;城市交通管理;计算机仿真;城市道路;交通流;路径选择
网络健康可信性研究
网络健康可信性研究网络健康信息;可信性;评估模式
脑瘫患儿家庭复原力的影响因素及干预模式雏形 研究
脑瘫患儿家庭复原力的影响因素及干预模式雏形研究脑瘫患儿;家庭功能;干预模式
地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较
地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较
个案工作 社会化
个案社会工作介入社区矫正再社会化研究——以东莞市清溪镇为例社会工作者;社区矫正人员;再社会化;角色定位
圆周运动加速度角速度
圆周运动向心加速度物理意义的理论分析匀速圆周运动,向心加速度,物理意义,角速度,物理量,线速度,周期
```

召回集，验证集，测试集与inbatch-negative实验的数据保持一致


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
    ├── dev_pairwise.csv   # 排序验证集
    └── train_pairwise.csv  # 排序训练集

```

<a name="模型训练"></a>

## 5. 模型训练

**语义索引预训练模型下载链接：**

以下模型结构参数为: `TrasformerLayer:12, Hidden:768, Heads:12, OutputEmbSize: 256`

|Model|训练参数配置|硬件|MD5|
| ------------ | ------------ | ------------ |-----------|
|[SimCSE](https://bj.bcebos.com/v1/paddlenlp/models/simcse_model.zip)|<div style="width: 150pt">epoch:3 lr:5E-5 bs:64 max_len:64 </div>|<div style="width: 100pt">4卡 v100-16g</div>|7c46d9b15a214292e3897c0eb70d0c9f|

### 训练环境说明

+ NVIDIA Driver Version: 440.64.00
+ Ubuntu 16.04.6 LTS (Docker)
+ Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz


### 单机单卡训练/单机多卡训练

这里采用单机多卡方式进行训练，通过如下命令，指定 GPU 0,1,2,3 卡, 基于SimCSE训练模型，无监督的数据量比较大，4卡的训练的时长在16个小时左右。如果采用单机单卡训练，只需要把`--gpu`参数设置成单卡的卡号即可。

训练的命令如下：

```shell
$ unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus '0,1,2,3' \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 3 \
	--save_steps 2000 \
	--eval_steps 100 \
	--max_seq_length 64 \
	--infer_with_fc_pooler \
	--dropout 0.2 \
    --output_emb_size 256 \
	--train_set_file "./recall/train_unsupervised.csv" \
	--test_set_file "./recall/dev.csv"
```
也可以使用bash脚本：

```
sh scripts/train.sh
```



可支持配置的参数：

* `infer_with_fc_pooler`：可选，在预测阶段计算文本 embedding 表示的时候网络前向是否会过训练阶段最后一层的 fc;  建议打开模型效果最好。
* `scale`：可选，在计算 cross_entropy loss 之前对 cosine 相似度进行缩放的因子；默认为 20。
* `dropout`：可选，SimCSE 网络前向使用的 dropout 取值；默认 0.1。
* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `max_seq_length`：可选，ERNIE-Gram 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `epochs`: 训练轮次，默认为1。
* `warmup_proption`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── model_100
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
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

基于评估集 `dev.csv` 和召回结果 `recall_result` 计算评估指标 Recall@k，其中k取值1，5，10，20，50.

运行如下命令进行 ANN 建库、召回，产出召回结果数据 `recall_result`

```
python -u -m paddle.distributed.launch --gpus "6" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "checkpoints/model_20000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "recall/dev.csv" \
        --corpus_file "recall/corpus.csv"
```
也可以使用下面的bash脚本：

```
sh scripts/run_build_index.sh
```

run_build_index.sh还包含cpu和gpu运行的脚本，默认是gpu的脚本


接下来，运行如下命令进行效果评估，产出Recall@1, Recall@5, Recall@10, Recall@20 和 Recall@50 指标:
```
python -u evaluate.py \
        --similar_text_pair "recall/dev.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50
```
也可以使用下面的bash脚本：

```
bash scripts/evaluate.sh
```

参数含义说明
* `similar_text_pair`: 由相似文本对构成的评估集
* `recall_result_file`: 针对评估集中第一列文本 *Source Text* 的召回结果
* `recall_num`: 对 1 个文本召回的相似文本数量

成功运行结束后，会输出如下评估指标:

```
recall@1=45.183
recall@5=60.444
recall@10=65.224
recall@20=69.562
recall@50=74.848
```


<a name="预测"></a>

## 7. 预测

我们可以基于语义索引模型预测文本的语义向量或者计算文本 Pair 的语义相似度。

### 7.1 功能一：抽取文本的语义向量

修改 inference.py 文件里面输入文本 id2corpus 和模型路径 params_path:

```
params_path='checkpoints/model_20000/model_state.pdparams'
id2corpus={0:'国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据'}
```
然后运行
```
python inference.py
```
预测结果位256维的向量：

```
[1, 256]
[[-6.70653954e-02 -6.46878220e-03 -6.78317016e-03  1.66617986e-02
   7.20006675e-02 -9.79134627e-03 -1.38441555e-03  4.37440760e-02
   4.78116237e-02  1.33881181e-01  1.82927232e-02  3.23656350e-02
   ...
```

### 7.2 功能二：计算文本 Pair 的语义相似度

### 准备预测数据

待预测数据为 tab 分隔的 tsv 文件，每一行为 1 个文本 Pair，部分示例如下:
```
热处理对尼龙6 及其与聚酰胺嵌段共聚物共混体系晶体熔融行为和结晶结构的影响        热处理对尼龙6及其与聚酰胺嵌段共聚物共混体系晶体熔融行为和结晶结构的影响尼龙6,聚酰胺嵌段共聚物,芳香聚酰胺,热处理
面向生态系统服务的生态系统分类方案研发与应用.   面向生态系统服务的生态系统分类方案研发与应用
huntington舞蹈病的动物模型      Huntington舞蹈病的动物模型
试论我国海岸带经济开发的问题与前景      试论我国海岸带经济开发的问题与前景海岸带,经济开发,问题,前景
```

### 开始预测

以上述 demo 数据为例，运行如下命令基于我们开源的 SimCSE无监督语义索引模型开始计算文本 Pair 的语义相似度:
```
root_dir="checkpoints"

python -u -m paddle.distributed.launch --gpus "3" \
    predict.py \
    --device gpu \
    --params_path "${root_dir}/model_20000/model_state.pdparams" \
    --output_emb_size 256 \
    --batch_size 128 \
    --max_seq_length 64 \
    --text_pair_file "recall/test.csv"
```

参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `params_path`： 预训练模型的参数文件名
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `text_pair_file`: 由文本 Pair 构成的待预测数据集

也可以运行下面的bash脚本：

```
sh scripts/predict.sh
```

产出如下结果
```
0.6477588415145874
0.9698382019996643
1.0
0.1787596344947815
```

<a name="部署"></a>

## 8. 部署

### 动转静导出

首先把动态图模型转换为静态图：

```
python export_model.py --params_path checkpoints/model_20000/model_state.pdparams --output_path=./output
```
也可以运行下面的bash脚本：

```
sh scripts/export_model.sh
```

### Paddle Inference预测

预测既可以抽取向量也可以计算两个文本的相似度。

修改id2corpus的样本：

```
# 抽取向量
id2corpus={0:'国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据'}
# 计算相似度
corpus_list=[['中西方语言与文化的差异','中西方文化差异以及语言体现中西方文化,差异,语言体现'],
                    ['中西方语言与文化的差异','飞桨致力于让深度学习技术的创新与应用更简单']]

```
然后使用PaddleInference

```
python deploy/python/predict.py --model_dir=./output
```
也可以运行下面的bash脚本：

```
sh deploy.sh
```
最终输出的是256维度的特征向量和句子对的预测概率

```
(1, 256)
[[-6.70653731e-02 -6.46873191e-03 -6.78317575e-03  1.66618153e-02
   7.20006898e-02 -9.79136024e-03 -1.38439541e-03  4.37440872e-02
   4.78115827e-02  1.33881137e-01  1.82927139e-02  3.23656537e-02
   .......

[0.5649663209915161, 0.03284594044089317]
```
## FAQ

#### SimCSE模型怎么部署？

+ SimCSE使用的模型跟 In-batch Negatives 训练出来的模型网络结构是一样的，使用 In-batch Negatives 的部署流程即可，参考[In-batch Negatives](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/recall/in_batch_negative/deploy/python)

## Reference
[1] Gao, Tianyu, Xingcheng Yao, and Danqi Chen. “SimCSE: Simple Contrastive Learning of Sentence Embeddings.” ArXiv:2104.08821 [Cs], April 18, 2021. http://arxiv.org/abs/2104.08821.
