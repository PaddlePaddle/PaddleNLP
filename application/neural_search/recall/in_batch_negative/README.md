# In-batch Negatives

 **目录**

* [背景介绍](#背景介绍)
* [In-batch Negatives](#In-batchNegatives)
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

在召回阶段，最常见的方式是通过双塔模型，学习Document(简写为Doc)的向量表示，对Doc端建立索引，用ANN召回。我们在这种方式的基础上，引入语义索引策略 [In-batch Negatives](https://arxiv.org/abs/2004.04906)，以如下Batch size=4的训练数据为例：


```
我手机丢了，我想换个手机     我想买个新手机，求推荐
求秋色之空漫画全集          求秋色之空全集漫画
学日语软件手机上的          手机学日语的软件
侠盗飞车罪恶都市怎样改车     侠盗飞车罪恶都市怎么改车
```

In-batch Negatives 策略的训练数据为语义相似的 Pair 对，策略核心是在 1 个 Batch 内同时基于 N 个负例进行梯度更新，将Batch 内除自身之外其它所有 Source Text 的相似文本 Target Text 作为负例，例如: 上例中“我手机丢了，我想换个手机” 有 1 个正例(”我想买个新手机，求推荐“)，3 个负例(1.求秋色之空全集漫画，2.手机学日语的软件，3.侠盗飞车罪恶都市怎么改车)。


<a name="In-batch Negatives"></a>

# In-batch Negatives 

<a name="技术方案"></a>

## 1. 技术方案和评估指标

### 技术方案

双塔模型，采用ERNIE1.0热启，在召回训练阶段引入In-batch Negatives  策略，使用hnswlib建立索引库，进行召回测试。


### 评估指标

采用 Recall@1，Recall@5 ，Recall@10 ，Recall@20  和 Recall@50 指标来评估语义索引模型的召回效果。

Recall@K召回率是指预测的前topK（top-k是指从最后的按得分排序的召回列表中返回前k个结果）结果中检索出的相关结果数和库中所有的相关结果数的比率，衡量的是检索系统的查全率。

**效果评估**

|  模型 |  Recall@1 | Recall@5 |Recall@10 |Recall@20 |Recall@50 |策略简要说明|
| ------------ | ------------ | ------------ |--------- |--------- |--------- |--------- |
|  In-batch Negatives |  51.301 | 65.309| 69.878| 73.996|78.881| Inbatch-negative有监督训练|



<a name="环境依赖"></a>

## 2. 环境依赖

推荐使用GPU进行训练，在预测阶段使用CPU或者GPU均可。

**环境依赖**
* python >= 3.6
* paddlepaddle >= 2.1.3
* paddlenlp >= 2.2
* [hnswlib](https://github.com/nmslib/hnswlib) >= 0.5.2
* visualdl >= 2.2.2

<a name="代码结构"></a>

## 3. 代码结构

```
|—— data.py # 数据读取、数据转换等预处理逻辑
|—— base_model.py # 语义索引模型基类
|—— train_batch_neg.py # In-batch Negatives 策略的训练主脚本
|—— batch_negative
    |—— model.py # In-batch Negatives 策略核心网络结构
|—— ann_util.py # Ann 建索引库相关函数


|—— recall.py # 基于训练好的语义索引模型，从召回库中召回给定文本的相似文本
|—— evaluate.py # 根据召回结果和评估集计算评估指标
|—— predict.py # 给定输入文件，计算文本 pair 的相似度
|—— export_model.py # 动态图转换成静态图
|—— scripts
    |—— export_model.sh  # 动态图转换成静态图脚本
    |—— predict.sh  # 预测bash版本
    |—— evaluate.sh # 评估bash版本
    |—— run_build_index.sh # 构建索引bash版本
    |—— train_batch_neg.sh  # 训练bash版本
|—— deploy
    |—— python
        |—— predict.py # PaddleInference
        |—— deploy.sh # Paddle Inference部署脚本
|—— inference.py # 动态图抽取向量

```

<a name="数据准备"></a>

## 4. 数据准备

### 数据集说明

我们基于某文献检索平台数据，构造面向语义索引的训练集、测试集、召回库。

**训练集** 和 **验证集** 格式一致，训练集4k条，测试集2w条，每行由一对语义相似的文本Pair构成，以tab符分割，第一列是检索query，第二列由相关文献标题（+关键词）构成。样例数据如下:

```
宁夏社区图书馆服务体系布局现状分析	       宁夏社区图书馆服务体系布局现状分析社区图书馆,社区图书馆服务,社区图书馆服务体系
人口老龄化对京津冀经济	                 京津冀人口老龄化对区域经济增长的影响京津冀,人口老龄化,区域经济增长,固定效应模型
英语广告中的模糊语	                  模糊语在英语广告中的应用及其功能模糊语,英语广告,表现形式,语用功能
甘氨酸二肽的合成	                      甘氨酸二肽合成中缩合剂的选择甘氨酸,缩合剂,二肽
```

**召回库** 用于模拟业务线上的全量语料库，评估模型的召回效果，计算相应的Recall指标。召回库总共30万条样本，每行由一列构成，文献标题（+关键词），样例数据如下：
```
陕西省贫困地区城乡青春期少女生长发育调查青春期,生长发育,贫困地区
五丈岩水库溢洪道加固工程中的新材料应用碳纤维布,粘钢加固技术,超细水泥,灌浆技术
木塑复合材料在儿童卫浴家具中的应用探索木塑复合材料,儿童,卫浴家具
泡沫铝准静态轴向压缩有限元仿真泡沫铝,准静态,轴向压缩,力学特性
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

<a name="模型训练"></a>

## 5. 模型训练

**语义索引训练模型下载链接：**

以下模型结构参数为: `TrasformerLayer:12, Hidden:768, Heads:12, OutputEmbSize: 256`

|Model|训练参数配置|硬件|MD5|
| ------------ | ------------ | ------------ |-----------|
|[batch_neg](https://bj.bcebos.com/v1/paddlenlp/models/inbatch_model.zip)|<div style="width: 150pt">margin:0.2 scale:30 epoch:3 lr:5E-5 bs:64 max_len:64 </div>|<div style="width: 100pt">4卡 v100-16g</div>|f3e5c7d7b0b718c2530c5e1b136b2d74|

### 训练环境说明


- NVIDIA Driver Version: 440.64.00 
- Ubuntu 16.04.6 LTS (Docker)
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz


### 单机单卡训练/单机多卡训练

这里采用单机多卡方式进行训练，通过如下命令，指定 GPU 0,1,2,3 卡, 基于 In-batch Negatives 策略训练模型，数据量比较小，几分钟就可以完成。如果采用单机单卡训练，只需要把`--gpus`参数设置成单卡的卡号即可。

如果使用CPU进行训练，则需要吧`--gpus`参数去除，然后吧`device`设置成cpu即可，详细请参考train_batch_neg.sh文件的训练设置

然后运行下面的命令使用GPU训练，得到语义索引模型：

```
root_path=recall
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    train_batch_neg.py \
    --device gpu \
    --save_dir ./checkpoints/${root_path} \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 3 \
    --output_emb_size 256 \
    --save_steps 10 \
    --max_seq_length 64 \
    --margin 0.2 \
    --train_set_file recall/train.csv 

```

参数含义说明

* `device`: 使用 cpu/gpu 进行训练
* `batch_size`: 训练的batch size的大小
* `learning_rate`: 训练的学习率的大小
* `epochs`: 训练的epoch数
* `save_dir`: 模型存储路径
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `save_steps`： 模型存储 checkpoint 的间隔 steps 个数
* `max_seq_length`: 输入序列的最大长度
* `margin`: 正样本相似度与负样本之间的目标 Gap
* `train_set_file`: 训练集文件


也可以使用bash脚本：

```
sh scripts/train_batch_neg.sh
```



<a name="评估"></a>

## 6. 评估

效果评估分为 4 个步骤:

a. 获取Doc端Embedding

基于语义索引模型抽取出Doc样本库的文本向量。

b. 采用hnswlib对Doc端Embedding建库

使用 ANN 引擎构建索引库(这里基于 [hnswlib](https://github.com/nmslib/hnswlib) 进行 ANN 索引)

c. 获取Query的Embedding并查询相似结果

基于语义索引模型抽取出评估集 *Source Text* 的文本向量，在第 2 步中建立的索引库中进行 ANN 查询，召回 Top50 最相似的 *Target Text*, 产出评估集中 *Source Text* 的召回结果 `recall_result` 文件。

d. 评估

基于评估集 `same_semantic.tsv` 和召回结果 `recall_result` 计算评估指标 Recall@k，其中k取值1，5，10，20，50。

运行如下命令进行 ANN 建库、召回，产出召回结果数据 `recall_result`

```
root_dir="checkpoints/inbatch" 
python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_40/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "recall/dev.csv" \
        --corpus_file "recall/corpus.csv" 
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
sh scripts/run_build_index.sh
```

run_build_index.sh还包含cpu和gpu运行的脚本，默认是gpu的脚本

成功运行结束后，会在 `./recall_result_dir/` 目录下产出 `recall_result.txt` 文件

```
热处理对尼龙6 及其与聚酰胺嵌段共聚物共混体系晶体熔融行为和结晶结构的影响        热处理对尼龙6及其与聚酰胺嵌段共聚物共混体系晶体熔融行为和结晶结构的影响尼龙6,聚酰胺嵌段共聚物,芳香聚酰胺,热处理      0.9831992387771606
热处理对尼龙6 及其与聚酰胺嵌段共聚物共混体系晶体熔融行为和结晶结构的影响        热处理方法对高强高模聚乙烯醇纤维性能的影响聚乙烯醇纤维,热处理,性能,热拉伸,热定型    0.8438636660575867
热处理对尼龙6 及其与聚酰胺嵌段共聚物共混体系晶体熔融行为和结晶结构的影响        制备工艺对PVC/ABS合金力学性能和维卡软化温度的影响PVC,ABS,正交试验,力学性能,维卡软化温度      0.8130228519439697
.....
```


接下来，运行如下命令进行效果评估，产出Recall@1, Recall@5, Recall@10, Recall@20 和 Recall@50 指标:
```
python -u evaluate.py \
        --similar_text_pair "recall/test.csv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50
```
也可以使用下面的bash脚本：

```
sh scripts/evaluate.sh
```

参数含义说明
* `similar_text_pair`: 由相似文本对构成的评估集 semantic_similar_pair.tsv
* `recall_result_file`: 针对评估集中第一列文本 *Source Text* 的召回结果
* `recall_num`: 对 1 个文本召回的相似文本数量

成功运行结束后，会输出如下评估指标:

```
recall@1=51.261
recall@5=65.279
recall@10=69.848
recall@20=73.971
recall@50=78.84
```

<a name="预测"></a>

## 7. 预测

我们可以基于语义索引模型预测文本的语义向量或者计算文本 Pair 的语义相似度。

### 7.1 功能一：抽取文本的语义向量

修改 inference.py 文件里面输入文本 id2corpus 和模型路径 params_path ：

```
params_path='checkpoints/inbatch/model_40/model_state.pdparams'
id2corpus={0:'国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据'}
```
然后运行：
```
python inference.py
```
预测结果为256维的向量：

```
[1, 256]
[[ 0.07766181 -0.13780491  0.03388524 -0.14910668 -0.0334941   0.06780092
   0.0104043   0.03168401  0.02605671  0.02088691  0.05520441 -0.0852212
   .....
```

### 7.2 功能二：计算文本 Pair 的语义相似度


### 准备预测数据

待预测数据为 tab 分隔的 csv 文件，每一行为 1 个文本 Pair，部分示例如下:
```
试论我国海岸带经济开发的问题与前景	试论我国海岸带经济开发的问题与前景海岸带,经济开发,问题,前景
外语阅读焦虑与英语成绩及性别的关系	外语阅读焦虑与英语成绩及性别的关系外语阅读焦虑,外语课堂焦虑,英语成绩,性别
数字图书馆	智能化图书馆
网络健康可信性研究	网络成瘾少年
```

### 开始预测

以上述 demo 数据为例，运行如下命令基于我们开源的 [In-batch Negatives](https://arxiv.org/abs/2004.04906) 策略语义索引模型开始计算文本 Pair 的语义相似度:
```
root_dir="checkpoints/inbatch" 

python -u -m paddle.distributed.launch --gpus "3" \
    predict.py \
    --device gpu \
    --params_path "${root_dir}/model_40/model_state.pdparams" \
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
predict.sh文件包含了cpu和gpu运行的脚本，默认是gpu运行的脚本

产出如下结果
```
0.9717282652854919
0.9371012449264526
0.7968897223472595
0.30377304553985596
```

<a name="部署"></a>

## 8. 部署

### 动转静导出

首先把动态图模型转换为静态图：

```
python export_model.py --params_path checkpoints/inbatch/model_40/model_state.pdparams --output_path=./output
```
也可以运行下面的bash脚本：

```
sh scripts/export_model.sh
```

### Paddle Inference预测

修改id2corpus的样本：

```
id2corpus={0:'国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据'}

```

然后使用PaddleInference

```
python deploy/python/predict.py --model_dir=./output
```
也可以运行下面的bash脚本：

```
sh deploy.sh
```
输出结果如下：

```
(1, 256)
[[-0.0394925  -0.04474756 -0.065534    0.00939134  0.04359895  0.14659195
  -0.0091779  -0.07303623  0.09413272 -0.01255222 -0.08685658  0.02762237
   0.10138468  0.00962821  0.10888419  0.04553023  0.05898942  0.00694253
   ....
```

## Reference

[1] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, Dense Passage Retrieval for Open-Domain Question Answering, Preprint 2020.
