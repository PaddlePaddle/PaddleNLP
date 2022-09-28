# 语义索引

语义索引技术是搜索引擎、推荐系统、广告系统在召回阶段的核心技术之一， 语义索引模型的效果直接决定了语义相关的物料能否被成功召回进入系统参与上层排序，从基础层面影响整个系统的效果。

语义索引库提供了前沿语义索引策略的训练、语义索引模型的效果评估方案、支持用户基于我们开源的语义索引模型进行文本 Pair 的相似度计算或者 Embedding 语义表示抽取。

我们基于 ERNIE1.0 热启，分别采用 [In-batch negatives](https://arxiv.org/abs/2004.04906) 策略和 HardestNeg 策略开源了 [batch_neg_v1.0](https://bj.bcebos.com/paddlenlp/models/semantic_index/batch_neg_v1.0.tar) 和 [hardest_neg_v1.0](https://bj.bcebos.com/paddlenlp/models/semantic_index/hardest_neg_v1.0.tar) 模型，相比 Baseline 模型效果有显著提升:

## 效果评估
|  模型 |  Recall@10 | Recall@50  |策略简要说明|
| ------------ | ------------ | ------------ |--------- |
|  Baseline |  46.99 |  60.84 | 标准 pair-wise 训练范式，通过随机采样产生负样本|
|  [In-batch negatives](https://arxiv.org/abs/2004.04906) | 51.20(**+4.21**)  | 67.24(**+6.4**)  | 在 Batch 内同时使用 batch_size 个负样本进行训练|
|  HardestNeg | 50.22(**+3.23**) |  65.17(**+4.33**) |<div style="width: 340pt"> 在 Batch 内先挖掘最难负样本，然后进行 pair-wise 训练</div>|


## 语义索引预训练模型下载
以下模型结构参数为:
`TrasformerLayer:12, Hidden:768, Heads:12, OutputEmbSize: 256`

|Model|训练参数配置|硬件|MD5|
| ------------ | ------------ | ------------ |-----------|
|[batch_neg_v1.0](https://bj.bcebos.com/paddlenlp/models/semantic_index/batch_neg_v1.0.tar)|<div style="width: 150pt">margin:0.2 scale:30 epoch:3 lr:5E-5 bs:128 max_len:64 </div>|<div style="width: 100pt">单卡v100-16g</div>|da1bb1487bd3fd6a53b8ef95c278f3e6|
|[hardest_neg_v1.0](https://bj.bcebos.com/paddlenlp/models/semantic_index/hardest_neg_v1.0.tar)|margin:0.2 epoch:3 lr:5E-5 bs:128 max_len:64 |单卡v100-16g|b535d890110ea608c8562c525a0b84b5|


## 数据准备
### 数据生成
我们基于开源语义匹配数据集构造生成了面向语义索引的训练集、评估集、召回库。
#### 构造训练集
从开源语义相似度任务评测数据集([LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)、[BQ Corpus](http://icrc.hitsz.edu.cn/Article/show/175.html)、[PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx))的训练集和测试集中抽取出所有语义相似的文本 Pair 作为训练集 [semantic_pair_train.tsv](https://bj.bcebos.com/paddlenlp/models/semantic_index/semantic_pair_train.tsv)。

[In-batch negatives](https://arxiv.org/abs/2004.04906) 策略和 HardestNeg 策略训练数据每一行由 `tab` 分隔的语义相似的文本 Pair 对，样例数据如下:
```
欢打篮球的男生喜欢什么样的女生   爱打篮球的男生喜欢什么样的女生
我手机丢了，我想换个手机        我想买个新手机，求推荐
求秋色之空漫画全集             求秋色之空全集漫画
学日语软件手机上的             手机学日语的软件
```


#### 构造评估集
从开源语义相似度数据集([LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)、[BQ Corpus](http://icrc.hitsz.edu.cn/Article/show/175.html)、[PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx)) 的验证集中抽取出正例文本 Pair 生成评估集 [same_semantic.tsv](https://bj.bcebos.com/paddlenlp/models/semantic_index/same_semantic.tsv)，其中第 1 列文本作为输入模型的源文本 *Source Text*、第 2 列文本作为语义相似的目标文本 *Target Text*。

#### 构造召回库
抽取出开源语义相似度数据集([LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)、[BQ Corpus](http://icrc.hitsz.edu.cn/Article/show/175.html)、[PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx))训练集中的所有文本和验证集中所有文本 Pair 的第 2 列 *Target Text* 生成召回库 [corpus_file](https://bj.bcebos.com/paddlenlp/models/semantic_index/corpus_file)


### 数据下载
|数据|描述|数量|MD5|
| ------------ | ------------ | ------------ | -------- |
|<div style="width: 180pt">[训练集(semantic_pair_train.tsv)](https://bj.bcebos.com/paddlenlp/models/semantic_index/semantic_pair_train.tsv)</div>|<div style="width: 220pt">每行为语义相似的文本 Pair 构成的训练集</div>|222546|590286f695200160350cc5838cb34f00|
|[评估集(same_semantic.tsv)](https://bj.bcebos.com/paddlenlp/models/semantic_index/same_semantic.tsv)|每行为语义相似文本 Pair 构成的评估集|10255|86ec1fd5234d944177574372dcf780c5|
|[召回库(corpus_file）](https://bj.bcebos.com/paddlenlp/models/semantic_index/corpus_file)|每行为单条文本构成的召回库|313714|a3fbc3421b5aeb939809876fc7beeaa8|


## 项目依赖:
- [hnswlib](https://github.com/nmslib/hnswlib)

## 代码结构及说明
```
|—— train_batch_neg.py # In-batch negatives 策略的训练主脚本
|—— train_hardest_neg.py # HardestNeg 策略的训练主脚本
|—— batch_negative
    |—— model.py # In-batch negatives 策略核心网络结构
|——hardest_negative
    |—— model.py # HardestNeg 策略核心网络结构
|—— ann_util.py # Ann 建索引库相关函数
|—— base_model.py # 语义索引模型基类
|—— data.py # 数据读取、数据转换等预处理逻辑
|—— evaluate.py # 根据召回结果和评估集计算评估指标
|—— predict.py # 给定输入文件，计算文本 pair 的相似度
|—— recall.py # 基于训练好的语义索引模型，从召回库中召回给定文本的相似文本
```

## 模型训练
### 基于 [In-batch negatives](https://arxiv.org/abs/2004.04906) 策略训练
以我们提供的语义相似度训练数据为例，通过如下命令，指定 GPU 0,1,2,3 卡, 基于 In-batch negatives 策略开始训练模型

```
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    train_batch_neg.py \
    --device gpu \
    --save_dir ./checkpoints/ \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 3 \
    --output_emb_size 256 \
    --save_steps 500 \
    --max_seq_length 64 \
    --margin 0.2 \
    --train_set_file semantic_pair_train.tsv \
```

参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `save_dir`: 模型存储路径
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `save_steps`： 模型存储 checkpoint 的间隔 steps 个数
* `margin`: 正样本相似度与负样本之间的目标 Gap
* `train_set_file`: 训练集文件


### 基于 HardestNeg 策略训练
以我们提供的语义相似度训练集为例子，通过如下命令，指定 GPU 0,1,2,3 卡, 开始模型训练

```
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    train_hardest_neg.py \
    --device gpu \
    --save_dir ./checkpoints/ \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 3 \
    --output_emb_size 256 \
    --save_steps 500 \
    --max_seq_length 64 \
    --margin 0.2 \
    --train_set_file semantic_pair_train.tsv \

```

## 效果评估
语义索引模型的目标是: 给定输入文本，模型可以从海量候选召回库中快速、准确地召回一批语义相关文本。

### 评估指标
采用 Recall@10 和 Recall@50 指标来评估语义索引模型的召回效果

### 开始评估
效果评估分为 3 个步骤:
1. ANN 建库
首先基于语义索引模型抽取出召回库的文本向量，然后使用 ANN 引擎建索引库(当前基于 [hnswlib](https://github.com/nmslib/hnswlib) 进行 ANN 索引)

2. 召回
基于语义索引模型抽取出评估集 *Source Text* 的文本向量，在第 1 步中建立的索引库中进行 ANN 查询召回 Top50 最相似的 *Target Text*, 产出评估集中 *Source Text* 的召回结果 `recall_result` 文件

3. 评估： 基于评估集 [same_semantic.tsv](https://bj.bcebos.com/paddlenlp/models/semantic_index/same_semantic.tsv) 和召回结果 `recall_result` 计算评估指标 R@10 和 R@50

运行如下命令进行 ANN 建库、召回，产出召回结果数据 `recall_result`

```
python -u -m paddle.distributed.launch --gpus "0" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${checkpoints_params_file}" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "semantic_similar_pair.tsv" \
        --corpus_file "corpus_file" \
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

成功运行结束后，会在 `./recall_result_dir/` 目录下产出 `recall_result.txt` 文件，部分召回示例结果如下:
```
开初婚未育证明怎么弄？  初婚未育证明怎么开？        0.9878678917884827
开初婚未育证明怎么弄？  初婚未育情况证明怎么开？    0.955365777015686
开初婚未育证明怎么弄？  初婚未育证明在哪里办理      0.9508345723152161
开初婚未育证明怎么弄？  到哪里开初婚未育证明？      0.949864387512207
```

接下来，运行如下命令进行效果评估，产出 R@10 和 R@50 指标:
```
  python -u evaluate.py \
        --similar_pair_file "semantic_similar_pair.tsv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50
```

参数含义说明
* `similar_text_pair`: 由相似文本对构成的评估集 semantic_similar_pair.tsv
* `recall_result_file`: 针对评估集中第一列文本 *Source Text* 的召回结果
* `recall_num`: 对 1 个文本召回的相似文本数量

成功运行结束后，会输出如下评估指标, 分别为 R@10 和 R@50
```
51.2    67.242
```

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

产出如下结果
```
0.8121148943901062
0.6034126281738281
0.968634843826294
0.9800204038619995
```

## 使用 FasterTransformer 加速预测

我们基于 Paddle 自定义算子功能集成了[NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer) 的高性能加速能力，通过简单易用的 Python API 即可得到 GPU 上更高性能预测能力。
- FT FP32 相比 Paddle 前向加速比为 1.13 ~ 4.36
- FT FP16 相比 Paddle 前向加速比为 3.65 ~ 5.42
- 支持 Post-Normalization 和 Pre-Normalizaiton 2 种 Transformer 结构
- 支持 GELU 和 RELU 2 个激活函数

详细性能评测数据如下表:

| batch size | max_seq_len | Paddle 前向(ms)|FT FP32(ms)  | FT FP16(ms) |Speedup(FT FP32/Paddle)|Speedup(FT FP16/Paddle)|
| ---------- | ----------- | ------------------- | ------------------- |------------------ |------------------ |------------------ |
| 16         | 16          | 23.56  |  5.40 | 5.38 | 4.36| 4.38|
| 16         | 32          | 22.34  |  8.11  | 5.57|2.75|4.01|
| 16         | 64          | 22.79   | 14.84  |5.39|1.54|4.23|
| 32         | 16          | 23.41      | 8.16   |5.30|2.87|4.42|
| 32         | 32          | 22.67      | 14.84   |6.21|1.53|3.65|
| 32         | 64          | 33.49 | 28.53   |6.05|1.17|5.54|
| 64         | 16          | 22.60  | 14.81   |5.59|1.53|4.04|
| 64         | 32          | 33.52  | 28.22   |6.24|1.19|5.37|
| 64         | 64          | 62.62  | 55.25   |11.55|1.13|5.42|

Note: 测试环境如下
```
硬件: NVIDIA Tesla V100 16G 单卡
Paddle Version: 2.2.1
CUDA: 10.1
cuDNN: 7.6
```

可参考如下命令使用高性能预测能力
```shell
python -u -m paddle.distributed.launch --gpus "0" faster_predict.py \
   --params_path "batch_neg_v1.0/model_state.pdparams"   \
   --output_emb_size 256   \
   --batch_size 32  \
   --max_seq_length 64  \
   --use_fp16 \
   --text_pair_file ${your_input_file} \
```

## 模型介绍
简要介绍 In-batch negatives 策略和 HardestNeg 策略思路

### [In-batch negatives](https://arxiv.org/abs/2004.04906) 核心思路

In-batch negatives 策略的训练数据为语义相似的 Pair 对，如下所示为 Batch size = 4 的训练数据样例:
```
我手机丢了，我想换个手机     我想买个新手机，求推荐
求秋色之空漫画全集          求秋色之空全集漫画
学日语软件手机上的          手机学日语的软件
侠盗飞车罪恶都市怎样改车     侠盗飞车罪恶都市怎么改车
```
In-batch negatives 策略核心是在 1 个 Batch 内同时基于 N 个负例进行梯度更新，将Batch 内除自身之外其它所有 *Source Text* 的相似文本 *Target Text* 作为负例，例如: 上例中 `我手机丢了，我想换个手机` 有 1 个正例(`1.我想买个新手机，求推荐`)，3 个负例(`1.求秋色之空全集漫画`，`2.手机学日语的软件`，`3.侠盗飞车罪恶都市怎么改车`)。

### HardestNeg 核心思路
HardestNeg 策略核心是在 1 个 Batch 内的所有负样本中先挖掘出最难区分的负样本，基于最难负样本进行梯度更新。例如: 上例中 *Source Text*: `我手机丢了，我想换个手机` 有 3 个负例(`1.求秋色之空全集漫画`，`2.手机学日语的软件`，`3.侠盗飞车罪恶都市怎么改车`)，其中最难区分的负例是 `手机学日语的软件`，模型训练过程中不断挖掘出类似这样的最难负样本，然后基于最难负样本进行梯度更新。

## Reference
[1] Xin Liu, Qingcai Chen, Chong Deng, Huajun Zeng, Jing Chen, Dongfang Li, Buzhou Tang, LCQMC: A Large-scale Chinese Question Matching Corpus,COLING2018.
[2] Jing Chen, Qingcai Chen, Xin Liu, Haijun Yang, Daohe Lu, Buzhou Tang, The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification EMNLP2018.
[3] Yang, Y., Zhang, Y., Tar, C., and Baldridge, J., “PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification”, <i>arXiv e-prints</i>, 2019.
[4] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, Dense Passage Retrieval for Open-Domain Question Answering, Preprint 2020.
