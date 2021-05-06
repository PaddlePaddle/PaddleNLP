# 语义索引

语义索引技术是搜索引擎、推荐系统、广告系统在召回阶段的核心技术之一， 语义索引模型的效果直接决定了语义相关的物料能否被成功召回进入系统参与上层排序，从基础层面影响整个系统的效果。

## 效果评估
|  模型 |  Recall@10 | Recall@50  |
| ------------ | ------------ | ------------ |
|  Baseline |  46.99 |  60.84 |
|  BatchNeg | 51.20(**+4.21**)  | 67.24(**+6.4**)  |
|   HardestNeg|  50.22(**+3.23**) |  65.17(**+4.33**) |


## 语义索引预训练模型下载
以下模型结构参数为:
`TrasformerLayer:12, Hidden:768, Heads:12, OutputEmbSize: 256`

|Model|训练参数配置|MD5|
| ------------ | ------------ | ------------ |
|[batch_neg_v1.0](https://paddlenlp.bj.bcebos.com/models/semantic_index/batch_neg_v1.0.tar)|<div style="width: 200pt">margin:0.2 scale:30 epoch:3 lr:5E-5</div>|da1bb1487bd3fd6a53b8ef95c278f3e6|
|[hardest_neg_v1.0](https://paddlenlp.bj.bcebos.com/models/semantic_index/hardest_neg_v1.0.tar)|margin:0.2 epoch:3 lr:5E-5|b535d890110ea608c8562c525a0b84b5|


## 数据下载
|数据|描述|数量|MD5|
| ------------ | ------------ | ------------ | -------- |
|<div style="width: 180pt">[训练集(semantic_pair_train.tsv)](https://paddlenlp.bj.bcebos.com/models/semantic_index/semantic_pair_train.tsv)</div>|<div style="width: 220pt">每行为语义相似的文本 Pair 构成的训练集</div>|222546|590286f695200160350cc5838cb34f00|
|[评估集(same_semantic.tsv)](https://paddlenlp.bj.bcebos.com/models/semantic_index/same_semantic.tsv)|每行为语义相似文本 Pair 构成的评估集|10255|86ec1fd5234d944177574372dcf780c5|
|[召回库(corpus_file）](https://paddlenlp.bj.bcebos.com/models/semantic_index/corpus_file)|每行为单条文本构成的召回库|313714|a3fbc3421b5aeb939809876fc7beeaa8|

## 快速开始
快速基于 BatchNeg 策略和 HadestNeg 策略训练产出语义索引模型。
### BatchNeg

#### 准备训练数据
BatchNeg 策略的训练数据每一行由 `tab` 分隔的语义相似的文本 Pair 对，样例数据如下:

```
欢打篮球的男生喜欢什么样的女生    爱打篮球的男生喜欢什么样的女生
我手机丢了，我想换个手机        我想买个新手机，求推荐
求秋色之空漫画全集        求秋色之空全集漫画
学日语软件手机上的        手机学日语的软件
```

您可以按照上述格式组织自己的训练数据，或者下载我们基于开源语义相似度任务评测数据集([LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)、[BQ Corpus](http://icrc.hitsz.edu.cn/Article/show/175.html)、[PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx))构造生成的的训练数据 [semantic_pair_train.tsv](https://paddlenlp.bj.bcebos.com/models/semantic_index/semantic_pair_train.tsv)。

#### 开始训练
以我们提供的语义相似度训练数据为例，通过如下命令，指定 GPU 0,1,2,3 卡, 开始模型训练

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

#### 效果评估
语义索引模型的目标是: 给定输入文本，模型可以从海量候选召回库中快速、准确地召回一批语义相关文本。我们基于开源的语义相似度数据集构造了语义索引模型的评估集与召回库数据。

##### 评估数据
1. 评估集:
我们从开源语义相似度数据集([LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)、[BQ Corpus](http://icrc.hitsz.edu.cn/Article/show/175.html)、[PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx)) 的验证集中抽取出正例文本 Pair 生成评估集 [same_semantic.tsv](https://paddlenlp.bj.bcebos.com/models/semantic_index/same_semantic.tsv)，其中第 1 列文本作为输入模型的源文本 *Source Text*、第 2 列文本作为语义相似的目标文本 *Target Text*

2. 召回库:
抽取出开源语义相似度数据集([LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)、[BQ Corpus](http://icrc.hitsz.edu.cn/Article/show/175.html)、[PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx))训练集中的所有文本和验证集中所有文本 Pair 的第 2 列 *Target Text* 生成召回库 [corpus_file](https://paddlenlp.bj.bcebos.com/models/semantic_index/corpus_file)

##### 评估指标
采用 Recall@10 和 Recall@50 指标来评估语义索引模型的召回效果

##### 开始评估

效果评估分为 3 个步骤:
1. ANN 建库
首先基于语义索引模型抽取出召回库的文本向量，然后使用 ANN 引擎建索引库(当前基于 [hnswlib](https://github.com/nmslib/hnswlib) 进行 ANN 索引)

2. 召回
基于语义索引模型抽取出评估集 *Source Text* 的文本向量，在第 1 步中建立的索引库中进行 ANN 查询召回 Top50 最相似的 *Target Text*, 产出评估集中 *Source Text* 的召回结果 `recall_result` 文件

3. 评估： 基于评估集 [same_semantic.tsv](https://paddlenlp.bj.bcebos.com/models/semantic_index/same_semantic.tsv) 和召回结果 `recall_result` 计算评估指标 R@10 和 R@50

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

成功运行结束后，会输出如下评估指标, 分别为 R@10 和 R@50
```
51.2    67.242
```

#### 开始预测
我们可以基于语义索引模型抽取文本的语义向量或者计算文本 Pair 的语义相似度，我们以计算文本 Pair 的语义相似度为例:

##### 准备预测数据
待预测数据为 tab 分隔的 tsv 文件，每一行为 1 个文本 Pair，部分示例如下:
```
西安下雪了？是不是很冷啊?       西安的天气怎么样啊？还在下雪吗？
第一次去见女朋友父母该如何表现？   第一次去见家长该怎么做
猪的护心肉怎么切            猪的护心肉怎么吃
显卡驱动安装不了，为什么？      显卡驱动安装不了怎么回事
```

##### 开始预测
以上述 demo 数据为例，运行如下命令基于我们开源的 BatchNeg 策略语义索引模型开始计算文本 Pair 的语义相似度:
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
产出如下结果
```
0.8121148943901062
0.6034126281738281
0.968634843826294
0.9800204038619995
```


### HardestNeg

#### 开始训练
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
#### 效果评估
[请参考 BatchNeg 策略效果评估](####效果评估)
#### 开始预测
[请参考 BatchNeg 策略开始预测](####开始预测)


## 模型介绍

### BatchNeg 策略

#### 核心思路
BatchNeg 策略的训练数据为语义相似的 Pair 对，如下所示为 Batch size = 4 的训练数据样例:
```
我手机丢了，我想换个手机        我想买个新手机，求推荐
求秋色之空漫画全集          求秋色之空全集漫画
学日语软件手机上的          手机学日语的软件
侠盗飞车罪恶都市怎样改车        侠盗飞车罪恶都市怎么改车
```
BatchNeg 策略核心是在 1 个 Batch 内同时基于 N 个负例进行梯度更新，将Batch 内除自身之外其它所有 *Sorce Text* 的相似文本 *Target Text* 作为负例，例如: 上例中 `我手机丢了，我想换个手机` 有 1 个正例(`1.我想买个新手机，求推荐`)，3 个负例(`1.求秋色之空全集漫画`，`2.手机学日语的软件`，`3.侠盗飞车罪恶都市怎么改车`)。

#### 训练目标
最大化正例的 Similarity,  最小化 Batch 内所有负例的 Similarity,  Loss function 交叉熵损失函数。

### HardestNeg 策略
#### 核心思路
HardestNeg 策略核心是在 1 个 Batch 内的所有负样本中先挖掘出最难区分的负样本，基于最难负样本进行梯度更新。例如: 上例中 *Source Text*: `我手机丢了，我想换个手机` 有 3 个负例(`1.求秋色之空全集漫画`，`2.手机学日语的软件`，`3.侠盗飞车罪恶都市怎么改车`)，其中最难区分的负例是 `手机学日语的软件`，模型训练过程中不断挖掘出类似这样的最难负样本，然后基于最难负样本进行梯度更新。

#### 训练目标
正例的 Similarity 比 Hardest 负样本的 Similarity 高出 1 个 margin, 一般采用 rank_marigin_loss。

## Reference
[1] Xin Liu, Qingcai Chen, Chong Deng, Huajun Zeng, Jing Chen, Dongfang Li, Buzhou Tang, LCQMC: A Large-scale Chinese Question Matching Corpus,COLING2018.  
[2] Jing Chen, Qingcai Chen, Xin Liu, Haijun Yang, Daohe Lu, Buzhou Tang, The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification EMNLP2018.  
[3] Yang, Y., Zhang, Y., Tar, C., and Baldridge, J., “PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification”, <i>arXiv e-prints</i>, 2019.  
[4] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, Dense Passage Retrieval for Open-Domain Question Answering, Preprint 2020.
