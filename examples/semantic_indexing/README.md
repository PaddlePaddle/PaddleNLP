



# 语义索引模型库

语义索引技术是搜索引擎、推荐系统、广告系统在召回阶段的核心技术之一， 语义索引模型的效果直接决定了语义相关的物料能否被成功召回进入系统参与上层排序，从基础层面影响整个系统的效果。

[TOCM]

[TOC]

### 效果评估
|  模型 |  Recall@10 | Recall@50  |
| ------------ | ------------ | ------------ |
|  Baseline |  46.99 |  60.84 |
|  BatchNeg | 51.20(+4.21)  | 67.24(+6.4)  |
|   HardestNeg|  50.22(+3.23) |  65.17(+4.33) |

### 快速开始
快速基于 BatchNeg 策略和 HadestNeg 策略训练产出语义索引模型。
#### BatchNeg

##### 准备数据
BatchNeg 策略的训练数据是语义相似的文本 Pair 对，样例数据如下:

```
todo
```

您可以按照上述格式组织自己的训练数据，也可以`点击这里`下载我们基于开源语义相似度任务评测数据集构造生成的的训练数据 `semantic_pair_train.tsv`。

##### 开始训练
以我们提供的语义相似度训练集为例子，通过如下命令，指定 GPU 0,1,2,3 卡, 开始模型训练

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

##### 效果评估
语义索引模型的目标是: 给定输入文本，模型可以从海量文本库中快速、准确地召回一批语义相关文本。我们基于开源的语义相似度数据集构造了语义索引模型的评估集合与召回库数据。

###### 数据构造
1. 评估集:
我们抽取出开源语义相似度数据集(LCQMC、BQ_Corpus、PAW-X) 中的正例文本 Pair 作为评估集(共 1.4 万) 文件 `same_semantic.tsv` ，其中第 1 列文本作为输入模型的源文本(Source text)、第 2 列文本作为语义相似的目标文本(Target text)

2. 召回库:
抽取出开源语义相似度数据集(LCQMC、BQ_Corpus、PAW-X) 训练集中的所有句子和评估集中所有文本 Pair 的第 2 列(Target) 作为召回库(共 33.4 万)

###### 评估指标
我们采用 Recall@10 和 Recall@50 2 个指标来评估语义索引模型的效果

###### 开始评估

效果评估分为 3 个步骤:
1. ANN 建库: 基于语义索引模型抽取出召回库文本的向量之后，使用 ANN 引擎建索引库(目前我们采用的 ANN 算法是 HNSWLib)
2. 召回: 基于语义索引模型抽取出评估集 Source text 的向量，去第 1 步中建立的索引库中进行 ANN 查询召回 Top50 最相似的 Text, 产出评估集中 Source text 的召回结果 `recall_result` 文件
3. 评估： 基于评估集 `same_semantic.tsv` 和 召回结果 `recall_result` 计算评估指标 R@10 和 R@50


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

成功运行结束后，会在 ./recall_result_dir/ 目录下产出 recall_result.txt 文件，部分召回示例结果如下:
```
too
```


接下来，运行如下命令进行效果评估，产出 R@10 和 R@50 指标:
```
  python -u evaluate.py \
        --similar_pair_file "semantic_similar_pair.tsv" \
        --recall_result_file "./recall_result_dir/recall_result.txt" \
        --recall_num 50
```

成功运行结束后，会输出如下的评估指标:
```
todo
```

##### 开始预测
我们可以基于语义索引模型抽取文本的语义向量或者计算文本 Pair 的语义相似度，我们以计算文本 Pair 的语义相似度为例:

###### 准备数据
待预测数据为 tab 分隔的 tsv 文件，每一行为 1 个文本 Pair，部分示例如下:
```
Todo
```

###### 开始预测
运行如下命令开始计算文本 Pair 的语义相似度:
```
python -u -m paddle.distributed.launch --gpus "0" \
    predict.py \
    --device gpu \
    --params_path "./checkpoints/model_20000/model_state.pdparams" \
    --batch_size 128 \
    --max_seq_length 64 \
    --text_pair_file ${your_input_file}
```

#### HardestNeg

##### 开始训练
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
##### 效果评估
请参考
##### 开始预测
请参考


### 模型介绍

#### BatchNeg 策略

##### 核心思路
BatchNeg 策略的训练数据为语义相似的 Pair 对，如下所示
```
Todo
```
BatchNeg 策略将 batch  内除自身之外其它所有的 文本的相似文本作为自身的负例，例如: 上例中  
##### 训练目标
最大化正例的 Similarity,  最小化 Batch 内所有负例的 Similarity,  Loss function 一般采用交叉熵损失函数

#### HardestNeg 策略
##### 核心思路
HardestNeg 策略的训练数据也是语义相似的 Pair 对，如下所示
```
Todo
```
BatchNeg 策略将 batch  内除自身之外其它所有的文本的相似文本作为自身的负例，从所有负例中选择出模型最难区分的负样本参与训练，例如: 上例中

##### 训练目标
正例的 Similarity 比 Hardest 负样本的 Similarity 高出 1 个 margin, 一般采用 rank_marigin_loss

### Reference
