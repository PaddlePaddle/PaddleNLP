# In-batch negatives

## 数据准备

### 数据生成
我们基于万方的语义匹配数据集构造生成了面向语义索引的训练集、评估集、召回库。

样例数据如下:
```
欢打篮球的男生喜欢什么样的女生   爱打篮球的男生喜欢什么样的女生
我手机丢了，我想换个手机        我想买个新手机，求推荐
求秋色之空漫画全集             求秋色之空全集漫画
学日语软件手机上的             手机学日语的软件
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


## 项目依赖:
- [hnswlib](https://github.com/nmslib/hnswlib)

## 代码结构及说明
```
|—— train_batch_neg.py # In-batch negatives 策略的训练主脚本
|—— batch_negative
    |—— model.py # In-batch negatives 策略核心网络结构
|—— ann_util.py # Ann 建索引库相关函数
|—— base_model.py # 语义索引模型基类
|—— data.py # 数据读取、数据转换等预处理逻辑
|—— evaluate.py # 根据召回结果和评估集计算评估指标
|—— predict.py # 给定输入文件，计算文本 pair 的相似度
|—— recall.py # 基于训练好的语义索引模型，从召回库中召回给定文本的相似文本
```

## 模型训练

基于 In-batch negatives 策略开始训练模型

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

也可以使用bash脚本：

```
sh train_batch_neg.sh
```

参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `save_dir`: 模型存储路径
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `save_steps`： 模型存储 checkpoint 的间隔 steps 个数
* `margin`: 正样本相似度与负样本之间的目标 Gap
* `train_set_file`: 训练集文件




### 开始评估
效果评估分为 3 个步骤:
1. ANN 建库
首先基于语义索引模型抽取出召回库的文本向量，然后使用 ANN 引擎建索引库(当前基于 [hnswlib](https://github.com/nmslib/hnswlib) 进行 ANN 索引)

2. 召回
基于语义索引模型抽取出评估集 *Source Text* 的文本向量，在第 1 步中建立的索引库中进行 ANN 查询召回 Top50 最相似的 *Target Text*, 产出评估集中 *Source Text* 的召回结果 `recall_result` 文件

3. 评估： 基于评估集和召回结果 `recall_result` 计算评估指标 Recall@1, Recall@5, Recall@10, Recall@20 和 Recall@50

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
也可以使用下面的bash脚本：

```
sh run_build_index.sh
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

## PaddleInference

首先把动态图模型转换为静态图：

```
python export_model.py --params_path checkpoints/train_0.001/model_40/model_state.pdparams --output_path=./output
```
也可以运行下面的bash脚本：

```
sh export.sh
```
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
