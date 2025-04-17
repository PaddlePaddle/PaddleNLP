# Gradient Cache 策略 [DPR](https://arxiv.org/abs/2004.04906)


### 实验结果

`Gradient Cache` 的实验结果如下，使用的评估指标是`Accuracy`：

|  DPR method | TOP-5  | TOP-10 | TOP-50| 说明 |
| :-----: | :----: | :----: | :----: | :---- |
|  Gradient_cache | 68.1 | 79.4| 86.2 | DPR 结合 GC 策略训练
| GC_Batch_size_512  | 67.3 | 79.6| 86.3| DPR 结合 GC 策略训练，且 batch_size 设置为512|

实验对应的超参数如下：

| Hyper Parameter | batch_size| learning_rate| warmup_steps| epoches| chunk_size|max_grad_norm |
| :----: | :----: | :----: | :----: | :---: | :----: | :----: |
| \ | 128/512| 2e-05 | 1237 | 40 | 2| 16/8 |

## 数据准备
我们使用 Dense Passage Retrieval 的[原始仓库](https://github.com/Elvisambition/DPR)
中提供的数据集进行训练和评估。可以使用[download_data.py](https://github.com/Elvisambition/DPR/blob/main/dpr/data/download_data.py)
脚本下载所需数据集。 数据集详细介绍见[原仓库](https://github.com/Elvisambition/DPR) 。

### 数据格式
```
[
  {
    "question": "....",
    "answers": ["...", "...", "..."],
    "positive_ctxs": [{
        "title": "...",
        "text": "...."
    }],
    "negative_ctxs": ["..."],
    "hard_negative_ctxs": ["..."]
  },
  ...
]
```

### 数据下载
在[原始仓库](https://github.com/Elvisambition/DPR)
下使用命令
```
python data/download_data.py --resource data.wikipedia_split.psgs_w100
python data/download_data.py --resource data.retriever.nq
python data/download_data.py --resource data.retriever.qas.nq
```
### 单独下载链接
[data.retriever.nq-train](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz)
[data.retriever.nq-dev](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz)
[data.retriever.qas.nq-dev](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv)
[data.retriever.qas.nq-test](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv)
[data.retriever.qas.nq-train](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv)
[psgs_w100.tsv](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz)


## 代码结构及说明
```
|—— train_gradient_cache_DPR.py # gradient_cache实现dense passage retrieval训练脚本
|—— train_gradient_cache.py # gradient_cache算法简单实现
|—— NQdataset.py # NQ数据集封装
|—— generate_dense_embeddings.py # 生成文本的稠密表示
|—— faiss_indexer.py # faiss相关indexer封装
|—— dense_retriever.py # 召回，指标检测
|—— qa_validation.py # 相关计算匹配函数
|—— tokenizers.py # tokenizer封装
```

## 模型训练
### 基于 [Dense Passage Retriever](https://arxiv.org/abs/2004.04906) 策略训练
```
python train_gradient_cache_DPR.py \
   --batch_size 128 \
   --learning_rate 2e-05 \
   --save_dir save_biencoder
   --warmup_steps 1237 \
   --epoches 40 \
   --max_grad_norm 2 \
   --train_data_path ./dataset_dir/biencoder-nq-train.json \
   --chunk_size 16 \
```

参数含义说明
* `batch_size`: 批次大小
* `learning_rate`: 学习率
* `save_dir`: 模型保存位置
* `warmupsteps`: 预热学习率参数
* `epoches`: 训练批次大小
* `max_grad_norm`: 详见 ClipGradByGlobalNorm
* `train_data_path`: 训练数据存放地址
* `chunk_size`: chunk 的大小

## 生成文章稠密向量表示

```
python generate_dense_embeddings.py \
   --ctx_file ./dataset_dir/psgs_w100.tsv \
   --out_file test_generate \
   --que_model_path ./save_dir/question_model_40 \
   --con_model_path ./save_dir/context_model_40
```


参数含义说明
* `ctx_file`: ctx 文件读取地址
* `out_file`: 生成后的文件输出地址
* `que_model_path`: question model path
* `con_model_path`： context model path


## 针对全部文档的检索器验证
```
python dense_retriever.py --hnsw_index \
    --out_file out_file \
    --encoded_ctx_file ./test_generate \
    --ctx_file ./dataset_dir/psgs_w100.tsv \
    --qa_file ./dataset_dir/nq.qa.csv \
    --que_model_path ./save_dir/question_model_40 \
    --con_model_path ./save_dir/context_model_40
```
参数含义说明
* `hnsw_index`：使用 hnsw_index
* `outfile`: 输出文件地址
* `encoded_ctx_file`: 编码后的 ctx 文件
* `ctx_file`: ctx 文件
* `qa_file`： qa_file 文件
* `que_model_path`: question encoder model
* `con_model_path`: context encoder model
