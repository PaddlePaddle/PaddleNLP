# Gradient Cache策略 [DPR](https://arxiv.org/abs/2004.04906) 策略
## 数据准备
我们使用Dense Passage Retrieval的[原始仓库](https://github.com/Elvisambition/DPR)
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

## 模型训练
### 基于 [Dense Passage Retriever](https://arxiv.org/abs/2004.04906) 策略训练
```
python train_dense_encoder.py \
   --batch_size 128 \
   --learning_rate 2e-05 \
   --save_dir save_biencoder
   --warmup_steps 1237 \
   --epoches 40 \
   --max_grad_norm 2 \
   --train_data_path {data_path} \
   --chunk_size 16 \
```

参数含义说明
* `batch_size`: 批次大小
* `learning_rate`: 学习率
* `save_dir`:模型保存位置
* `warmupsteps`： 预热学习率参数
* `epoches`: 训练批次大小
* `max_grad_norm`: 详见ClipGradByGlobalNorm
* `train_data_path`:训练数据存放地址
* `chunk_size`:chunk大小

## 生成文章稠密向量表示

```
python generate_dense_embeddings.py \
   --model_file {path to biencoder} \
   --ctx_file {path to psgs_w100.tsv file} \
   --shard_id {shard_num, 0-based} --num_shards {total number of shards} \
   --out_file ${out files location + name PREFX}  \
   --que_model_path {que_model_path} \
   --con_model_path {con_model_path}
```

## 如果只有一台机器，可以直接使用

```
python generate_dense_embedding \
   --ctx_file {data/psgs_w100.tsv} \
   --out_file {test_generate} \
   --que_model_path {que_model_path} \
   --con_model_path {con_model_path}
```


参数含义说明
* `ctx_file`: ctx文件读取地址
* `out_file`: 生成后的文件输出地址
* `que_model_path`: question model path
* `con_model_path`： context model path


## 针对全部文档的检索器验证
```
python dense_retriever.py --hnsw_index \
    --out_file {out_file} \
    --encoded_ctx_file {encoded_ctx} \
    --ctx_file {ctx} \
    --qa_file {nq.qa.csv} \
    --que_model_path {que_model_path} \
    --con_model_path {con_model_path}
```
参数含义说明
* `hnsw_index`：使用hnsw_index
* `outfile`: 输出文件地址
* `encoded_ctx_file`: 编码后的ctx文件
* `ctx_file`: ctx文件
* `qa_file`： qa_file文件
* `que_model_path`: question encoder model
* `con_model_path`: context encoder model
