#语义索引 [DPR](https://arxiv.org/abs/2004.04906) 策略
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
   --warmup_steps 1237 \
   --epoches 40 \
   --max_grad_norm 2.0 \
   --train_data_path {data_path} \
   --chunk_size 40 \
```

参数含义说明
* `batch_size`: 批次大小
* `learning_rate`: 学习率
* `warmupsteps`： 预热学习率参数
* `epoches`: 训练批次大小
* `max_grad_norm`: 详见ClipGradByGlobalNorm
* `train_data_path`:训练数据存放地址
* `chunk_size`:chunk大小

##生成文章稠密向量表示

```
python generate_dense_embeddings.py \
   --model_file {path to biencoder} \
   --ctx_file {path to psgs_w100.tsv file} \
   --shard_id {shard_num, 0-based} --num_shards {total number of shards} \
   --out_file ${out files location + name PREFX}  \

```

参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `save_dir`: 模型存储路径
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `save_steps`： 模型存储 checkpoint 的间隔 steps 个数
* `margin`: 正样本相似度与负样本之间的目标 Gap
* `train_set_file`: 训练集文件  


##针对全部文档的检索器验证
```
python dense_retriever.py \
   --model_file {path to checkpoint file from step 1} \
   --ctx_file {path to psgs_w100.tsv file} \
   --qa_file {path to test/dev qas file} \
   --encoded_ctx_file "{glob expression for generated files from step 3}" \
   --out_file {path for output json files} \
   --n-docs 100 \
   --validation_workers 32 \
   --batch_size 64  
```
参数含义说明
* `device`: 使用 cpu/gpu 进行训练
* `save_dir`: 模型存储路径
* `output_emb_size`: Transformer 顶层输出的文本向量维度
* `save_steps`： 模型存储 checkpoint 的间隔 steps 个数
* `margin`: 正样本相似度与负样本之间的目标 Gap
* `train_set_file`: 训练集文件


