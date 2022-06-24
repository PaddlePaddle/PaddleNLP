# 无监督语义匹配模型 [DiffCSE](https://arxiv.org/pdf/2204.10298.pdf)

我们借鉴 DiffCSE 论文的关键思想，实现了 DiffCSE 模型。相比于 SimCSE 模型，DiffCSE模型会更关注语句之间的差异性，具有精确的向量表示能力。DiffCSE 模型同样适合缺乏监督数据，但是又有大量无监督数据的匹配和检索场景。

## 快速开始
### 代码结构说明

以下是本项目主要代码结构及说明：

```
DiffCSE/
├── model.py # DiffCSE 模型组网代码
├── custom_ernie.py # 为适配DiffCSE模型，对ERNIE模型进行了部分修改
├── data.py # 无监督语义匹配训练数据、测试数据的读取逻辑
├── run_diffcse.py # 模型训练、评估、预测的主脚本
├── eval_metrics.py # 模型测试用的指标计算脚本，包括 spearman, Precison, Recall 等指标
├── utils.py # 包括一些常用的工具式函数
├── run_train.sh # 模型训练的脚本
├── run_eval.sh # 模型评估的脚本
└── run_infer.sh # 模型预测的脚本
```

### 模型训练
可以运行如下命令，开始模型训练并且进行模型测试。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u -m paddle.distributed.launch --gpus ${gpu_ids} \
	run_diffcse.py \
	--mode "train" \
	--extractor_name "rocketqa-zh-dureader-query-encoder" \
	--generator_name "ernie-1.0" \
	--discriminator_name "ernie-1.0" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--train_set_file ./data/train.txt \
	--eval_set_file ./data/test_v1.txt \
	--save_dir ./checkpoints \
	--save_steps "10000" \
	--eval_steps "1000" \
	--batch_size "32" \
	--epochs "3" \
	--learning_rate "3e-5" \
	--weight_decay "0.01" \
	--warmup_proportion "0.01" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--seed "0" \
	--device "gpu"
```

可支持配置的参数：
* `mode`：可选，用于指明本次运行是模型训练、模型评估还是模型预测，仅支持[train, eval, infer]三种模式；默认为 infer。
* `extractor_name`：可选，DiffCSE模型中用于向量抽取的模型名称；默认为 ernie-1.0。
* `generator_name`: 可选，DiffCSE模型中生成器的模型名称；默认为 ernie-3.0-base-zh。
* `discriminator_name`: 可选，DiffCSE模型中判别器的模型名称；默认为 rocketqa-zh-dureader-query-encoder。
* `epochs`: 模型训练轮次，默认为3。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune 的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.01。
* `warmup_proption`：可选，学习率 warmup 策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到 learning_rate, 而后再缓慢衰减，默认为0.01。
* `max_seq_length`：可选，ERNIE-Gram 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `output_emb_size`：可选，向量抽取模型输出向量的维度；默认为32。
* `dropout`：可选，DiffCSE 网络前向使用的 dropout 取值；默认 0.1。
* `margin`：可选，用于计算损失时，保持正例相对于负例的 margin 值；默认 0.0.
* `scale`：可选，在计算 cross_entropy loss 之前对 cosine 相似度进行缩放的因子；默认为 20。
* `train_set_file`：可选，用于指定训练集的路径。
* `eval_set_file`：可选，用于指定验证集的路径。
* `save_dir`：可选，保存训练模型的目录；
* `save_steps`：可选，用于指定模型训练过程中每隔多少 step 保存一次模型。
* `eval_steps`：可选，用于指定模型训练过程中每隔多少 step，使用验证集评估一次模型。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选 cpu 或 gpu。如使用 gpu 训练则参数 gpus 指定GPU卡号。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── model_10000
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
└── ...
```

### 模型评估
可以运行如下命令，进行模型评估。

```shell
export CUDA_VISIBLE_DEVICES=0

python run_diffcse.py \
	--mode "eval" \
	--extractor_name "rocketqa-zh-dureader-query-encoder" \
	--generator_name "ernie-1.0" \
	--discriminator_name "ernie-1.0" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--eval_set_file ./data/test_v1.txt \
	--ckpt_dir ./checkpoints/checkpoint_10000 \
	--batch_size "8" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--dup_rate "0.0" \
	--seed "0" \
	--device "gpu"
```


### 基于动态图模型预测

测试数据示例如下，：
```text
谁有狂三这张高清的  这张高清图，谁有
英雄联盟什么英雄最好    英雄联盟最好英雄是什么
这是什么意思，被蹭网吗  我也是醉了，这是什么意思
现在有什么动画片好看呢？    现在有什么好看的动画片吗？
请问晶达电子厂现在的工资待遇怎么样要求有哪些    三星电子厂工资待遇怎么样啊
```

执行如下命令开始预测：
```shell
export CUDA_VISIBLE_DEVICES=0

python run_diffcse.py \
	--mode "eval" \
	--extractor_name "rocketqa-zh-dureader-query-encoder" \
	--generator_name "ernie-1.0" \
	--discriminator_name "ernie-3.0-base-zh" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--infer_set_file "./data/test.txt" \
	--ckpt_dir "./checkpoints/best_spearman" \
    --save_infer_path "./infer_result.txt" \
	--batch_size "8" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--dup_rate "0.0" \
	--seed "0" \
	--device "gpu"

```


## Reference
[1] [1] Chuang Y S ,  Dangovski R ,  Luo H , et al. DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings[J]. arXiv e-prints, 2022. https://arxiv.org/pdf/2204.10298.pdf.