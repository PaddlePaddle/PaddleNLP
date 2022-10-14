# 基于预训练模型 ERNIE-Gram 的单塔文本匹配

我们基于预训练模型 ERNIE-Gram 给出了单塔文本匹配的 2 种训练范式: Point-wise 和 Pair-wise。其中单塔 Point-wise 匹配模型适合直接对文本对进行 2 分类的应用场景: 例如判断 2 个文本是否为语义相似；Pair-wise 匹配模型适合将文本对相似度作为特征之一输入到上层排序模块进行排序的应用场景。

## 模型下载
本项目使用语义匹配数据集 LCQMC 作为训练集 , 基于 ERNIE-Gram 预训练模型热启训练并开源了单塔 Point-wise 语义匹配模型， 用户可以直接基于这个模型对文本对进行语义匹配的 2 分类任务。

| 模型  | dev acc |
| ---- | ------- |
| [ERNIE-1.0-Base](https://bj.bcebos.com/paddlenlp/models/text_matching/ernie1.0_zh_pointwise_matching_model.tar)  | 89.43 |
| [ERNIE-Gram-Base](https://bj.bcebos.com/paddlenlp/models/text_matching/ernie_gram_zh_pointwise_matching_model.tar)  | 90.60 |

## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```
ernie_matching/
├── deply # 部署
|   └── python
|       └── predict.py # python 预测部署示例
├── export_model.py # 动态图参数导出静态图参数脚本
├── model.py # Point-wise & Pair-wise 匹配模型组网
├── data.py # Point-wise & Pair-wise 训练样本的转换逻辑 、Pair-wise 生成随机负例的逻辑
├── train_pointwise.py # Point-wise 单塔匹配模型训练脚本
├── train_pairwise.py # Pair-wise 单塔匹配模型训练脚本
├── predict_pointwise.py # Point-wise 单塔匹配模型预测脚本，输出文本对是否相似: 0、1 分类
├── predict_pairwise.py # Pair-wise 单塔匹配模型预测脚本，输出文本对的相似度打分
└── train.py # 模型训练评估
```

### 模型训练

我们以中文文本匹配公开数据集 LCQMC 为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行单塔 Point-wise 模型训练，并在开发集（dev.tsv）验证。Pair-wise 匹配模型只需要采用 `train_pairwise.py` 脚本即可。

```shell
$ unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0" train_pointwise.py \
        --device gpu \
        --save_dir ./checkpoints \
        --batch_size 32 \
        --learning_rate 2E-5
```

可支持配置的参数：

* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `max_seq_length`：可选，ERNIE-Gram 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `epochs`: 训练轮次，默认为3。
* `warmup_proption`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。

代码示例中使用的预训练模型是 ERNIE-Gram，如果想要使用其他预训练模型如 ERNIE, BERT，RoBERTa，Electra等，只需更换`model` 和 `tokenizer`即可。

```python

# 使用 ERNIE-3.0-medium-zh 预训练模型
model = AutoModel.from_pretrained('ernie-3.0-medium-zh')
tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

# 使用 ERNIE-Gram 预训练模型
model = AutoModel.from_pretrained('ernie-gram-zh')
tokenizer = AutoTokenizer.from_pretrained('ernie-gram-zh')

# 使用 ERNIE 预训练模型
# ernie-1.0
#model = AutoModel.from_pretrained('ernie-1.0-base-zh'))
#tokenizer = AutoTokenizer.from_pretrained('ernie-1.0-base-zh')

# ernie-tiny
# model = AutoModel.from_pretrained('ernie-tiny'))
# tokenizer = AutoTokenizer.from_pretrained('ernie-tiny')


# 使用 BERT 预训练模型
# bert-base-chinese
# model = AutoModel.from_pretrained('bert-base-chinese')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

# bert-wwm-chinese
# model = AutoModel.from_pretrained('bert-wwm-chinese')
# tokenizer = AutoTokenizer.from_pretrained('bert-wwm-chinese')

# bert-wwm-ext-chinese
# model = AutoModel.from_pretrained('bert-wwm-ext-chinese')
# tokenizer = AutoTokenizer.from_pretrained('bert-wwm-ext-chinese')


# 使用 RoBERTa 预训练模型
# roberta-wwm-ext
# model = AutoModel.from_pretrained('roberta-wwm-ext')
# tokenizer = AutoTokenizer.from_pretrained('roberta-wwm-ext')

# roberta-wwm-ext
# model = AutoModel.from_pretrained('roberta-wwm-ext-large')
# tokenizer = AutoTokenizer.from_pretrained('roberta-wwm-ext-large')

```
更多预训练模型，参考[transformers](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer)

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

**NOTE:**
* 如需恢复模型训练，则可以设置`init_from_ckpt`， 如`init_from_ckpt=checkpoints/model_100/model_state.pdparams`。
* 如需使用ernie-tiny模型，则需要提前先安装sentencepiece依赖，如`pip install sentencepiece`

### 基于动态图模型预测

我们用 LCQMC 的测试集作为预测数据,  测试数据示例如下，：
```text
谁有狂三这张高清的  这张高清图，谁有
英雄联盟什么英雄最好    英雄联盟最好英雄是什么
这是什么意思，被蹭网吗  我也是醉了，这是什么意思
现在有什么动画片好看呢？    现在有什么好看的动画片吗？
请问晶达电子厂现在的工资待遇怎么样要求有哪些    三星电子厂工资待遇怎么样啊
```

启动预测：

```shell
$ unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0" \
        predict_pointwise.py \
        --device gpu \
        --params_path "./checkpoints/model_4400/model_state.pdparams"\
        --batch_size 128 \
        --max_seq_length 64 \
        --input_file 'test.tsv'
```

输出预测结果如下:
```text
{'query': '谁有狂三这张高清的', 'title': '这张高清图，谁有', 'pred_label': 1}
{'query': '英雄联盟什么英雄最好', 'title': '英雄联盟最好英雄是什么', 'pred_label': 1}
{'query': '这是什么意思，被蹭网吗', 'title': '我也是醉了，这是什么意思', 'pred_label': 1}
{'query': '现在有什么动画片好看呢？', 'title': '现在有什么好看的动画片吗？', 'pred_label': 1}
{'query': '请问晶达电子厂现在的工资待遇怎么样要求有哪些', 'title': '三星电子厂工资待遇怎么样啊', 'pred_label': 0}
```

### 基于静态图部署预测
#### 模型导出
使用动态图训练结束之后，可以使用静态图导出工具 `export_model.py` 将动态图参数导出成静态图参数。 执行如下命令：

```shell
python export_model.py --params_path checkpoints/model_300/model_state.pdparams --output_path=./output
```

其中`params_path`是指动态图训练保存的参数路径，`output_path`是指静态图参数导出路径。

#### 预测部署
导出静态图模型之后，可以基于静态图模型进行预测，`deploy/python/predict.py` 文件提供了静态图预测示例。执行如下命令：

```shell
python deploy/python/predict.py --model_dir ./output
```

## Reference
[1] Xin Liu, Qingcai Chen, Chong Deng, Huajun Zeng, Jing Chen, Dongfang Li, Buzhou Tang, LCQMC: A Large-scale Chinese Question Matching Corpus,COLING2018.
[2] Xiao, Dongling, Yu-Kun Li, Han Zhang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. “ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding.” ArXiv:2010.12148 [Cs].
