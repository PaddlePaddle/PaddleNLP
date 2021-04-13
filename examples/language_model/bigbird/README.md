# Big Bird

## 模型介绍
[Big Bird](https://arxiv.org/abs/2007.14062)(Transformers for Longer Sequences) 是Google的研究人员提出的针对长序列预训练模型，使用了稀疏注意力机制，将计算复杂度、空间复杂度降到线性复杂度，大大提升了长序列任务的预测能力。

本项目是 Big Bird 的 PaddlePaddle 实现， 包含模型训练，模型验证等内容。以下是本例的简要目录结构及说明：

```text
.
├── args.py                 # 预训练任务的配置
├── run_classifier.py       # IMDB数据集的分类任务
├── run_pretrain.py         # 预训练任务脚本
├── README.md               # 文档
└── data/                    # 示例数据
```
## 快速开始

### 环境依赖

- sentencepiece

安装命令：`pip install sentencepiece`

### 数据准备
根据论文中的信息，目前 Big Bird 的预训练数据是主要是由 Books，CC-News，Stories, Wikipedia 4种预训练数据来构造，用户可以根据自己的需要来下载和清洗相应的数据。目前已提供一份示例数据在 data 目录。


### 预训练任务

下面是预训练任务的具体的执行方式

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" --log_dir log  run_pretrain.py --model_name_or_path bigbird-base-uncased \
    --input_dir "./data" \
    --output_dir "output" \
    --batch_size 4 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --max_steps 100000 \
    --save_steps 10000 \
    --logging_steps 1 \
    --max_encoder_length 512 \
    --max_pred_length 75
```

其中参数释义如下：

- `gpus` paddle.distributed.launch参数，用于指定使用哪张显卡。单卡格式："0"；多卡格式："0,1,2"。
- `log_dir` paddle.distributed.launch参数，用于指定训练日志输出的目录，默认值为`log`。（注意，如果需要在同一目录多次启动run_pretrain.py，需要设置不同的log_dir，否则日志会重定向到相同的文件中）。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。目前支持的训练模型配置有："bigbird-base-uncased"。若模型相关内容保存在本地，这里也可以提供相应目录地址，例如："./checkpoint/model_xx/"
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `batch_size` 训练的batch大小
- `weight_decay` AdamW权重衰减参数
- `learning_rate` 训练的学习率
- `max_steps` 最大训练步数
- `save_steps` 保存模型间隔
- `logging_steps` 打印日志的步数
- `max_encoder_length` MLM任务的最大的token数目
- `max_pred_length` MLM任务最大的需要预测token的数目


### 验证任务

#### Imdb分类任务
通过预训练任务训练完成之后，可以预训练的模型参数，在 Big Bird 的验证任务中通过IMDB数据集来进行最终模型效果的验证，[IMDB数据集](http://ai.stanford.edu/~amaas/data/sentiment/) ，IMDB数据集是关于电影用户评论情感分析的数据集，主要是包含了50000条偏向明显的评论，其中25000条作为训练集，25000作为测试集。label为pos(positive)和neg(negative)，是一个序列文本分类任务，具体的执行脚本如下。


```shell
export CUDA_VISIBLE_DEVICES=0
python run_classifier.py --model_name_or_path bigbird-base-uncased \
    --output_dir "output" \
    --batch_size 2 \
    --learning_rate 5e-6 \
    --max_steps 16000 \
    --save_steps 1000 \
    --max_encoder_length 3072
```

其中参数释义如下：

- `model_name_or_path` 指示了finetune使用的具体预训练模型以及预训练时使用的tokenizer，目前支持的预训练模型有："bigbird-base-uncased"。若模型相关内容保存在本地，这里也可以提供相应目录地址，例如："./checkpoint/model_xx/"。
- `output_dir` 指定输出文件。
- `batch_size` 训练的batch大小。
- `learning_rate` 训练的学习率。
- `max_steps` 最大训练步数。
- `save_steps` 保存模型间隔。
- `logging_steps` 打印日志的步数。
- `max_encoder_length` MLM任务的最大的token数目。


基于`bigbird-base-uncased`在IMDB评测任务上Fine-tuning后，在验证集上有如下结果：

| Task  | Metric                       | Result            |
|:-----:|:----------------------------:|:-----------------:|
| IMDB  | Accuracy                     |      0.9449       |

#### Glue任务

以GLUE中的SST-2任务为例，启动Fine-tuning的方式如下：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type bigbird \
    --model_name_or_path bigbird-base-uncased \
    --task_name SST-2 \
    --max_encoder_length 128 \
    --batch_size 32   \
    --learning_rate 1e-5 \
    --epochs 5 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu
```

其中参数释义如下：
- `model_type` 指示了模型类型，使用bigbird模型时设置为bigbird即可。
- `model_name_or_path` 指示了finetune使用的具体预训练模型以及预训练时使用的tokenizer，目前支持的预训练模型有："bigbird-base-uncased"。若模型相关内容保存在本地，这里也可以提供相应目录地址，例如："./checkpoint/model_xx/"。
- `task_name` 表示Fine-tuning的任务。
- `max_encoder_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。

基于`bigbird-base-uncased`在GLUE各评测任务上Fine-tuning后，在验证集上有如下结果：

| Task  | Metric                       | Result            |
|:-----:|:----------------------------:|:-----------------:|
| SST-2 | Accuracy                     |      0.9365       |
| QNLI  | Accuracy                     |      0.9017       |
| CoLA  | Mattehew's corr              |      0.5708       |
| MRPC  | F1/Accuracy                  |  0.9019 / 0.8603  |
| STS-B | Person/Spearman corr         |  0.8591 / 0.8607  |
| QQP   | Accuracy/F1                  |  0.9132 / 0.8828  |
| MNLI  | Matched acc/MisMatched acc   |  0.8615 / 0.8606  |
| RTE   | Accuracy                     |      0.7004       |

### 致谢

* 感谢[Google 研究团队](https://github.com/google-research/bigbird)提供BigBird开源代码的实现以及预训练模型。

### 参考论文

* Zaheer, et al. "Big bird: Transformers for longer sequences" Advances in Neural Information Processing Systems, 2020
