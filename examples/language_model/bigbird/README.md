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

### 安装说明

* PaddlePaddle安装

    本项目依赖于 PaddlePaddle 2.0.1 及以上版本或适当的develop版本，请参考 [安装指南](https://www.paddlepaddle.org.cn/install/quick) 进行安装。

* Sentencepiece 安装

   ```shell
   pip install sentencepiece
   ```

* PaddleNLP 安装

   ```shell
   pip install paddlenlp\>=2.0.0rc5
   ```

* 下载代码

    克隆代码库到本地


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
- `model_name_or_path` 要训练的模型或者之前训练的checkpoint。
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
通过预训练任务训练完成之后，可以预训练的模型参数，在 Big Bird 的验证任务中通过IMDB数据集来进行最终模型效果的验证，[IMDB数据集](http://ai.stanford.edu/~amaas/data/sentiment/) ，IMDB数据集是关于电影用户评论情感分析的数据集，主要是包含了50000条偏向明显的评论，其中25000条作为训练集，25000作为测试集。label为pos(positive)和neg(negative)，是一个序列文本分类任务，具体的执行脚本如下。


```shell
export CUDA_VISIBLE_DEVICES=0
python run_classifier.py --model_name_or_path bigbird-base-uncased-finetune \
    --output_dir "output" \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_steps 10000 \
    --save_steps 1000 \
    --max_encoder_length 3072
```

其中参数释义如下：

- `model_name_or_path` 要训练的模型或者之前训练的checkpoint。
- `output_dir` 指定输出文件。
- `batch_size` 训练的batch大小。
- `learning_rate` 训练的学习率。
- `max_steps` 最大训练步数。
- `save_steps` 保存模型间隔。
- `logging_steps` 打印日志的步数。
- `max_encoder_length` MLM任务的最大的token数目。


基于`bigbird-base-uncased-finetune`在IMDB评测任务上Fine-tuning后，在验证集上有如下结果：

| Task  | Metric                       | Result            |
|:-----:|:----------------------------:|:-----------------:|
| IMDB  | Accuracy                     |      0.9449       |

### 致谢

* 感谢[Google 研究团队](https://github.com/google-research/bigbird)提供BigBird开源代码的实现以及预训练模型。

### 参考论文

* Zaheer, et al. "Big bird: Transformers for longer sequences" Advances in Neural Information Processing Systems, 2020
