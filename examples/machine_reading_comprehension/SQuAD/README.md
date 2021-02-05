# 阅读理解 SQuAD

# 简介

## 1. 任务说明
本文主要介绍基于Bert预训练模型的SQuAD（Stanford Question Answering Dataset）数据集的阅读理解任务，给定一篇文章和一个问题，计算答案在文章中的起始位置和结束位置。对于SQuAD2.0数据集，还可以返回答案在文章中不存在的概率。

## 2. 数据集

此任务的数据集包括以下数据集：

SQuAD v1.1
- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

SQuAD v2.0
- [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)


# 快速开始


## 1. 开始第一次模型调用

### 安装说明

* PaddlePaddle 安装

   本项目依赖于 PaddlePaddle 2.0-rc1 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

* PaddleNLP 安装

   ```shell
   pip install paddlenlp>=2.0.0b
   ```

* 环境依赖

    Python的版本要求 3.6+


### 数据准备
为了方便开发者进行测试，我们内置了数据下载脚本，用户可以通过命令行传入`--version_2_with_negative`控制所需要的SQuAD数据集版本，也可以通过`--data_path`传入本地数据集的位置，数据集需保证与SQuAD数据集格式一致。


### Fine-tune

对于 SQuAD v1.1,按如下方式启动 Fine-tuning:

```shell
python -u ./run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 100 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --n_gpu 1
 ```

* `model_type`: 预训练模型的种类。如bert，ernie，roberta等。
* `model_name_or_path`: 预训练模型的具体名称。如bert-base-uncased，bert-large-cased等。或者是模型文件的本地路径。
* `output_dir`: 保存模型checkpoint的路径。

训练结束后模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "exact": 81.18259224219489,
  "f1": 88.68817481234801,
  "total": 10570,
  "HasAns_exact": 81.18259224219489,
  "HasAns_f1": 88.68817481234801,
  "HasAns_total": 10570
}
```

对于 SQuAD v2.0,按如下方式启动 Fine-tuning:

```shell
python -u ./run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --n_gpu 1 \
    --version_2_with_negative
 ```

* `version_2_with_negative`: 使用squad2.0数据集和评价指标的标志。

训练结束后会在模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "exact": 73.25865408910974,
  "f1": 76.63096554166046,
  "total": 11873,
  "HasAns_exact": 73.22874493927125,
  "HasAns_f1": 79.98303877802545,
  "HasAns_total": 5928,
  "NoAns_exact": 73.28847771236333,
  "NoAns_f1": 73.28847771236333,
  "NoAns_total": 5945,
  "best_exact": 74.31988545439232,
  "best_exact_thresh": -2.5820093154907227,
  "best_f1": 77.20521797731851,
  "best_f1_thresh": -1.559523582458496
}
```

其中会输出 `best_f1_thresh` 是最佳阈值，可以使用这个阈值重新训练，或者从 `all_nbest_json`变量中获取最终 `prediction`。
训练方法与前面大体相同，只需要设定 `--null_score_diff_threshold` 参数的值为测评时输出的 `best_f1_thresh` ，通常这个值在 -1.0 到 -5.0 之间。

**NOTE:** 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如`--model_name_or_path=./tmp/squad/model_19000/`，程序会自动加载模型参数`/model_state.pdparams`，也会自动加载词表，模型config和tokenizer的config。

## 2. 目录结构

```text
.
├── README.md           # 文档
├── run_squad.py        # 训练代码  
├── args.py             # 参数读取
```

# 其他

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
