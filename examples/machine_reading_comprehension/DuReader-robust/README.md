# 阅读理解 DuReader-robust

# 简介

## 1. 任务说明
阅读理解模型的鲁棒性是衡量该技术能否在实际应用中大规模落地的重要指标之一。随着当前技术的进步，模型虽然能够在一些阅读理解测试集上取得较好的性能，但在实际应用中，这些模型所表现出的鲁棒性仍然难以令人满意。DuReader-robust数据集作为首个关注阅读理解模型鲁棒性的中文数据集，旨在考察模型在真实应用场景中的过敏感性、过稳定性以及泛化能力等问题。

## 2. 数据集

DuReader-robust数据集是单篇章、抽取式阅读理解数据集，具体的任务定义为：
对于一个给定的问题q和一个篇章p，参赛系统需要根据篇章内容，给出该问题的答案a。数据集中的每个样本，是一个三元组<q, p, a>，例如：

**问题 q**: 乔丹打了多少个赛季

**篇章 p**: 迈克尔.乔丹在NBA打了15个赛季。他在84年进入nba，期间在1993年10月6日第一次退役改打棒球，95年3月18日重新回归，在99年1月13日第二次退役，后于2001年10月31日复出，在03年最终退役…

**参考答案 a**: [‘15个’,‘15个赛季’]

关于该数据集的详细内容，可参考数据集[论文](https://arxiv.org/abs/2004.11142)。

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

为了方便开发者进行测试，我们内置了数据下载脚本，也可以通过`--data_path`传入本地数据集的位置，数据集需保证与DuReader-robust数据集格式一致。


### Fine-tune

按如下方式启动 Fine-tuning:

```shell
python -u ./run_du.py \
    --task_name dureader-robust \
    --model_type bert \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/dureader-robust/ \
    --n_gpu 1 \
 ```

* `task_name`: 数据集的名称，不区分大小写，如dureader-robust，cmrc等。
* `model_type`: 预训练模型的种类。如bert，ernie，roberta等。
* `model_name_or_path`: 预训练模型的具体名称。如bert-base-uncased，bert-large-cased等。或者是模型文件的本地路径。
* `output_dir`: 保存模型checkpoint的路径。

训练结束后模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "exact": 68.59562455892731,
  "f1": 84.23267270105613,
  "total": 1417,
  "HasAns_exact": 68.59562455892731,
  "HasAns_f1": 84.23267270105613,
  "HasAns_total": 1417
}
```

评估结束后模型会自动对测试集进行预测，并将可提交的结果生成在`prediction.json`中。


**NOTE:** 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如`--model_name_or_path=./tmp/dureader-robust/model_19000/`，程序会自动加载模型参数`/model_state.pdparams`，也会自动加载词表，模型config和tokenizer的config。

## 2. 目录结构

```text
.
├── README.md           # 文档
├── run_du.py           # 训练代码  
├── args.py             # 参数读取
```

# 其他

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
