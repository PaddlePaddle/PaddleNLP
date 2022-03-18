# 阅读理解 DuReader-robust

# 简介

## 任务说明
阅读理解模型的鲁棒性是衡量该技术能否在实际应用中大规模落地的重要指标之一。随着当前技术的进步，模型虽然能够在一些阅读理解测试集上取得较好的性能，但在实际应用中，这些模型所表现出的鲁棒性仍然难以令人满意。DuReader-robust数据集作为首个关注阅读理解模型鲁棒性的中文数据集，旨在考察模型在真实应用场景中的过敏感性、过稳定性以及泛化能力等问题。

## 数据集

DuReader-robust数据集是单篇章、抽取式阅读理解数据集，具体的任务定义为：
对于一个给定的问题q和一个篇章p，参赛系统需要根据篇章内容，给出该问题的答案a。数据集中的每个样本，是一个三元组<q, p, a>，例如：

**问题 q**: 乔丹打了多少个赛季

**篇章 p**: 迈克尔.乔丹在NBA打了15个赛季。他在84年进入nba，期间在1993年10月6日第一次退役改打棒球，95年3月18日重新回归，在99年1月13日第二次退役，后于2001年10月31日复出，在03年最终退役…

**参考答案 a**: [‘15个’,‘15个赛季’]

关于该数据集的详细内容，可参考数据集[论文](https://arxiv.org/abs/2004.11142)。

## 快速开始

### 数据准备

为了方便开发者进行测试，我们已将数据集上传至HuggingFace。


### Fine-tune

按如下方式启动 Fine-tuning:

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_du.py \
    --model_type ernie_gram \
    --model_name_or_path ernie-gram-zh \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/dureader-robust/ \
    --do_train \
    --do_predict \
    --device gpu \
 ```

* `model_type`: 预训练模型的种类。如bert，ernie，roberta等。
* `model_name_or_path`: 预训练模型的具体名称。如bert-base-chinese，roberta-wwm-ext等。或者是模型文件的本地路径。
* `output_dir`: 保存模型checkpoint的路径。
* `do_train`: 是否进行训练。
* `do_predict`: 是否进行预测。

训练结束后模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "exact": 72.90049400141143,
  "f1": 86.95957173352133,
  "total": 1417,
  "HasAns_exact": 72.90049400141143,
  "HasAns_f1": 86.95957173352133,
  "HasAns_total": 1417
}
```

评估结束后模型会自动对测试集进行预测，并将可提交的结果生成在`prediction.json`中。


**NOTE:** 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如`--model_name_or_path=./tmp/dureader-robust/model_19000/`，程序会自动加载模型参数`/model_state.pdparams`，也会自动加载词表，模型config和tokenizer的config。
