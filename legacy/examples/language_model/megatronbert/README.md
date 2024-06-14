# MegatronBert with PaddleNLP

[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)

**模型简介：**
近期在语言建模方面的工作表明，训练大型transformers模型提高了自然语言处理应用的技术水平。然而，由于内存限制，非常大的模型可能难以训练。在这项工作中，
作者提出了训练大型transformers模型的技术，并实现了一种简单、高效的模型运算并行方法，该方法能够训练具有数十亿个参数的transformers模型。

本项目是 MegatronBert 在 Paddle 2.x上的开源实现。

## 快速开始

### 下游任务微调

#### 1、SQuAD1.1 & SQuAD2.0
SQuAD1.1数据集

```shell
python -m paddle.distributed.launch run_squad.py \
    --do_train \
    --do_predict \
    --batch_size=8 \
    --model_name_or_path=megatronbert-cased
    --learning_rate=1e-5 \
    --output_dir=output/ \
    --device=gpu \
    --num_train_epochs=2
```
其中参数释义如下：
- `model_name_or_path` 指示了模型类型，当前支持`megatronbert-cased`和`megatronbert-uncased`模型。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `output_dir` 表示模型保存路径。
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。
- `num_train_epochs` 表示需要训练的epoch数量

训练结束后模型会对模型进行评估，其评估在验证集上完成, 训练完成后你将看到如下结果:
```text
{
  "exact": 88.78902554399243,
  "f1": 94.4082803514958,
  "total": 10570,
  "HasAns_exact": 88.78902554399244,
  "HasAns_f1": 94.4082803514958,
  "HasAns_total": 10570
}
```

SQuAD2.0数据集
```shell
python -m paddle.distributed.launch run_squad.py \
    --do_train \
    --version_2_with_negative \
    --do_predict \
    --batch_size=8 \
    --model_name_or_path=megatronbert-cased
    --learning_rate=1e-5 \
    --output_dir=output/ \
    --device=gpu \
    --num_train_epochs=2
```

其中参数释义如下：
- `version_2_with_negative`  是否使用SQuAD2.0数据集

训练结束后模型会对模型进行评估，其评估在验证集上完成, 训练完成后你将看到如下结果:
```text
{
  "exact": 85.85867093405206,
  "f1": 88.70579950475263,
  "total": 11873,
  "HasAns_exact": 82.47300944669365,
  "HasAns_f1": 88.17543143048748,
  "HasAns_total": 5928,
  "NoAns_exact": 89.23465096719933,
  "NoAns_f1": 89.23465096719933,
  "NoAns_total": 5945,
  "best_exact": 85.99343047250063,
  "best_exact_thresh": -1.6154582500457764,
  "best_f1": 88.75296534320918,
  "best_f1_thresh": -0.20494508743286133
}
```

#### 2、mnli数据集

```shell
python -m paddle.distributed.launch run_glue.py \
    --task_name=mnli \
    --output_dir=output/ \
    --model_name_or_path=megatronbert-cased \
    --learning_rate=1e-5 \
    --device=gpu \
    --num_train_epochs=2
```
训练结束后模型会对模型进行评估，其评估在测试集上完成, 训练完成后你将看到如下结果:
```text
eval loss: 0.186327, acc: 0.8992358634742741, eval loss: 0.332409, acc: 0.8968673718470301, eval done total : 118.65499472618103 s
```

# Reference

* [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)
