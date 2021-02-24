# XLNet

## 模型简介

[XLNet](https://arxiv.org/abs/1906.08237) （XLNet: Generalized Autoregressive Pretraining for Language Understanding）
XLNet模型是一种自回归的预训练语言模型。有别于传统的单向自回归模型，XLNet通过最大化序列所有可能排列的期望，可以同时关注到上下文的信息。另外，XLNet在预训练阶段集成了 [Transformer-XL](https://arxiv.org/abs/1901.02860) 中的片段循环机制(Segment Recurrent Mechanism)和相对位置编码(Relative Positional Encoding)以支持更长的序列，在涉及到超长文本序列的任务上有更好的表现。

本项目是XLNet在 Paddle 2.0上的开源实现，包含了在 [GLUE评测任务](https://gluebenchmark.com/tasks) 上的微调代码。

## 快速开始

### 安装说明

* PaddlePaddle 安装

   本项目依赖于 PaddlePaddle 2.0.0 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

* PaddleNLP 安装

   ```shell
   pip install paddlenlp>=2.0.0b
   ```

### 数据准备

##### GLUE评测任务数据

GLUE评测任务所含数据集已在paddlenlp中以API形式提供，无需预先准备，使用`run_glue.py`执行微调时将会自动下载。

### 执行Fine-tuning

以GLUE中的SST-2任务为例，启动Fine-tuning的方式如下：

```shell
python -u ./run_glue.py \
    --model_type xlnet \
    --model_name_or_path xlnet-base-cased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --save_steps 1000 \
    --output_dir ./tmp/ \
    --n_gpu 1
```

其中参数释义如下：
- `model_type` 指示了模型类型，使用XLNet模型时设置为xlnet即可。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `task_name` 表示Fine-tuning的任务。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `n_gpu` 表示使用的 GPU 卡数。若希望使用多卡训练，将其设置为指定数目即可。

基于`xlnet-base-cased`在GLUE各评测任务上Fine-tuning后，在验证集上有如下结果：

| Task  | Metric                       | Result            |
|:-----:|:----------------------------:|:-----------------:|
| SST-2 | Accuracy                     |      0.93922      |
| QNLI  | Accuracy                     |      0.91708      |
| CoLA  | Mattehew's corr              |            |
| MRPC  | F1/Accuracy                  |  0.91459/0.88235  |
| STS-B | Person/Spearman corr         |  0.88847/0.88350  |
| QQP   | Accuracy/F1                  |  0.90581/0.87347  |
| MNLI  | Matched acc/MisMatched acc   |  0.84422/0.84825  |
| RTE   | Accuracy                     |      0.711191     |
