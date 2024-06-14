# XLNet

## 模型简介

[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) 是一款无监督的自回归预训练语言模型。 有别于传统的单向自回归模型，XLNet通过最大化输入序列所有排列的期望来进行语言建模，这使得它可以同时关注到上下文的信息。 另外，XLNet在预训练阶段集成了 [Transformer-XL](https://arxiv.org/abs/1901.02860) 模型，Transformer-XL中的片段循环机制(Segment Recurrent Mechanism)和 相对位置编码(Relative Positional Encoding)机制能够支持XLNet接受更长的输入序列，这使得XLNet在长文本序列的语言任务上有着优秀的表现。

本项目是XLNet在 Paddle 2.0上的开源实现，包含了在 [GLUE评测任务](https://gluebenchmark.com/tasks) 上的微调代码。

## 快速开始

### 环境依赖

- sentencepiece

安装命令：`pip install sentencepiece`

### 数据准备

GLUE评测任务所含数据集已在paddlenlp中以API形式提供，无需预先准备，使用`run_glue.py`执行时将会自动下载。

### 执行Fine-tuning

以GLUE中的SST-2任务为例，启动Fine-tuning的方式如下：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" ./run_glue.py \
    --model_name_or_path xlnet-base-cased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --save_steps 500 \
    --output_dir ./tmp/
```

其中参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `task_name` 表示Fine-tuning的任务。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将与learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。

基于`xlnet-base-cased`在GLUE各评测任务上Fine-tuning后，在验证集上有如下结果：

| Task  | Metric                       | Result             |
|:-----:|:----------------------------:|:------------------:|
| SST-2 | Accuracy                     |      94.266        |
| QNLI  | Accuracy                     |      91.708        |
| CoLA  | Mattehew's corr              |      50.264        |
| MRPC  | F1/Accuracy                  |   91.071/87.745    |
| STS-B | Person/Spearman corr         |   86.243/85.973    |
| QQP   | Accuracy/F1                  |   90.838/87.644    |
| MNLI  | Matched acc/MisMatched acc   |   87.468/86.859    |
| RTE   | Accuracy                     |      70.036        |
| WNLI  | Accuracy                     |      56.338        |

## Reference

- [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
- [zihangdai/xlnet](https://github.com/zihangdai/xlnet)
