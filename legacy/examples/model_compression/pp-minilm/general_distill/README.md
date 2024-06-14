# PP-MiniLM 任务无关蒸馏

## 环境要求

本实验基于 NVIDIA Tesla V100 32G 8 卡进行，训练周期约为 2-3 天。

## 原理介绍

任务无关知识蒸馏是用较大（层数更多、宽度更宽的）的基于 Transformer Layer 的预训练模型对较小（层数更少、宽度更窄的）的基于 Transformer Layer 的预训练模型进行蒸馏，从而得到更小、效果与较大模型更接近的预训练模型。

PP-MiniLM 参考了 MiniLMv2 提出的 Multi-Head Self-Attention Relation Distillation 蒸馏策略。MiniLMv2 算法是用 24 层 large-size 的教师模型倒数几层的 Q-Q、K-K、V-V 之间的relation对6层学生模型最后一层 Q-Q、K-K、V-V 之间的 relation 进行蒸馏。具体的做法是，首先将学生、教师用于蒸馏的层上的 Q、K、V 的 Head 数进行统一，然后计算各自 Q-Q、K-K、V-V 的点积，最后对教师和学生的点积计算KL散度损失。由于 relation 的 shape 是 `[batch_size, head_num, seq_len, seq_len]`，因此可以认为这里的relation是一种Token与Token之间的关系。

本方案在 MiniLMv2 策略的基础上，做了进一步优化: 通过引入多视角的注意力关系知识来进一步提升模型效果。MiniLMv2 的自注意力关系知识仅建模了 Token 与 Token 之间的关系，PP-MiniLM 在此基础上额外引入了样本与样本间的自注意力关系知识，也就是挖掘出更多教师模型所蕴含的知识，从而进一步优化模型效果。

具体来说，PP-MiniLM 利用了 `roberta-wwm-ext-large` 第 20 层的 Q-Q、K-K、V-V 之间的 Sample 与 Sampl 之间关系对 6 层学生模型 PP-MiniLM 第 6 层的 Q-Q、K-K、V-V 之间的 Sample 与 Sample 之间的关系进行蒸馏。与MiniLMv2不同的是，PP-MiniLM的策略需要在统一Q、K、V的Head数之后，对Q、K、V转置为 `[seq_len, head_num, batch_size, head_dim]`，这样Q-Q、K-K、V-V 的点积则可以表达样本间的关系。经过我们的实验，这种方法比使用原始 MiniLMv2 算法在 CLUE 上平均准确率高 0.36。


### 数据介绍

任务无关知识蒸馏的训练数据一般是预训练语料，可以使用公开的预训练语料 [CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020/)。需要将数据处理成一行一个句子的格式，再将数据文件分割成多个子文件（例如 64 个），放在同一个目录下。


### 运行方式

```shell
sh run.sh # 包含general_distill.py的运行配置
cd ..
```

其中 `general_distill.py` 参数释义如下：

- `model_type` 指示了学生模型类型，当前仅支持 'ppminilm'、'roberta'。
- `num_relation_heads` relation head 的个数，一般对于 large-size 的教师模型是64，对于 base-size 的教师模型是 48。
- `teacher_model_type`指示了教师模型类型，当前仅支持 'roberta'。
- `teacher_layer_index`蒸馏时使用的教师模型的层
- `student_layer_index` 蒸馏时使用的学生模型的层
- `teacher_model_name_or_path`教师模型的名称，例如`'roberta-wwm-ext-large'`
- `max_seq_length` 最大的样本长度
- `num_layers` 学生模型的层数，目前仅支持 2，4，6
- `logging_steps` 日志间隔
- `max_steps` 最大迭代次数
- `warmup_steps` 学习率增长得到`learning_rate`所需要的步数
- `save_steps`保存模型的间隔步数
- `weight_decay` 表示AdamW优化器中使用的 weight_decay 的系数
- `output_dir`训练相关文件以及模型保存的输出路径
- `device`设备选择，推荐使用 gpu
- `input_dir` 训练数据目录
- `use_amp` 是否使用混合精度训练，默认 False
- `alpha`head间关系的权重，默认 0.0
- `beta`样本间关系的权重，默认 0.0

将最终得到的模型绝对路径保存至 `$GENERAL_MODEL_DIR`，例如：

```shell
GENERAL_MODEL_DIR=PaddleNLP/examples/model_compression/PP-MiniLM/general_distill/pretrain/model_400000
```

## 模型精度

在 CLUE 数据集上经过超参寻优后，得到 CLUE 上各个任务上的最高准确率如下表：

| AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL   | Avg   |
| ----- | ----- | ------- | ----- | ----- | ----------- | ----- | ----- |
| 74.14 | 57.43 | 61.75   | 81.01 | 76.17 | 86.18       | 79.17 | 73.69 |
