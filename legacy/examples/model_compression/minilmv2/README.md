# MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers

以下是本例的简要目录结构及说明：
```
.
├── general_distill.py       # 通用蒸馏脚本
├── run_clue.py              # 在下游任务上的微调脚本
└── README.md                # 文档，本文件
```
## 简介
本目录下的实验主要参考论文[《MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers》](https://arxiv.org/abs/2012.15828)实现。

MiniLMv2也是从层数深的Transformer类模型到层数较浅的Transformer类模型的蒸馏策略。它的优势是只需要取教师模型和学生模型中的各一层进行蒸馏训练，而不像其他方法需要蒸馏更多的层，避免面对更加复杂的layer mapping问题，并且效果优于TinyBert的蒸馏策略。

MiniLMv2蒸馏的目标是教师模型某层的q与q, k与k, v与v的矩阵乘结果和学生模型最后一层的q与q, k与k, v与v的矩阵乘之间的kl散度loss。其中教师模型是large size时，选择实验并选取倒数某一层，当教师模型是base size时，选择最后一层进行蒸馏即可。

为了防止教师模型是large size时，head size与学生模型不同，蒸馏目标的shape无法匹配，MiniLMv2还需要对head进行重组，先合并再按relation_head_num重新分割head_num和head_size。

## 数据、预训练模型介绍及获取

### 数据获取
由于本实验是通用场景下的蒸馏，因此数据和预训练类似。可以参考[NLP Chinese Corpus](https://github.com/brightmart/nlp_chinese_corpus)中提供的数据。
数据下载完成后，需要将所有数据集整理成每行一条文本数据，再将数据切分成多个小文件，并放在一个目录下，以便使用多卡并行训练。

### 训练启动方式

假设我们把切分好的预训练数据文件都放在`${dataset}`下，那么我们可以运行如下命令用单机八卡进行预训练蒸馏：
```shell

dataset=/PaddleNLP/dataset
output_dir=./pretrain

python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" general_distill.py \
    --student_model_type tinybert \
    --num_relation_heads 48 \
    --student_model_name_or_path tinybert-6l-768d-zh \
    --init_from_student False \
    --teacher_model_type bert \
    --teacher_model_name_or_path bert-base-chinese \
    --max_seq_length 128 \
    --batch_size 256 \
    --learning_rate 6e-4 \
    --logging_steps 20 \
    --max_steps 100000 \
    --warmup_steps 4000 \
    --save_steps 5000 \
    --teacher_layer_index 11 \
    --student_layer_index 5 \
    --weight_decay 1e-2 \
    --output_dir ${output_dir} \
    --device gpu \
    --input_dir ${dataset} \

```

其中参数释义如下：

- `student_model_type` 学生模型的类型
- `num_relation_heads` head重新组合之后的head数
- `student_model_name_or_path` 学生模型的名字（需要与学生模型类型对应），或者是学生模型的路径
- `init_from_student` 本次蒸馏的学生模型是否用`student_model_name_or_path`中的参数进行初始化，是个bool类型的参数。默认是False
- `teacher_model_type bert` 教师模型的类型
- `teacher_model_name_or_path`  教师模型的名字
- `max_seq_length 128` 表示最大句子长度，超过该长度将被截断。
- `warmup_steps` 学习率warmup up的步数
- `save_steps` 保存模型的频率
- `teacher_layer_index`表示学生模型从教师模型学习的教师层
- `student_layer_index` 表示学生模型从教师模型学习的学生层
- `output_dir` 模型输出的目录
- `device gpu` 表示运行该程序的设备，默认是gpu
- `input_dir` 预训练数据的存放地址



### 评价方法

假设预训练完成后的模型存储在`${pretrained_models}`下，这里也提供了我们已经预训练完成的一版[模型](https://bj.bcebos.com/paddlenlp/models/general_distill/minilmv2_6l_768d_ch.tar.gz)可供参考，模型与`tinybert-6l-768d-zh`结构相同，因此可以使用`TinyBertForSequenceClassification.from_pretrained()`对模型直接进行加载。
本示例训练出的通用模型需要在下游任务上Fine-tuning，利用下游任务上的指标进行评价。
我们可以运行如下脚本在单卡上进行Fine-tuning：

```shell

export CUDA_VISIBLE_DEVICES="0"

python -u ./run_clue.py \
    --model_type tinybert  \
    --model_name_or_path ${pretrained_models} \
    --task_name ${TASK_NAME} \
    --max_seq_length ${max_seq_len} \
    --batch_size 16   \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --logging_steps 100 \
    --seed 42  \
    --save_steps  100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --device gpu  \

```


其中不同的任务下，`${learning_rate}`、`${num_train_epochs}`、`${max_seq_len}`，我们推荐不同的Fine-tuning的超参数，可以参考以下配置：

| TASK_NAME        | AFQMC | TNEWS | IFLYTEK | OCNLI | CMNLI | CLUEWSC2020 | CSL  |
| ---------------- | ----- | ----- | ------- | ----- | ----- | ----------- | ---- |
| learning_rate    | 2e-5  | 2e-5  | 2e-5    | 3e-5  | 3e-5  | 1e-5        | 1e-5 |
| num_train_epochs | 3     | 3     | 6       | 6     | 3     | 50          | 8    |
| max_seq_len      | 128   | 128   | 128     | 128   | 128   | 128         | 256  |


### 蒸馏实验结果

本示例选择的是CLUE中的分类任务，以`bert-base-chinese`作教师模型，利用MiniLMv2策略对6层模型进行蒸馏，可以得到的通用模型在CLUE上的指标为：

| CLUE    | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL   |
| ------- | ----- | ----- | ------- | ----- | ----- | ----------- | ----- |
| Acc (%) | 71.38 | 56.46 | 58.87   | 79.01 | 73.02 | 68.42       | 77.73 |


## 参考文献

Wang W, Bao H, Huang S, Dong L, Wei F. [MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers](https://arxiv.org/abs/2012.15828)[J]. arXiv preprint arXiv:2012.15828v2, 2021.
