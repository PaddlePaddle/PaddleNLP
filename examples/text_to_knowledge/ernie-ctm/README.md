# 文本知识标注工具集 - ERNIE-CTM

## 简介

ERNIE for **C**hinese **T**ext **M**ining，适用于中文挖掘任务的预训练语言模型，包含全中文词表、中英文标点、单位符号、汉语拼音等token，大幅减少tokenize中UNK的情况，对中文标注、挖掘类任务有较大增益。

## 快速开始

### 数据准备

我们提供了少数样本用以示例输入数据格式。执行以下命令，下载并解压示例数据集：

```bash
python download.py --data_dir ./  
```

训练使用的数据可以由用户根据实际的应用场景，自己组织数据。每行数据都由tokens、tags、cls_label组成，tags采用 BIOES 标注体系，cls_label是整个句子的分类，包含"编码/引用/列表","外语句子","古文/古诗句","其他文本"四种，由于目前发布的预训练模型针对的是现代文，因此前三种文本只用于训练文本分类，不用于训练序列标注。

示例如下：

```text
{"tokens": ["1", ".", "1", ".", "8", "车", "辆", "自", "动", "驾", "驶", "及", "控", "制", " ", "8"], "tags": ["B-数量词", "I-数量词", "I-数量词", "I-数量词", "E-数量词", "B-物体类", "E-物体类", "B-场景事件", "I-场景事件", "I-场景事件", "E-场景事件", "S-连词", "B-场景事件", "E-场景事件", "S-w", "S-数量词"], "cls_label": "编码/引用/列表"}
{"tokens": ["亦", "在", "空", "中", "捕", "食", "，", "边", "飞", "翔", "边", "捕", "食", "。"], "tags": ["S-词汇用语", "S-介词", "B-位置方位", "E-位置方位", "B-场景事件", "E-场景事件", "S-w", "S-词汇用语", "B-场景事件", "E-场景事件", "S-词汇用语", "B-场景事件", "E-场景事件", "S-w"], "cls_label": "其他文本"}
```


### 模型训练

#### 单卡训练

```bash
python -u train.py \
    --max_seq_len 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir ./tmp/ \
    --device "gpu"
```

#### 多卡训练
```bash
python -m paddle.distributed.launch --gpus "0,1"  train.py \
    --max_seq_len 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir ./tmp/ \
    --device "gpu"
```

其中参数释义如下：
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。


### 模型评估

通过加载训练过程中保存的模型，可以对验证集数据进行验证，启动方式如下：

```bash
python -u eval.py \
    --max_seq_len 128 \
    --batch_size 32   \
    --init_ckpt_dir ./tmp/ernie_ctm_ft_model_1.pdparams \
    --device "gpu"
```

其中 init_ckpt_dir 是模型加载路径，请根据具体的模型路径填写该项。

### 模型预测

对无标签数据可以启动模型预测：

```bash
python -u predict.py \
    --max_seq_len 128 \
    --batch_size 32   \
    --init_ckpt_dir ./tmp/ernie_ctm_ft_model_1.pdparams \
    --device "gpu"
```
