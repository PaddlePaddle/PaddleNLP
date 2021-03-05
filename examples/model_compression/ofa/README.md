# BERT Compression Based on PaddleSlim

BERT-base模型是一个迁移能力很强的通用语义表示模型，但是模型中也有一些参数冗余。本教程将介绍如何使用PaddleSlim对BERT-base模型进行压缩。

## 压缩结果

基于`bert-base-uncased` 在GLUE dev数据集上的finetune结果进行压缩。压缩后模型精度和压缩前模型在GLUE dev数据集上的精度对比如下表所示:

| Task  | Metric                       | Result            | Result with PaddleSlim |
|:-----:|:----------------------------:|:-----------------:|:----------------------:|
| SST-2 | Accuracy                     |      0.93005      |       0.931193         |
| QNLI  | Accuracy                     |      0.91781      |       0.920740         |
| CoLA  | Mattehew's corr              |      0.59557      |       0.601244         |
| MRPC  | F1/Accuracy                  |  0.91667/0.88235  |   0.91740/0.88480      |
| STS-B | Person/Spearman corr         |  0.88847/0.88350  |   0.89271/0.88958      |
| QQP   | Accuracy/F1                  |  0.90581/0.87347  |   0.90994/0.87947      |
| MNLI  | Matched acc/MisMatched acc   |  0.84422/0.84825  |   0.84687/0.85242      |
| RTE   | Accuracy                     |      0.711191     |       0.718412         |

压缩后模型相比压缩前加速约59%（测试环境: T4, FP32, batch_size=16），模型参数大小减小26%（从110M减少到81M）。

## 快速开始
本教程示例以GLUE/SST-2 数据集为例。

### Fine-tuing
首先需要对Pretrain-Model在实际的下游任务上进行Finetuning，得到需要压缩的模型。

```shell
cd ../../glue/
```

```python
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=SST-2

python -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/$TASK_NAME/ \
    --n_gpu 1 \
```
参数详细含义参考[README.md](../../glue)
Fine-tuning 在dev上的结果如压缩结果表格中Result那一列所示。

### 环境配置
压缩功能依赖PaddlePaddle 2.0及以上版本，以及最新版本的PaddleNLP和PaddleSlim.
```shell
pip install paddlenlp\>=2.0rc
pip install paddleslim==2.0.0 -i https://pypi.org/simple
```

### 压缩训练

```python
python -u ./run_glue_ofa.py --model_type bert \
          --model_name_or_path ${task_pretrained_model_dir} \
          --task_name $TASK_NAME --max_seq_length 128     \
          --batch_size 32       \
          --learning_rate 2e-5     \
          --num_train_epochs 6     \
          --logging_steps 10     \
          --save_steps 100     \
          --output_dir ./tmp/$TASK_NAME \
          --n_gpu 1 \
          --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5
```
其中参数释义如下：
- `model_type` 指示了模型类型，当前仅支持BERT模型。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `task_name` 表示 Fine-tuning 的任务。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `n_gpu` 表示使用的 GPU 卡数。若希望使用多卡训练，将其设置为指定数目即可；若为0，则使用CPU。
- `width_mult_list` 表示压缩训练过程中，对每层Transformer Block的宽度选择的范围。

压缩训练之后在dev上的结果如压缩结果表格中Result with PaddleSlim那一列所示，速度相比原始模型加速59%。

## 压缩原理

1. 对Fine-tuning得到模型通过计算参数及其梯度的乘积得到参数的重要性，把模型参数根据重要性进行重排序。
2. 超网络中最大的子网络选择和Bert-base模型网络结构一致的网络结构，其他小的子网络是对最大网络的进行不同的宽度选择来得到的，宽度选择具体指的是网络中的参数进行裁剪，所有子网络在整个训练过程中都是参数共享的。
2. 用重排序之后的模型参数作为超网络模型的初始化参数。
3. Fine-tuning之后的模型作为教师网络，超网络作为学生网络，进行知识蒸馏。

<p align="center">
<img src="./imgs/ofa_bert.jpg" width="950"/><br />
整体流程图
</p>

## 参考论文

1. Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, Qun Liu. DynaBERT: Dynamic BERT with Adaptive Width and Depth.
2. H. Cai, C. Gan, T. Wang, Z. Zhang, and S. Han. Once for all: Train one network and specialize it for efficient deployment.
