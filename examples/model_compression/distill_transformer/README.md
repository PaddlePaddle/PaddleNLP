# Patient Knowledge Distillation for BERT Model Compression
以下是本例的简要目录结构及说明：
```
.
├── run_patient_kd.py     # 蒸馏脚本
├── data.py               # 定义了dataloader等数据读取接口
├── args.py               # 参数配置脚本
├── kd_loss.py            # 损失函数定义脚本
└── README.md             # 文档，本文件
```
## 简介
本目录下的实验是将特定任务下BERT base模型的知识蒸馏到层数较少的学生模型中，主要参考论文[《Patient Knowledge Distillation for BERT Model Compression》](https://arxiv.org/abs/1908.09355)实现。

在模型蒸馏中，较大的模型（在本例中是BERT base）通常被称为教师模型，较小的模型（在本例中是层数为6的BERT，下文都称BERT6）通常被称为学生模型。知识的蒸馏通常是通过让学生模型学习蒸馏相关的损失函数实现，在本实验中，损失函数由三个部分组成，分别是学生模型对真实标签下的交叉熵损失、学生模型输出的logits和教师模型输出概率之间的交叉熵、教师模型中间层与学生模型中间层的均方误差和。

本实验分为三个训练过程：在特定任务上对BERT6的fine-tuning、对BERT6的knowledge Distillation（用于对比蒸馏效果）、对BERT6的Patient-knowledge Distillation

## 数据、预训练模型介绍及获取

本实验使用GLUE中的SST-2、QQP以及中文情感分类数据集ChnSentiCorp中的训练集作为训练语料，用数据集中的验证集评估模型的效果。运行本目录下的实验，数据集会被自动下载到`paddlenlp.utils.env.DATA_HOME` 路径下，例如在linux系统下，例如对于GLUE中的QQP数据集，默认存储路径是`~/.paddlenlp/datasets/glue/QQP`，对于ChnSentiCorp数据集，则会下载到 `~/.paddlenlp/datasets/chnsenticorp`。

对于BERT的fine-tuning任务，本实验中使用了预训练模型`bert-bas-uncased`、`bert-wwm-ext-chinese`、`bert-base-chinese`。同样，这几个模型在训练时会被自动下载到`paddlenlp.utils.env.DATA_HOME`路径下。例如，对于`bert-base-uncased`模型，在linux系统下，会被下载到`~/.paddlenlp/models/bert-base-uncased`下。

## 蒸馏实验过程

### 训练较少层数的BERT FT(Fine-tuning)模型
如果训练的是6层的BERT，那么就使用bert-base-uncased的前6层hidden layer对模型的参数初始化。
以GLUE的SST-2任务为例，调用BERT fine-tune的训练脚本，配置如下的参数，训练SST-2任务：

```shell
export TASK_NAME=SST-2

CUDA_VISIBLE_DEVICES="1"  nohup python run_patient_kd.py \
    --task_name $TASK_NAME \
    --model_name_or_path bert-base-uncased \
    --teacher_finetuned_model_path ../distill_lstm/pretrained_models/SST-2/best_model_610 \
    --num_train_epochs 4 \
    --batch_size 32 \
    --alpha 0.0 \
    --T 5 \
    --beta 0 \
    --warmup_proportion 0.1 \
    --output_dir ft_models/$TASK_NAME \
    --logging_steps 10 \
    --save_steps 10 > ft.log &
```

```shell
export TASK_NAME=chnsenticorp
CUDA_VISIBLE_DEVICES="7" nohup python run_patient_kd.py \
    --task_name $TASK_NAME \
    --model_name_or_path bert-wwm-ext-chinese \
    --num_train_epochs 40 \
    --batch_size 32 \
    --teacher_finetuned_model_path ../distill_lstm/pretrained_models/chnsenticorp/best_bert_wwm_ext_model_880/ \
    --warmup_proportion 0.0 \
    --output_dir ft_models/$TASK_NAME \
    --logging_steps 100 \
    --save_steps 100 \
    --alpha 0.0 \
    --beta 0  > ft_$TASK_NAME.log &
```

训练完成之后，可将训练效果最好的模型保存在本项目下的`pretrained_models/$TASK_NAME/`下。模型目录下有`model_config.json`, `model_state.pdparams`, `tokenizer_config.json`及`vocab.txt`这几个文件。

### 训练BERT-KD(Knowledge Distillation)模型
export TASK_NAME="SST-2"
CUDA_VISIBLE_DEVICES="0" nohup python -u run_patient_kd.py \
    --task_name $TASK_NAME \
    --model_name_or_path bert-base-uncased \
    --teacher_finetuned_model_path ../distill_lstm/pretrained_models/SST-2/best_model_610 \
    --num_train_epochs 4 \
    --batch_size 32 \
    --beta 0 \
    --warmup_proportion 0.1 \
    --output_dir kd_models/$TASK_NAME \
    --logging_steps 10 \
    --save_steps 10 \
    --alpha 0.8 \
    --T 10 > kd_$TASK_NAME_0.8_10.log &

```shell
export TASK_NAME=chnsenticorp
CUDA_VISIBLE_DEVICES="0" nohup python run_patient_kd.py \
    --task_name $TASK_NAME \
    --model_name_or_path bert-wwm-ext-chinese \
    --num_train_epochs 30 \
    --batch_size 32 \
    --teacher_finetuned_model_path ../distill_lstm/pretrained_models/chnsenticorp/best_bert_wwm_ext_model_880/ \
    --warmup_proportion 0.1 \
    --output_dir ft_models/$TASK_NAME \
    --logging_steps 100 \
    --save_steps 100 \
    --beta 0 \
    --alpha 0.8 \
    --T 0.8  > kd_$TASK_NAME.log &

```

### 训练BERT-PKD(Patient Knowledge Distillation)模型
```shell
export TASK_NAME="SST-2"
CUDA_VISIBLE_DEVICES="2" nohup python run_patient_kd.py \
    --task_name $TASK_NAME \
    --model_name_or_path bert-base-uncased \
    --teacher_finetuned_model_path ../distill_lstm/pretrained_models/SST-2/best_model_610 \
    --num_train_epochs 8 \
    --batch_size 32 \
    --warmup_proportion 0.1 \
    --output_dir pkd_models/$TASK_NAME \
    --logging_steps 10 \
    --strategy skip \
    --alpha 0.9 \
    --T 10 \
    --beta 700 \
    --save_steps 10 > pkd2.log &

```

## 蒸馏实验结果
TODO
| Model           | SST-2(dev acc) |
| --------------- | -------------- |
| BERT12(Teacher) | 0.930046       |
| BERT6-FT        | 0.913991       |
| BERT6-KD        | 0.916284       |
| BERT6-PKD       | 0.919725       |

## 参考文献

Sun S, Cheng Y, Gan Z, Liu J. [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355)[J]. arXiv preprint arXiv:1908.09355, 2019.
