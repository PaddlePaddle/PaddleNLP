# 使用PaddleNLP完成中文命名实体识别

## 1. 简介

MSRA-NER 数据集由微软亚研院发布，其目标是识别文本中具有特定意义的实体，主要包括人名、地名、机构名等。示例如下：

```
不\002久\002前\002，\002中\002国\002共\002产\002党\002召\002开\002了\002举\002世\002瞩\002目\002的\002第\002十\002五\002次\002全\002国\002代\002表\002大\002会\002。    O\002O\002O\002O\002B-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002O\002O\002O\002O\002O\002O\002O\002O\002B-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002I-ORG\002O
这\002次\002代\002表\002大\002会\002是\002在\002中\002国\002改\002革\002开\002放\002和\002社\002会\002主\002义\002现\002代\002化\002建\002设\002发\002展\002的\002关\002键\002时\002刻\002召\002开\002的\002历\002史\002性\002会\002议\002。    O\002O\002O\002O\002O\002O\002O\002O\002B-LOC\002I-LOC\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O\002O
```

PaddleNLP集成的数据集MSRA-NER数据集对文件格式做了调整：每一行文本、标签以特殊字符"\t"进行分隔，每个字之间以特殊字符"\002"分隔。

## 快速开始

### 模型训练

#### 单卡训练

```shell
export CUDA_VISIBLE_DEVICES=0

python -u ./train.py \
    --model_type bert \
    --model_name_or_path bert-base-multilingual-uncased \
    --dataset msra_ner \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/msra_ner/ \
    --device gpu
```

其中参数释义如下：
- `model_type`: 指定模型的类型，可选的有 bert、ernie、ernie-ctm。
- `model_name_or_path`: 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer，支持[PaddleNLP Transformer API](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer) 中除ernie-gen以外的所有模型。若使用其他系列模型，需修改脚本导入相应的Task和Tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `dataset`: 目前支持 msra_ner 和 peoples_daily_ner 数据集。
- `max_seq_length`: 表示最大句子长度，超过该长度将被截断。
- `batch_size`: 表示每次迭代**每张卡**上的样本数目。
- `learning_rate`: 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs`: 表示训练轮数。
- `logging_steps`: 表示日志打印间隔。
- `save_steps`: 表示模型保存及评估间隔。
- `output_dir`: 表示模型保存路径。
- `device`: 训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。

#### 多卡训练
```shell
python -m paddle.distributed.launch --gpus "0,1" ./train.py \
    --model_type bert \
    --model_name_or_path bert-base-multilingual-uncased \
    --dataset msra_ner \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/msra_ner/ \
    --device gpu
```


训练过程将按照 `logging_steps` 和 `save_steps` 的设置打印如下日志：

```
global step 3996, epoch: 2, batch: 1184, loss: 0.008593, speed: 4.15 step/s
global step 3997, epoch: 2, batch: 1185, loss: 0.008453, speed: 4.17 step/s
global step 3998, epoch: 2, batch: 1186, loss: 0.002294, speed: 4.19 step/s
global step 3999, epoch: 2, batch: 1187, loss: 0.005351, speed: 4.16 step/s
global step 4000, epoch: 2, batch: 1188, loss: 0.004734, speed: 4.18 step/s
eval loss: 0.006829, precision: 0.908957, recall: 0.926683, f1: 0.917734
```

使用以上命令进行单卡 Fine-tuning ，在验证集上有如下结果：
 Metric                       | Result      |
------------------------------|-------------|
Precision                     | 0.908957    |
Recall                        | 0.926683    |
F1                            | 0.917734    |

### 模型评估
目前支持bert类型模型，其他模型可修改为对应的Task和Tokenizer。支持msra_ner数据集。
```shell
export CUDA_VISIBLE_DEVICES=0

python -u ./eval.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 32 \
    --device gpu \
    --init_checkpoint_path tmp/msra_ner/model_500.pdparams
```

其中参数释义如下：
- `model_name_or_path`: 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_seq_length`: 表示最大句子长度，超过该长度将被截断。
- `batch_size`: 表示每次迭代**每张卡**上的样本数目。
- `use_gpu`: 是否使用GPU。
- `init_checkpoint_path`: 模型加载路径。

### 模型预测

目前支持bert类型模型，其他模型可修改为对应的Task和Tokenizer。支持msra_ner数据集。
```shell
export CUDA_VISIBLE_DEVICES=0

python -u ./predict.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 32 \
    --device gpu \
    --init_checkpoint_path tmp/msra_ner/model_500.pdparams
```

### 使用其它预训练模型

请参考[Transformer API文档](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer) 了解更多PaddleNLP支持的预训练模型信息，并更换`--model_name_or_path`参数即可对比其他预训练模型的效果。

## Reference

- [The third international Chinese language processing bakeoff: Word segmentation and named entity recognition](https://faculty.washington.edu/levow/papers/sighan06.pdf)
