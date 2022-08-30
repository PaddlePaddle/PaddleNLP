# BERT

## 模型简介

[BERT](https://arxiv.org/abs/1810.04805) （Bidirectional Encoder Representations from Transformers）以[Transformer](https://arxiv.org/abs/1706.03762) 编码器为网络基本组件，使用掩码语言模型（Masked Language Model）和邻接句子预测（Next Sentence Prediction）两个任务在大规模无标注文本语料上进行预训练（pre-train），得到融合了双向内容的通用语义表示模型。以预训练产生的通用语义表示模型为基础，结合任务适配的简单输出层，微调（fine-tune）后即可应用到下游的NLP任务，效果通常也较直接在下游的任务上训练的模型更优。此前BERT即在[GLUE评测任务](https://gluebenchmark.com/tasks)上取得了SOTA的结果。

本项目是BERT在 Paddle 2.0上的开源实现，包含了预训练和[GLUE评测任务](https://gluebenchmark.com/tasks)上的微调代码。

## 快速开始

### 环境依赖

本教程除了需要安装PaddleNLP库，还需以下依赖

```text
h5py
```

### 数据准备

#### Pre-training数据准备

`create_pretraining_data.py` 是创建预训练程序所需数据的脚本。其以文本文件（使用换行符换行和空白符分隔，data目录下提供了部分示例数据）为输入，经由BERT tokenizer进行tokenize后再做生成sentence pair正负样本、掩码token等处理，最后输出hdf5格式的数据文件。使用方式如下：

```shell
python create_pretraining_data.py \
  --input_file=data/sample_text.txt \
  --output_file=data/training_data.hdf5 \
  --bert_model=bert-base-uncased \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

其中参数释义如下：
- `input_file` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有`.txt`文件。
- `output_file` 指定输出文件。
- `bert_model` 指定使用特定BERT模型对应的tokenizer进行tokenize处理。
- `max_seq_length` 指定最大句子长度，超过该长度将被截断，不足该长度的将会进行padding。
- `max_predictions_per_seq` 表示每个句子中会被mask的token的最大数目。
- `masked_lm_prob` 表示每个token被mask的概率。
- `random_seed` 指定随机种子。
- `dupe_factor` 指定输入数据被重复处理的次数，每次处理将重新产生随机mask。

使用以上预训练数据生成程序可以用于处理领域垂类数据后进行二次预训练。若需要使用BERT论文中预训练使用的英文Wiki和BookCorpus数据，可以参考[这里](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)进行处理，得到的数据可以直接接入本项目中的预训练程序使用。

#### Fine-tunning数据准备

##### GLUE评测任务数据

GLUE评测任务所含数据集已在paddlenlp中以API形式提供，无需预先准备，使用`run_glue.py`执行微调时将会自动下载。

### 执行Pre-training

#### GPU训练
```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 3 \
    --input_dir data/ \
    --output_dir pretrained_models/ \
    --logging_steps 1 \
    --save_steps 20000 \
    --max_steps 1000000 \
    --device gpu \
    --use_amp False
```

#### XPU训练
```shell
unset FLAGS_selected_xpus
python -m paddle.distributed.launch --xpus "0" run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 3 \
    --input_dir data/ \
    --output_dir pretrained_models/ \
    --logging_steps 1 \
    --save_steps 20000 \
    --max_steps 1000000 \
    --device xpu \
    --use_amp False
```
其中参数释义如下：
- `model_type` 指示了模型类型，使用BERT模型时设置为bert即可。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_predictions_per_seq` 表示每个句子中会被mask的token的最大数目，与创建预训练数据时的设置一致。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `adam_epsilon` 表示AdamW优化器中使用的epsilon值。
- `warmup_steps` 表示动态学习率热启的step数。
- `num_train_epochs` 表示训练轮数。
- `input_dir` 表示输入数据的目录，该目录下所有文件名中包含training的文件将被作为训练数据。
- `output_dir` 表示模型的保存目录。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `max_steps` 表示最大训练步数。若训练`num_train_epochs`轮包含的训练步数大于该值，则达到`max_steps`后就提前结束。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `use_amp` 指示是否启用自动混合精度训练。

**NOTICE**: 预训练时data目录存放的是经过 `create_pretraining_data.py` 处理后的数据，因此需要通过该数据处理脚本预先处理，否则预训练将会出现报错。

### 执行Fine-tunning

以GLUE中的SST-2任务为例，启动Fine-tuning的方式如下：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False
```

其中参数释义如下：
- `model_type` 指示了模型类型，使用BERT模型时设置为bert即可。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。注：`bert-base-uncased`等对应使用的预训练模型转自[huggingface/transformers](https://github.com/huggingface/transformers)，具体可参考当前目录下converter中的内容。
- `task_name` 表示Fine-tuning的任务。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `use_amp` 指示是否启用自动混合精度训练。

基于`bert-base-uncased`在GLUE各评测任务上Fine-tuning后，在验证集上有如下结果：

| Task  | Metric                       | Result            |
|:-----:|:----------------------------:|:-----------------:|
| SST2 | Accuracy                     |      0.92660      |
| QNLI  | Accuracy                     |      0.91707      |
| CoLA  | Mattehew's corr              |      0.59557      |
| MRPC  | F1/Accuracy                  |  0.91667/0.88235  |
| STSB | Person/Spearman corr         |  0.88847/0.88350  |
| QQP   | Accuracy/F1                  |  0.90581/0.87347  |
| MNLI  | Matched acc/MisMatched acc   |  0.84422/0.84825  |
| RTE   | Accuracy                     |      0.711191     |


### 预测

在Fine-tuning完成后，我们可以使用如下方式导出希望用来预测的模型：

```shell
python -u ./export_model.py \
    --model_type bert \
    --model_path bert-base-uncased \
    --output_path ./infer_model/model
```

其中参数释义如下：
- `model_type` 指示了模型类型，使用BERT模型时设置为bert即可。
- `model_path` 表示训练模型的保存路径，与训练时的`output_dir`一致。
- `output_path` 表示导出预测模型文件的前缀。保存时会添加后缀（`pdiparams`，`pdiparams.info`，`pdmodel`）；除此之外，还会在`output_path`包含的目录下保存tokenizer相关内容。

然后按照如下的方式进行GLUE中的评测任务进行预测（基于Paddle的[Python预测API](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/python_infer_cn.html)）：

```shell
python -u ./predict_glue.py \
    --task_name SST2 \
    --model_type bert \
    --model_path ./infer_model/model \
    --batch_size 32 \
    --max_seq_length 128
```

其中参数释义如下：
- `task_name` 表示Fine-tuning的任务。
- `model_type` 指示了模型类型，使用BERT模型时设置为bert即可。
- `model_path` 表示预测模型文件的前缀，和上一步导出预测模型中的`output_path`一致。
- `batch_size` 表示每个预测批次的样本数目。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。

同时支持使用输入样例数据的方式进行预测任务，这里仅以文本情感分类数据[SST-2](https://nlp.stanford.edu/sentiment/index.html)为例，输出样例数据的分类预测结果：

```shell
python -u ./predict.py \
    --model_path ./infer_model/model \
    --device gpu \
    --max_seq_length 128
```

其中参数释义如下：
- `model_path` 表示预测模型文件的前缀，和上一步导出预测模型中的`output_path`一致。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。

样例中的待预测数据返回输出的预测结果如下：

```text
Data: against shimmering cinematography that lends the setting the ethereal beauty of an asian landscape painting
 Label: positive
 Negative prob: 0.0004963805549778044
 Positive prob: 0.9995037317276001

Data: the situation in a well-balanced fashion
 Label: positive
 Negative prob: 0.000471479695988819
 Positive prob: 0.9995285272598267

Data: at achieving the modest , crowd-pleasing goals it sets for itself
 Label: positive
 Negative prob: 0.0019163173856213689
 Positive prob: 0.998083770275116

Data: so pat it makes your teeth hurt
 Label: negative
 Negative prob: 0.9988648295402527
 Positive prob: 0.0011351780267432332

Data: this new jangle of noise , mayhem and stupidity must be a serious contender for the title .
 Label: negative
 Negative prob: 0.9884825348854065
 Positive prob: 0.011517543345689774
```

## 扩展

上述的介绍是基于动态图的BERT的预训练任务和微调任务以及预测任务的实践过程，同时在我们也提供了基于PaddlePaddle Fleet API的静态图的BERT相关实践，在组网代码层面保持动静统一，在计算速度以及多机联合训练方面有着更优的性能，具体的细节可以参考 [BERT静态图](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/bert/static)  。
