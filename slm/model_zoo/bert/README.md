# BERT

## 模型简介

[BERT](https://arxiv.org/abs/1810.04805) （Bidirectional Encoder Representations from Transformers）以[Transformer](https://arxiv.org/abs/1706.03762) 编码器为网络基本组件，使用掩码语言模型（Masked Language Model）和邻接句子预测（Next Sentence Prediction）两个任务在大规模无标注文本语料上进行预训练（pre-train），得到融合了双向内容的通用语义表示模型。以预训练产生的通用语义表示模型为基础，结合任务适配的简单输出层，微调（fine-tune）后即可应用到下游的 NLP 任务，效果通常也较直接在下游的任务上训练的模型更优。此前 BERT 即在[GLUE 评测任务](https://gluebenchmark.com/tasks)上取得了 SOTA 的结果。

本项目是 BERT 在 Paddle 2.0上的开源实现，包含了预训练和[GLUE 评测任务](https://gluebenchmark.com/tasks)上的微调代码。

## 快速开始

### 环境依赖

本教程除了需要安装 PaddleNLP 库，还需以下依赖

```text
h5py
```

### 数据准备

#### Pre-training 数据准备

`create_pretraining_data.py` 是创建预训练程序所需数据的脚本。其以文本文件（使用换行符换行和空白符分隔，data 目录下提供了部分示例数据）为输入，经由 BERT tokenizer 进行 tokenize 后再做生成 sentence pair 正负样本、掩码 token 等处理，最后输出 hdf5格式的数据文件。使用方式如下：

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
- `bert_model` 指定使用特定 BERT 模型对应的 tokenizer 进行 tokenize 处理。
- `max_seq_length` 指定最大句子长度，超过该长度将被截断，不足该长度的将会进行 padding。
- `max_predictions_per_seq` 表示每个句子中会被 mask 的 token 的最大数目。
- `masked_lm_prob` 表示每个 token 被 mask 的概率。
- `random_seed` 指定随机种子。
- `dupe_factor` 指定输入数据被重复处理的次数，每次处理将重新产生随机 mask。

使用以上预训练数据生成程序可以用于处理领域垂类数据后进行二次预训练。若需要使用 BERT 论文中预训练使用的英文 Wiki 和 BookCorpus 数据，可以参考[这里](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)进行处理，得到的数据可以直接接入本项目中的预训练程序使用。

#### Fine-tunning 数据准备

##### GLUE 评测任务数据

GLUE 评测任务所含数据集已在 paddlenlp 中以 API 形式提供，无需预先准备，使用`run_glue.py`执行微调时将会自动下载。

### 执行 Pre-training
<details>
<summary>GPU 训练</summary>
<pre><code>unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir data/ \
    --output_dir pretrained_models/ \
    --logging_steps 1 \
    --save_steps 20000 \
    --max_steps 1000000 \
    --device gpu \
    --use_amp False</code></pre>

其中参数释义如下：
- `model_type` 指示了模型类型，使用 BERT 模型时设置为 bert 即可。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_predictions_per_seq` 表示每个句子中会被 mask 的 token 的最大数目，与创建预训练数据时的设置一致。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `weight_decay` 表示 AdamW 优化器中使用的 weight_decay 的系数。
- `adam_epsilon` 表示 AdamW 优化器中使用的 epsilon 值。
- `warmup_steps` 表示动态学习率热启的 step 数。
- `input_dir` 表示输入数据的目录，该目录下所有文件名中包含 training 的文件将被作为训练数据。
- `output_dir` 表示模型的保存目录。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `max_steps` 表示最大训练步数，达到`max_steps`后就提前结束。注意，我们必须设置 `max_steps`。
- `device` 表示训练使用的设备, 'gpu'表示使用 GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用 CPU。
- `use_amp` 指示是否启用自动混合精度训练。
</details>

#### GPU 训练（Trainer 版本）
```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_pretrain_trainer.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --per_device_train_batch_size 32   \
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
    --fp16 False \
    --do_train
```

<details>
<summary>XPU 训练</summary>
<pre><code>unset FLAGS_selected_xpus
python -m paddle.distributed.launch --xpus "0" run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir data/ \
    --output_dir pretrained_models/ \
    --logging_steps 1 \
    --save_steps 20000 \
    --max_steps 1000000 \
    --device xpu \
    --use_amp False</code></pre>

其中参数释义如下：
- `model_type` 指示了模型类型，使用 BERT 模型时设置为 bert 即可。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_predictions_per_seq` 表示每个句子中会被 mask 的 token 的最大数目，与创建预训练数据时的设置一致。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `weight_decay` 表示 AdamW 优化器中使用的 weight_decay 的系数。
- `adam_epsilon` 表示 AdamW 优化器中使用的 epsilon 值。
- `warmup_steps` 表示动态学习率热启的 step 数。
- `input_dir` 表示输入数据的目录，该目录下所有文件名中包含 training 的文件将被作为训练数据。
- `output_dir` 表示模型的保存目录。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `max_steps` 表示最大训练步数，达到`max_steps`后就提前结束。注意，我们必须设置 `max_steps`。
- `device` 表示训练使用的设备, 'gpu'表示使用 GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用 CPU。
- `use_amp` 指示是否启用自动混合精度训练。
</details>

#### XPU 训练（Trainer 版本）
```shell
unset FLAGS_selected_xpus
python -m paddle.distributed.launch --xpus "0" run_pretrain_trainer.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --per_device_train_batch_size 32   \
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
    --fp16 False \
    --do_train
```
其中参数释义如下：
- `model_type` 指示了模型类型，使用 BERT 模型时设置为 bert 即可。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_predictions_per_seq` 表示每个句子中会被 mask 的 token 的最大数目，与创建预训练数据时的设置一致。
- `per_device_train_batch_size` 表示用于训练的每个 GPU 核心/CPU 的 batch 大小.（`int`，可选，默认为 8）
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `weight_decay` 表示 AdamW 优化器中使用的 weight_decay 的系数。
- `adam_epsilon` 表示 AdamW 优化器中使用的 epsilon 值。
- `warmup_steps` 表示动态学习率热启的 step 数。
- `num_train_epochs` 表示训练轮数。
- `input_dir` 表示输入数据的目录，该目录下所有文件名中包含 training 的文件将被作为训练数据。
- `output_dir` 表示模型的保存目录。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `max_steps` 表示最大训练步数，达到`max_steps`后就提前结束。注意，我们必须设置 `max_steps`。
- `device` 表示训练使用的设备, 'gpu'表示使用 GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用 CPU。
- `fp16` 是否使用 fp16 混合精度训练而不是 fp32 训练。(`bool`, 可选, 默认为 `False`)
- `do_train` 是否进行训练任务。(`bool`, 可选, 默认为 `False`)

**NOTICE**: 预训练时 data 目录存放的是经过 `create_pretraining_data.py` 处理后的数据，因此需要通过该数据处理脚本预先处理，否则预训练将会出现报错。

### 执行 Fine-tunning

以 GLUE 中的 SST-2任务为例，启动 Fine-tuning 的方式如下：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_glue_trainer.py \
    --model_name_or_path bert-base-uncased \
    --task_name SST2 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32   \
    --per_device_eval_batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu \
    --fp16 False\
    --do_train \
    --do_eval
```

其中参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。注：`bert-base-uncased`等对应使用的预训练模型转自[huggingface/transformers](https://github.com/huggingface/transformers)，具体可参考当前目录下 converter 中的内容。
- `task_name` 表示 Fine-tuning 的任务。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `per_device_train_batch_size` 表示用于训练的每个 GPU 核心/CPU 的 batch 大小.（`int`，可选，默认为 8）
- `per_device_eval_batch_size` 表示用于评估的每个 GPU 核心/CPU 的 batch 大小.（`int`，可选，默认为 8）
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用 GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用 CPU, 'npu'表示使用华为昇腾卡。
- `fp16` 是否使用 fp16 混合精度训练而不是 fp32 训练。(`bool`, 可选, 默认为 `False`)
- `do_train` 是否进行训练任务。(`bool`, 可选, 默认为 `False`)
- `do_eval` 是否进行评估任务。同上。(`bool`, 可选, 默认为 `False`)

基于`bert-base-uncased`在 GLUE 各评测任务上 Fine-tuning 后，在验证集上有如下结果：

| Task |           Metric           |     Result      |
|:----:|:--------------------------:|:---------------:|
| SST2 |          Accuracy          |     0.92660     |
| QNLI |          Accuracy          |     0.91707     |
| CoLA |      Mattehew's corr       |     0.59557     |
| MRPC |        F1/Accuracy         | 0.91667/0.88235 |
| STSB |    Person/Spearman corr    | 0.88847/0.88350 |
| QQP  |        Accuracy/F1         | 0.90581/0.87347 |
| MNLI | Matched acc/MisMatched acc | 0.84422/0.84825 |
| RTE  |          Accuracy          |    0.711191     |


### 预测

在 Fine-tuning 完成后，我们可以使用如下方式导出希望用来预测的模型：

```shell
python -u ./export_model.py \
    --model_type bert \
    --model_path bert-base-uncased \
    --output_path ./infer_model/model
```

其中参数释义如下：
- `model_type` 指示了模型类型，使用 BERT 模型时设置为 bert 即可。
- `model_path` 表示训练模型的保存路径，与训练时的`output_dir`一致。
- `output_path` 表示导出预测模型文件的前缀。保存时会添加后缀（`pdiparams`，`pdiparams.info`，`pdmodel`）；除此之外，还会在`output_path`包含的目录下保存 tokenizer 相关内容。

完成模型导出后，可以开始部署。`deploy/python/seq_cls_infer.py` 文件提供了 python 部署预测示例。可执行以下命令运行部署示例：

```shell
python deploy/python/seq_cls_infer.py --model_dir infer_model/ --device gpu --backend paddle
```

运行后预测结果打印如下：

```bash
[INFO] fastdeploy/runtime/runtime.cc(266)::CreatePaddleBackend	Runtime initialized with Backend::PDINFER in Device::GPU.
Batch id: 0, example id: 0, sentence1: against shimmering cinematography that lends the setting the ethereal beauty of an asian landscape painting, label: positive, negative prob: 0.0003, positive prob: 0.9997.
Batch id: 1, example id: 0, sentence1: the situation in a well-balanced fashion, label: positive, negative prob: 0.0002, positive prob: 0.9998.
Batch id: 2, example id: 0, sentence1: at achieving the modest , crowd-pleasing goals it sets for itself, label: positive, negative prob: 0.0017, positive prob: 0.9983.
Batch id: 3, example id: 0, sentence1: so pat it makes your teeth hurt, label: negative, negative prob: 0.9986, positive prob: 0.0014.
Batch id: 4, example id: 0, sentence1: this new jangle of noise , mayhem and stupidity must be a serious contender for the title ., label: negative, negative prob: 0.9806, positive prob: 0.0194.
```

更多详细用法可参考 [Python 部署](deploy/python/README.md)。

## 扩展

上述的介绍是基于动态图的 BERT 的预训练任务和微调任务以及预测任务的实践过程，同时在我们也提供了基于 PaddlePaddle Fleet API 的静态图的 BERT 相关实践，在组网代码层面保持动静统一，在计算速度以及多机联合训练方面有着更优的性能，具体的细节可以参考 [BERT 静态图](./static/)。
