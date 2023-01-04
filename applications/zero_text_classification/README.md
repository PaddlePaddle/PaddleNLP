# 统一标签文本分类

**目录**
- [1. 统一标签文本分类应用](#1)
- [2. 快速开始](#2)
    - [2.1 代码结构](#代码结构)
    - [2.2 数据标注](#数据标注)
    - [2.3 模型微调](#模型微调)
    - [2.4 模型评估](#模型评估)
    - [2.5 定制模型一键预测](#定制模型一键预测)
    - [2.6 实验指标](#实验指标)

<a name="1"></a>

## 1. 统一标签文本分类应用

本项目提供基于 UTC（Unified Tag Classification）微调的文本分类端到端应用方案，打通**数据标注-模型训练-模型调优-预测部署全流程**，可快速实现文本分类产品落地。

文本分类简单来说就是对给定的一个句子或一段文本使用分类模型分类。在文本分类的落地过程中通常面临领域多变、任务多样、数据稀缺等许多挑战。针对文本分类领域的痛点和难点，PaddleNLP统一标签文本分类应用 UTC 统一建模的思想，支持文本分类、情感分析、语义匹配等任务的统一训练，助力开发者简单高效实现多任务文本分类数据标注、训练、调优、上线，降低文本分类落地技术门槛。

**统一标签文本分类应用亮点：**

- **覆盖场景全面🎓：**  覆盖文本分类各类主流任务，支持多任务训练，满足开发者多样文本分类落地需求。
- **效果领先🏃：**  具有突出分类效果的UTC模型作为训练基座，提供良好的零样本和小样本学习能力。
- **简单易用：** 通过Taskflow实现三行代码可实现无标注数据的情况下进行快速调用，一行命令即可开启文本分类，轻松完成部署上线，降低多任务文本分类落地门槛。
- **高效调优✊：** 开发者无需机器学习背景知识，即可轻松上手数据标注及模型训练流程。

<a name="2"></a>

## 2. 快速开始

对于简单的文本分类可以直接使用```paddlenlp.Taskflow```实现零样本（zero-shot）分类，对于细分场景我们推荐使用定制功能（标注少量数据进行模型微调）以进一步提升效果。

<a name="代码结构"></a>

### 2.1 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── train.py          # 模型微调、压缩脚本
├── predict.py        # 模型评估脚本
└── README.md
```

<a name="数据标注"></a>

### 2.2 数据标注

我们推荐使用doccano数据标注工具进行标注，如果已有标注好的本地数据集，我们需要将数据集整理为文档要求的格式，详见[统一标签文本分类标注指南]()。

这里我们提供预先标注好的`医疗意图分类数据集`的文件，可以运行下面的命令行下载数据集，我们将展示如何使用数据转化脚本生成训练/验证/测试集文件，并使用UTC模型进行微调。

下载医疗意图分类数据集：


```shell
wget https://bj.bcebos.com/paddlenlp/datasets/medical.tar.gz
tar -xvf medical.tar.gz
mv medical data
rm medical.tar.gz
```

生成训练/验证集文件：
```shell
python ../doccano.py \
    --doccano_file ./data/doccano.json \
    --save_dir ./data \
    --splits 0.76 0.24 0 \
    --label_file ./data/label.txt
```
多任务训练场景可分别进行数据转换再进行混合。

<a name="模型微调"></a>

### 2.3 模型微调

推荐使用 PromptTrainer API 对模型进行微调，该 API 封装了提示定义功能，且继承自 [Trainer API ](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md) 。只需输入模型、数据集等就可以使用 Trainer API 高效快速地进行预训练、微调等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，Trainer API 还针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

使用下面的命令，使用 `utc-large` 作为预训练模型进行模型微调，将微调后的模型保存至`$finetuned_model`：

单卡启动：

```shell
python train.py  \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path utc-large \
    --output_dir ./checkpoint/model_best \
    --dataset_path ./data/ \
    --max_seq_length 512  \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model macro_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1
```

如果在GPU环境中使用，可以指定gpus参数进行多卡训练：

```shell
python -u -m paddle.distributed.launch --gpus "0,1" train.py \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path utc-large \
    --output_dir ./checkpoint/model_best \
    --dataset_path ./data/ \
    --max_seq_length 512  \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model macro_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1
```

该示例代码中由于设置了参数 `--do_eval`，因此在训练完会自动进行评估。

可配置参数说明：
* `device`: 训练设备，可选择 'cpu'、'gpu' 其中的一种；默认为 GPU 训练。
* `logging_steps`: 训练过程中日志打印的间隔 steps 数，默认10。
* `save_steps`: 训练过程中保存模型 checkpoint 的间隔 steps 数，默认100。
* `eval_steps`: 训练过程中保存模型 checkpoint 的间隔 steps 数，默认100。
* `seed`：全局随机种子，默认为 42。
* `model_name_or_path`：进行 few shot 训练使用的预训练模型。默认为 "utc-large"。
* `output_dir`：必须，模型训练或压缩后保存的模型目录；默认为 `None` 。
* `dev_path`：开发集路径；默认为 `None` 。
* `max_seq_len`：文本最大切分长度，包括标签的输入超过最大长度时会对输入文本进行自动切分，标签部分不可切分，默认为512。
* `per_device_train_batch_size`:用于训练的每个 GPU 核心/CPU 的batch大小，默认为8。
* `per_device_eval_batch_size`:用于评估的每个 GPU 核心/CPU 的batch大小，默认为8。
* `num_train_epochs`: 训练轮次，使用早停法时可以选择 100；默认为10。
* `learning_rate`：训练最大学习率，UTC 推荐设置为 1e-5；默认值为3e-5。
* `do_train`:是否进行微调训练，设置该参数表示进行微调训练，默认不设置。
* `do_eval`:是否进行评估，设置该参数表示进行评估，默认不设置。
* `do_export`:是否进行导出，设置该参数表示进行静态图导出，默认不设置。
* `export_model_dir`:静态图导出地址，默认为None。
* `overwrite_output_dir`： 如果 `True`，覆盖输出目录的内容。如果 `output_dir` 指向检查点目录，则使用它继续训练。
* `disable_tqdm`： 是否使用tqdm进度条。
* `metric_for_best_model`：最优模型指标, UTC 推荐设置为 `macro_f1`，默认为None。
* `load_best_model_at_end`：训练结束后是否加载最优模型，通常与`metric_for_best_model`配合使用，默认为False。
* `save_total_limit`：如果设置次参数，将限制checkpoint的总数。删除旧的checkpoints `输出目录`，默认为None。

<a name="模型评估"></a>

### 2.4 模型评估

通过运行以下命令进行模型评估预测：

```shell
python predict.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --per_device_eval_batch_size 2 \
    --max_seq_len 512 \
    --output_dir ./checkpoint_test \
```

可配置参数说明：

- `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`及配置文件`model_config.json`。
- `test_path`: 进行评估的测试集文件。
- `per_device_eval_batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。

<a name="定制模型一键预测"></a>

### 2.5 定制模型一键预测

`paddlenlp.Taskflow`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件`model_state.pdparams`。

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow
>>> my_cls = Taskflow("zero_text_classification", choices=[], task_path='./checkpoint/model_best')
>>> pprint(my_cls(""))


```

<a name="实验指标"></a>

### 2.6 实验指标

医疗意图分类数据集实验指标：

|  |  Accuracy  | Micro F1 | Macro F1  |
  | :---: | :--------: | :--------: | :--------: |
  | 0-shot | |  |  |
  | 5-shot | |  |  |
  | 10-shot | | | |
  | full-set | | | |


商业版本UTC模型支持极多标签分类，可联系xxx使用，零样本和小样本在业务数据集上指标：


|  |  Accuracy  | Micro F1 | Macro F1  |
  | :---: | :--------: | :--------: | :--------: |
  | 0-shot | |  |  |
  | 5-shot | |  |  |
  | 10-shot | | | |
  | full-set | | | |
