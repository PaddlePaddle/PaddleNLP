# 文档信息抽取

**目录**
- [1. 文档信息抽取应用](#1)
- [2. 快速开始](#2)
  - [2.1 代码结构](#代码结构)
  - [2.2 数据标注](#数据标注)
  - [2.3 模型微调](#模型微调)
  - [2.4 模型评估](#模型评估)
  - [2.5 定制模型一键预测](#定制模型一键预测)
  - [2.6 实验指标](#实验指标)

<a name="1"></a>

## 1. 文档信息抽取应用

本项目提供基于UIE微调的文档抽取端到端应用方案，打通**数据标注-模型训练-模型调优-预测部署全流程**，可快速实现文档信息抽取产品落地。

信息抽取通俗地说就是从给定的文本/图片等输入数据中抽取出结构化信息的过程。在信息抽取的落地过程中通常面临领域多变、任务多样、数据稀缺等许多挑战。针对信息抽取领域的难点和痛点，PaddleNLP信息抽取应用UIE统一建模的思想，提供了文档信息抽取产业级应用方案，支持**文档/图片/表格和纯文本场景下实体、关系、事件、观点等不同任务信息抽取**。该应用**不限定行业领域和抽取目标**，可实现从产品原型研发、业务POC阶段到业务落地、迭代阶段的无缝衔接，助力开发者实现特定领域抽取场景的快速适配与落地。

**文档信息抽取应用亮点：**

- **覆盖场景全面🎓：** 覆盖文档信息抽取各类主流任务，支持多语言，满足开发者多样信息抽取落地需求。
- **效果领先🏃：** 以在多模态信息抽取上有突出效果的模型UIE-X作为训练基座，具有广泛成熟的实践应用性。
- **简单易用⚡：** 通过Taskflow实现三行代码可实现无标注数据的情况下进行快速调用，一行命令即可开启信息抽取训练，轻松完成部署上线，降低信息抽取技术落地门槛。
- **高效调优✊：** 开发者无需机器学习背景知识，即可轻松上手数据标注及模型训练流程。

<a name="2"></a>

## 2. 快速开始

对于简单的抽取目标可以直接使用```paddlenlp.Taskflow```实现零样本（zero-shot）抽取，对于细分场景我们推荐使用定制功能（标注少量数据进行模型微调）以进一步提升效果。

<a name="代码结构"></a>

### 2.1 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── finetune.py       # 模型微调、压缩脚本
├── evaluate.py       # 模型评估脚本
└── README.md
```

<a name="数据标注"></a>

### 2.2 数据标注
我们推荐使用 [Label Studio](https://labelstud.io/) 进行文档信息抽取数据标注，本项目打通了从数据标注到训练的通道，也即Label Studio导出数据可以通过 [label_studio.py](../label_studio.py) 脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。标注方法的详细介绍请参考 [Label Studio数据标注指南](../label_studio.md)。

这里我们提供预先标注好的`增值税发票数据集`的文件，可以运行下面的命令行下载数据集，我们将展示如何使用数据转化脚本生成训练/验证/测试集文件，并使用UIE-X模型进行微调。

下载增值税发票数据集：
```shell
wget https://paddlenlp.bj.bcebos.com/datasets/tax.tar.gz
tar -zxvf tax.tar.gz
mv tax data
rm tax.tar.gz
```

生成训练/验证集文件：
```shell
python ../label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --save_dir ./data \
    --splits 0.8 0.2 0\
    --task_type ext
```

生成训练/验证集文件，可以使用PPStructure的布局分析优化OCR结果的排序：
```shell
python ../label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --save_dir ./data \
    --splits 0.8 0.2 0\
    --task_type ext \
    --layout_analysis True
```

更多不同类型任务（含实体抽取、关系抽取、文档分类等）的标注规则及参数说明，请参考[Label Studio数据标注指南](../label_studio_doc.md)。

<a name="模型微调"></a>

### 2.3 模型微调

推荐使用 [Trainer API ](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md) 对模型进行微调。只需输入模型、数据集等就可以使用 Trainer API 高效快速地进行预训练、微调和模型压缩等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，Trainer API 还针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

使用下面的命令，使用 `uie-x-base` 作为预训练模型进行模型微调，将微调后的模型保存至`./checkpoint/model_best`：

单卡启动：

```shell
python finetune.py  \
    --device gpu \
    --logging_steps 5 \
    --save_steps 25 \
    --eval_steps 25 \
    --seed 42 \
    --model_name_or_path uie-x-base \
    --output_dir ./checkpoint/model_best \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_len 512  \
    --per_device_train_batch_size  8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --label_names 'start_positions' 'end_positions' \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1
```

如果在GPU环境中使用，可以指定gpus参数进行多卡训练：

```shell
python -u -m paddle.distributed.launch --gpus "0" finetune.py \
    --device gpu \
    --logging_steps 5 \
    --save_steps 25 \
    --eval_steps 25 \
    --seed 42 \
    --model_name_or_path uie-x-base \
    --output_dir ./checkpoint/model_best \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_len 512  \
    --per_device_train_batch_size  8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --label_names 'start_positions' 'end_positions' \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
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
* `model_name_or_path`：进行 few shot 训练使用的预训练模型。默认为 "uie-x-base"。
* `output_dir`：必须，模型训练或压缩后保存的模型目录；默认为 `None` 。
* `train_path`：训练集路径；默认为 `None` 。
* `dev_path`：开发集路径；默认为 `None` 。
* `max_seq_len`：文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
* `per_device_train_batch_size`:用于训练的每个 GPU 核心/CPU 的batch大小，默认为8。
* `per_device_eval_batch_size`:用于评估的每个 GPU 核心/CPU 的batch大小，默认为8。
* `num_train_epochs`: 训练轮次，使用早停法时可以选择 100；默认为10。
* `learning_rate`：训练最大学习率，UIE-X 推荐设置为 1e-5；默认值为3e-5。
* `label_names`：训练数据标签label的名称，UIE-X 设置为'start_positions' 'end_positions'；默认值为None。
* `do_train`:是否进行微调训练，设置该参数表示进行微调训练，默认不设置。
* `do_eval`:是否进行评估，设置该参数表示进行评估，默认不设置。
* `do_export`:是否进行导出，设置该参数表示进行静态图导出，默认不设置。
* `export_model_dir`:静态图导出地址，默认为None。
* `overwrite_output_dir`： 如果 `True`，覆盖输出目录的内容。如果 `output_dir` 指向检查点目录，则使用它继续训练。
* `disable_tqdm`： 是否使用tqdm进度条。
* `metric_for_best_model`：最优模型指标,UIE-X 推荐设置为 `eval_f1`，默认为None。
* `load_best_model_at_end`：训练结束后是否加载最优模型，通常与`metric_for_best_model`配合使用，默认为False。
* `save_total_limit`：如果设置次参数，将限制checkpoint的总数。删除旧的checkpoints `输出目录`，默认为None。

<a name="模型评估"></a>

### 2.4 模型评估

```shell
python evaluate.py \
    --device "gpu" \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --output_dir ./checkpoint/model_best \
    --label_names 'start_positions' 'end_positions'\
    --max_seq_len 512 \
    --per_device_eval_batch_size 16
```
评估方式说明：采用单阶段评价的方式，即关系抽取、事件抽取等需要分阶段预测的任务对每一阶段的预测结果进行分别评价。验证/测试集默认会利用同一层级的所有标签来构造出全部负例。

可开启`debug`模式对每个正例类别分别进行评估，该模式仅用于模型调试：

```shell
python evaluate.py \
    --device "gpu" \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --output_dir ./checkpoint/model_best \
    --label_names 'start_positions' 'end_positions'\
    --max_seq_len 512 \
    --per_device_eval_batch_size 16 \
    --debug True
```

输出结果：
```text
[2022-11-14 09:41:18,424] [    INFO] - ***** Running Evaluation *****
[2022-11-14 09:41:18,424] [    INFO] -   Num examples = 160
[2022-11-14 09:41:18,424] [    INFO] -   Pre device batch size = 4
[2022-11-14 09:41:18,424] [    INFO] -   Total Batch size = 4
[2022-11-14 09:41:18,424] [    INFO] -   Total prediction steps = 40
[2022-11-14 09:41:26,451] [    INFO] - -----Evaluate model-------
[2022-11-14 09:41:26,451] [    INFO] - Class Name: ALL CLASSES
[2022-11-14 09:41:26,451] [    INFO] - Evaluation Precision: 0.94521 | Recall: 0.88462 | F1: 0.91391
[2022-11-14 09:41:26,451] [    INFO] - -----------------------------
[2022-11-14 09:41:26,452] [    INFO] - ***** Running Evaluation *****
[2022-11-14 09:41:26,452] [    INFO] -   Num examples = 8
[2022-11-14 09:41:26,452] [    INFO] -   Pre device batch size = 4
[2022-11-14 09:41:26,452] [    INFO] -   Total Batch size = 4
[2022-11-14 09:41:26,452] [    INFO] -   Total prediction steps = 2
[2022-11-14 09:41:26,692] [    INFO] - Class Name: 开票日期
[2022-11-14 09:41:26,692] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-11-14 09:41:26,692] [    INFO] - -----------------------------
[2022-11-14 09:41:26,693] [    INFO] - ***** Running Evaluation *****
[2022-11-14 09:41:26,693] [    INFO] -   Num examples = 8
[2022-11-14 09:41:26,693] [    INFO] -   Pre device batch size = 4
[2022-11-14 09:41:26,693] [    INFO] -   Total Batch size = 4
[2022-11-14 09:41:26,693] [    INFO] -   Total prediction steps = 2
[2022-11-14 09:41:26,952] [    INFO] - Class Name: 名称
[2022-11-14 09:41:26,952] [    INFO] - Evaluation Precision: 0.87500 | Recall: 0.87500 | F1: 0.87500
[2022-11-14 09:41:26,952] [    INFO] - -----------------------------
...
```

可配置参数：
* `device`: 评估设备，可选择 'cpu'、'gpu' 其中的一种；默认为 GPU 评估。
* `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`及配置文件`model_config.json`。
* `test_path`: 进行评估的测试集文件。
* `label_names`：训练数据标签label的名称，UIE-X 设置为'start_positions' 'end_positions'；默认值为None。
* `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
* `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
* `per_device_eval_batch_size`:用于评估的每个 GPU 核心/CPU 的batch大小，默认为8。
* `debug`: 是否开启debug模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。
* `schema_lang`: 选择schema的语言，可选有`ch`和`en`。默认为`ch`，英文数据集请选择`en`。

<a name="定制模型一键预测"></a>

### 2.5 定制模型一键预测

`paddlenlp.Taskflow`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件`model_state.pdparams`。

```python
from pprint import pprint
from paddlenlp import Taskflow
schema = ['开票日期', '名称', '纳税人识别号', '开户行及账号', '金额', '价税合计', 'No', '税率', '地址、电话', '税额']
my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best', precison='fp16')
```

我们可以根据设置的`schema`，对指定的`doc_path`文档进行信息抽取：

```python
doc_path = "./data/images/b201.jpg"
pprint(my_ie({"doc": doc_path}))
```

<a name="实验指标"></a>

### 2.6 实验指标

我们在自标注的增值税数据集上进行实验：



  |  |  Precision  | Recall | F1 Score |
  | :---: | :--------: | :--------: | :--------: |
  | 0-shot| 0.44944 | 0.51282 | 0.47904 |
  | 5-shot| 0.86076 | 0.87179 | 0.86624 |
  | 10-shot| 0.92405 | 0.93590 |  0.92994 |
  | 20-shot| 0.93671 | 0.94872 | 0.94268 |
  | 30-shot|  0.96154  | 0.96154  | 0.96154 |
  | 30-shot+PPStructure| 0.98718  | 0.96250 |  0.97468 |


n-shot表示训练集包含n张标注图片数据进行模型微调，实验表明UIE-X可以通过少量数据（few-shot）和PPStructure的布局分析进一步提升结果。
