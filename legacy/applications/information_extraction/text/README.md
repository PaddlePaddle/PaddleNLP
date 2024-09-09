简体中文 | [English](README_en.md)

# 文本信息抽取

**目录**
- [1. 文本信息抽取应用](#1)
- [2. 快速开始](#2)
  - [2.1 代码结构](#代码结构)
  - [2.2 数据标注](#数据标注)
  - [2.3 模型微调](#模型微调)
  - [2.4 模型评估](#模型评估)
  - [2.5 定制模型一键预测](#定制模型一键预测)
  - [2.6 实验指标](#实验指标)
  - [2.7 封闭域蒸馏](#封闭域蒸馏)

<a name="1"></a>

## 1. 文本信息抽取应用

本项目提供基于UIE微调的纯文本抽取端到端应用方案，打通**数据标注-模型训练-模型调优-预测部署全流程**，可快速实现文档信息抽取产品落地。

信息抽取通俗地说就是从给定的文本/图片等输入数据中抽取出结构化信息的过程。在信息抽取的落地过程中通常面临领域多变、任务多样、数据稀缺等许多挑战。针对信息抽取领域的难点和痛点，PaddleNLP信息抽取应用UIE统一建模的思想，提供了文档信息抽取产业级应用方案，支持**文档/图片/表格和纯文本场景下实体、关系、事件、观点等不同任务信息抽取**。该应用**不限定行业领域和抽取目标**，可实现从产品原型研发、业务POC阶段到业务落地、迭代阶段的无缝衔接，助力开发者实现特定领域抽取场景的快速适配与落地。

**文本信息抽取应用亮点：**

- **覆盖场景全面🎓：** 覆盖文本信息抽取各类主流任务，支持多语言，满足开发者多样信息抽取落地需求。
- **效果领先🏃：** 以在纯文本具有突出效果的UIE系列模型作为训练基座，提供多种尺寸的预训练模型满足不同需求，具有广泛成熟的实践应用性。
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

我们推荐使用 [Label Studio](https://labelstud.io/) 进行文本信息抽取数据标注，本项目打通了从数据标注到训练的通道，也即Label Studio导出数据可以通过 [label_studio.py](../label_studio.py) 脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。标注方法的详细介绍请参考 [Label Studio数据标注指南](../label_studio_text.md)。

这里我们提供预先标注好的`军事关系抽取数据集`的文件，可以运行下面的命令行下载数据集，我们将展示如何使用数据转化脚本生成训练/验证/测试集文件，并使用UIE模型进行微调。

下载军事关系抽取数据集：

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/military.tar.gz
tar -xvf military.tar.gz
mv military data
rm military.tar.gz
```

生成训练/验证集文件：
```shell
python ../label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --save_dir ./data \
    --splits 0.76 0.24 0 \
    --negative_ratio 3 \
    --task_type ext
```

更多不同类型任务（含实体抽取、关系抽取、文档分类等）的标注规则及参数说明，请参考[Label Studio数据标注指南](../label_studio_text.md)。


<a name="模型微调"></a>

### 2.3 模型微调

推荐使用 [Trainer API ](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md) 对模型进行微调。只需输入模型、数据集等就可以使用 Trainer API 高效快速地进行预训练、微调和模型压缩等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，Trainer API 还针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

使用下面的命令，使用 `uie-base` 作为预训练模型进行模型微调，将微调后的模型保存至`$finetuned_model`：

单卡启动：

```shell
python finetune.py  \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path uie-base \
    --output_dir ./checkpoint/model_best \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_len 512  \
    --per_device_train_batch_size  16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
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
python -u -m paddle.distributed.launch --gpus "0,1" finetune.py \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path uie-base \
    --output_dir ./checkpoint/model_best \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_len 512  \
    --per_device_train_batch_size  8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
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
* `device`: 训练设备，可选择 'cpu'、'gpu'、'npu' 其中的一种；默认为 GPU 训练。
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

通过运行以下命令进行模型评估：

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --device gpu \
    --batch_size 16 \
    --max_seq_len 512
```

通过运行以下命令对 UIE-M 进行模型评估：

```
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --device gpu \
    --max_seq_len 512 \
    --multilingual
```

评估方式说明：采用单阶段评价的方式，即关系抽取、事件抽取等需要分阶段预测的任务对每一阶段的预测结果进行分别评价。验证/测试集默认会利用同一层级的所有标签来构造出全部负例。

可开启`debug`模式对每个正例类别分别进行评估，该模式仅用于模型调试：

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --debug
```

输出打印示例：

```text
[2022-11-21 12:48:41,794] [    INFO] - -----------------------------
[2022-11-21 12:48:41,795] [    INFO] - Class Name: 武器名称
[2022-11-21 12:48:41,795] [    INFO] - Evaluation Precision: 0.96667 | Recall: 0.96667 | F1: 0.96667
[2022-11-21 12:48:44,093] [    INFO] - -----------------------------
[2022-11-21 12:48:44,094] [    INFO] - Class Name: X的产国
[2022-11-21 12:48:44,094] [    INFO] - Evaluation Precision: 1.00000 | Recall: 0.99275 | F1: 0.99636
[2022-11-21 12:48:46,474] [    INFO] - -----------------------------
[2022-11-21 12:48:46,475] [    INFO] - Class Name: X的研发单位
[2022-11-21 12:48:46,475] [    INFO] - Evaluation Precision: 0.77519 | Recall: 0.64935 | F1: 0.70671
[2022-11-21 12:48:48,800] [    INFO] - -----------------------------
[2022-11-21 12:48:48,801] [    INFO] - Class Name: X的类型
[2022-11-21 12:48:48,801] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
```

可配置参数说明：

- `device`: 评估设备，可选择 'cpu'、'gpu'、'npu' 其中的一种；默认为 GPU 评估。
- `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`及配置文件`model_config.json`。
- `test_path`: 进行评估的测试集文件。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `debug`: 是否开启debug模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。
- `multilingual`: 是否是跨语言模型，默认关闭。
- `schema_lang`: 选择schema的语言，可选有`ch`和`en`。默认为`ch`，英文数据集请选择`en`。

<a name="定制模型一键预测"></a>

### 2.5 定制模型一键预测

`paddlenlp.Taskflow`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件`model_state.pdparams`。

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> schema = {"武器名称": ["产国", "类型", "研发单位"]}
# 设定抽取目标和定制化模型权重路径
>>> my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
>>> pprint(my_ie("威尔哥（Virgo）减速炸弹是由瑞典FFV军械公司专门为瑞典皇家空军的攻击机实施低空高速轰炸而研制，1956年开始研制，1963年进入服役，装备于A32“矛盾”、A35“龙”、和AJ134“雷”攻击机，主要用于攻击登陆艇、停放的飞机、高炮、野战火炮、轻型防护装甲车辆以及有生力量。"))
[{'武器名称': [{'end': 14,
            'probability': 0.9998632702221926,
            'relations': {'产国': [{'end': 18,
                                  'probability': 0.9998815094394331,
                                  'start': 16,
                                  'text': '瑞典'}],
                          '研发单位': [{'end': 25,
                                    'probability': 0.9995875123178521,
                                    'start': 18,
                                    'text': 'FFV军械公司'}],
                          '类型': [{'end': 14,
                                  'probability': 0.999877336059086,
                                  'start': 12,
                                  'text': '炸弹'}]},
            'start': 0,
            'text': '威尔哥（Virgo）减速炸弹'}]}]
```

<a name="实验指标"></a>

### 2.6 实验指标

军事关系抽取数据集实验指标：

  |  |  Precision  | Recall | F1 Score |
  | :---: | :--------: | :--------: | :--------: |
  | 0-shot | 0.64634| 0.53535 | 0.58564 |
  | 5-shot | 0.89474 | 0.85000 | 0.87179 |
  | 10-shot | 0.92793 | 0.85833 | 0.89177 |
  | full-set | 0.93103 | 0.90000 | 0.91525 |


<a name="封闭域蒸馏"></a>

### 2.7 封闭域蒸馏

在一些工业应用场景中对性能的要求较高，模型若不能有效压缩则无法实际应用。因此，我们基于数据蒸馏技术构建了UIE Slim数据蒸馏系统。其原理是通过数据作为桥梁，将UIE模型的知识迁移到封闭域信息抽取小模型，以达到精度损失较小的情况下却能达到大幅度预测速度提升的效果。详细介绍请参考[UIE Slim 数据蒸馏](./data_distill/README.md)
