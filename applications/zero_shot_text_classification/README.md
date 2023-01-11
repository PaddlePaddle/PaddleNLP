# 零样本文本分类

**目录**
- [1. 零样本文本分类应用](#1)
- [2. 快速开始](#2)
    - [2.1 代码结构](#代码结构)
    - [2.2 数据标注](#数据标注)
    - [2.3 模型微调](#模型微调)
    - [2.4 模型评估](#模型评估)
    - [2.5 定制模型一键预测](#定制模型一键预测)
    - [2.6 模型部署](#模型部署)
    - [2.7 实验指标](#实验指标)

<a name="1"></a>

## 1. 零样本文本分类应用

本项目提供基于通用文本分类 UTC（Universial Text Classification） 模型微调的文本分类端到端应用方案，打通**数据标注-模型训练-模型调优-预测部署全流程**，可快速实现文本分类产品落地。

<div align="center">
    <img width="700" alt="UTC模型结构图" src="https://user-images.githubusercontent.com/25607475/211755652-dac155ca-649e-470c-ac8b-06156b444b58.png">
</div>

文本分类简单来说就是对给定的句子或文本使用分类模型分类。在文本分类的落地过程中通常面临领域多变、任务多样、数据稀缺等许多挑战。针对文本分类领域的痛点和难点，PaddleNLPl零样本文本分类应用 UTC 通过统一语义匹配方式 USM（Unified Semantic Matching）统一建模标签与文本的语义匹配能力，具备低资源迁移能力，支持通用分类、评论情感分析、语义相似度计算、蕴含推理、多项式阅读理解等众多“泛分类”任务，助力开发者简单高效实现多任务文本分类数据标注、训练、调优、上线，降低文本分类落地技术门槛。


**零样本文本分类应用亮点：**

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
├── deploy/simple_serving/ # 模型部署脚本
├── utils.py               # 数据处理工具
├── run_train.py           # 模型微调脚本
├── run_eval.py            # 模型评估脚本
├── label_studio.py        # 数据格式转换脚本
├── label_studio_text.md   # 数据标注说明文档
└── README.md
```

<a name="数据标注"></a>

### 2.2 数据标注

我们推荐使用[Label Studio](https://labelstud.io/) 数据标注工具进行标注，如果已有标注好的本地数据集，我们需要将数据集整理为文档要求的格式，详见[Label Studio数据标注指南](./label_studio_text.md)。

这里我们提供预先标注好的`医疗意图分类数据集`的文件，可以运行下面的命令行下载数据集，我们将展示如何使用数据转化脚本生成训练/验证/测试集文件，并使用UTC模型进行微调。

下载医疗意图分类数据集：


```shell
wget https://bj.bcebos.com/paddlenlp/datasets/utc-medical.tar.gz
tar -xvf utc-medical.tar.gz
mv utc-medical data
rm utc-medical.tar.gz
```

生成训练/验证集文件：
```shell
python label_studio.py \
    --label_studio_file ./data/label_studio.json \
    --save_dir ./data \
    --splits 0.8 0.1 0.1 \
    --options ./data/label.txt
```
多任务训练场景可分别进行数据转换再进行混合。

<a name="模型微调"></a>

### 2.3 模型微调

推荐使用 PromptTrainer API 对模型进行微调，该 API 封装了提示定义功能，且继承自 [Trainer API ](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md) 。只需输入模型、数据集等就可以使用 Trainer API 高效快速地进行预训练、微调等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，Trainer API 还针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

使用下面的命令，使用 `utc-large` 作为预训练模型进行模型微调，将微调后的模型保存至`$finetuned_model`：

单卡启动：

```shell
python run_train.py  \
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
python -u -m paddle.distributed.launch --gpus "0,1" run_train.py \
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
python run_eval.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/test.txt \
    --per_device_eval_batch_size 2 \
    --max_seq_len 512 \
    --output_dir ./checkpoint_test
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
>>> schema = ["病情诊断", "治疗方案", "病因分析", "指标解读", "就医建议", "疾病表述", "后果表述", "注意事项", "功效作用", "医疗费用", "其他"]
>>> my_cls = Taskflow("zero_shot_text_classification", schema=schema, task_path='./checkpoint/model_best', precision="fp16")
>>> pprint(my_cls("中性粒细胞比率偏低"))
```

<a name="模型部署"></a>

### 2.6 模型部署

在UTC的服务化能力中我们提供基于PaddleNLP SimpleServing 来搭建服务化能力，通过几行代码即可搭建服务化部署能力

```
# Save at server.py
from paddlenlp import SimpleServer, Taskflow

schema = ["病情诊断", "治疗方案", "病因分析", "指标解读", "就医建议"]
utc = Taskflow("zero_shot_text_classification",
               schema=schema,
               task_path="../../checkpoint/model_best/",
               precision="fp32")
app = SimpleServer()
app.register_taskflow("taskflow/utc", utc)
```

```
# Start the server
paddlenlp server server:app --host 0.0.0.0 --port 8990
```

支持FP16半精度推理加速，详见[UTC SimpleServing 使用方法](./deploy/simple_serving/README.md)

<a name="实验指标"></a>

### 2.7 实验指标

医疗意图分类数据集 KUAKE-QIC 验证集实验指标：

  |          |  Accuracy  | Micro F1   | Macro F1   |
  | :------: | :--------: | :--------: | :--------: |
  | 0-shot   | 28.69 | 87.03 | 60.90 |
  | 5-shot   | 64.75 | 93.34 | 80.33 |
  | 10-shot  | 65.88 | 93.76 | 81.34 |
  | full-set | 81.81 | 96.65 | 89.87 |

其中 k-shot 表示每个标签有 k 条标注样本用于训练。
