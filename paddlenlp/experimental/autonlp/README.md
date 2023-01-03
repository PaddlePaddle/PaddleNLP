# AutoNLP

**AutoNLP目前在实验阶段。在正式发布之前，AutoNLP API有可能会变动**

**AutoNLP** 是 PaddleNLP 的一个早期的实验性质的项目，旨在让NLP技术赋能百业。交付一个成功的 NLP 项目并不容易，因为它需要深入的NLP领域知识，而我们经常看到开发者应用NLP技术的过程中遇到困难。这就是我们开发 **AutoNLP** 项目的原因。与为获得最先进的模型精度而使用大规模计算资源的传统 AutoML 方法相比，我们有不同的理念：

1. 我们的目标不是在大型集群，大型数据集上训练最先进的模型，而是**在有限计算资源下的训练出不错模型**。我们假设我们的用户最多只有几个 GPU，并且希望在8小时内训练出不错的模型。您可以在 [Baidu AiStudio](https://aistudio.baidu.com/aistudio) 免费获得此级别的计算资源。
2. AutoNLP的目标是提供**低代码**的解决方案，使您能够用几行代码训练出不错的模型，但它不是无代码的模型训练服务。
3. 我们将尽可能地**自动化和抽象化** PaddleNLP已有的**全流程能力**（例如 预处理，分词，微调，提示学习，模型压缩，一键部署等等），助力开发者对于自己的使用场景进行快速适配与落地。
4. 我们的工作是**免费和开源**的。

## 安装

安装 **AutoNLP** 与安装 PaddleNLP 非常相似，唯一的区别是 需要添加`[autonlp]`的标签。

```
pip install -U paddlenlp[autonlp]
```

您还可以从我们的 [GitHub](https://github.com/PaddlePaddle/PaddleNLP) clone并通过“pip install .[autonlp]”从源代码安装来获取develop分支中的最新成果。

## 基础使用

由于目前AutoNLP唯一支持的任务是文本分类，因此以下文档是关于 **AutoTrainerForTextClassification** 的使用用法。您也可以参考我们的 AiStudio notebook (To be added)

### 创建AutoTrainerForTextClassification对象

`AutoTrainerForTextClassification` 是您用来运行模型实验并与经过训练的模型交互的主要类，您可以像下面这样构造它：

```python
auto_trainer = AutoTrainerForTextClassification(
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    label_column="labels",
    text_column="sentence",
    language="Chinese",
    output_dir="temp"
)
```

Args:

- train_dataset (Dataset, required): `paddle.io.Dataset` 格式的训练数据集，必须包含下面指定的 `text_column` 和 `label_column`
- eval_dataset (Dataset, required): `paddle.io.Dataset`格式的评估数据集，必须包含下面指定的`text_column`和`label_column`
- text_column (string, required): 数据集中的文本字段，为模型的主要输入。
- label_column (string, required): 数据集中的标签字段
- language (string, required): 文本语言
- metric_for_best_model (string, optional): 用来选择最优模型的评估指标
- greater_is_better (bool, optional): 更好的模型是否应该有更大的指标。与`metric_for_best_model`结合使用
- problem_type (str, optional): 根据问题的性质在 [`multi_class`, `multi_label`] 中选择
- output_dir (str, optional): 输出目录，默认为`autpnlp_results`

### 训练

您可以使用以下命令开始训练模型：

```python
auto_trainer.train(
    num_cpus=2,
    num_gpus=1,
    max_concurrent_trials=1,
    num_models=10,
    time_budget_s=60 * 10,
    verbosity=1
)
```
Args:

- num_models (int, required): 模型试验数量
- num_gpus (str, optional): 实验使用的 GPU 数量。默认情况下，这是根据检测到的 GPU 设置的。
- num_cpus (str, optional): 实验使用的 CPU 数量。默认情况下，这是根据检测到的 vCPU 设置的。
- max_concurrent_trials (int, optional): 同时运行的最大试验数。必须是非负数。如果为 None 或 0，则不应用任何限制。默认为None。
- time_budget_s: (int|float|datetime.timedelta, optional) 以秒为单位的全局时间预算，超过时间后停止所有模型试验。
- experiment_name: (str, optional): 实验的名称。实验日志将存储在"<output_dir>/<experiment_name>"下。默认为 UNIX 时间戳。
- verbosity: (int, optional): 控制日志的详细程度。默认为“0”，将日志级别设置为 INFO。如果需要减少日志量，请使用 `verbosity > 0` 将日志级别设置为 WARNINGS
- hp_overrides: (dict[str, Any], optional): （仅限高级用户）。覆盖每个候选模型的超参数。例如，`{"TrainingArguments.max_steps"：5}`。
- custom_model_candiates: (dict[str, Any], optional): （仅限高级用户）。运行用户提供的候选模型而不 PaddleNLP 的默认候选模型。可以参考 `._model_candidates` 属性


### 评估和检查实验结果

#### 检查实验结果

实验结束后，您可以像下面这样检查实验结果，它会打印一个 pandas DataFrame：

```
auto_trainer.show_training_results()
```

您还可以在`<output_dir>/experiment_results.csv`下找到实验结果。不同实验产生的模型的标识符是`trial_id`，您可以在 DataFrame 或 csv 文件中找到这个字段。

#### 加载以前的实验结果

您可以从之前的运行（包括未完成的运行）中恢复实验结果，如下所示：

```python
auto_trainer.load("path/to/previous/results")
```

这使您能够使用 `show_training_results` API 来检查结果。再次调用 train() 将覆盖之前的结果。

#### 使用不同的评估数据集

除了使用构建 AutoTrainerForTextClassification 的时候提供的评估数据集以外，您也可以使用其他的数据集进行评估：

```
auto_trainer.evaluate(
    trial_id="trial_123456",
    eval_dataset=new_eval_dataset
)
```

Args:
- trial_id (str, optional): 通过 `trial_id` 指定要评估的模型。默认为由`metric_for_best_model`决定的最佳模型
- eval_dataset (Dataset, optional): 自定义评估数据集，并且必须包含`text_column`和`label_column`字段。如果未提供，则默认为构建时使用的评估数据集



### 模型输出与部署

如果需要导出模型供以后使用，可以使用以下的API：

```
auto_trainer.export(
    trial_id="trial_123456",
    export_path="different/path/to/store/the/model"
)
```

Args:
- export_path (str, required): 输出路径
- trial_id (int, required): 通过 `trial_id` 指定要评估的模型。默认为由`metric_for_best_model`决定的最佳模型

同时我们还提供了`to_taskflow()`的API，可以直接将模型转换为 `Taskflow` 进行推理：

```
taskflow = auto_trainer.to_taskflow()
taskflow("this is a test input")
```

Args:
- trial_id (int, required): 通过 `trial_id` 指定要评估的模型。默认为由`metric_for_best_model`决定的最佳模型
