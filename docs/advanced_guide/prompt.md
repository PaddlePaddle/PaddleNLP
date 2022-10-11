# 提示学习：Prompt API

随着预训练语言模型规模的增长，“预训练-微调”范式在下游自然语言处理任务上的表现越来越好，但与之相应地对训练数据量和计算存储资源的要求也越来越高。为了充分利用预训练语言模型学习到的知识，同时降低对数据和资源的依赖，**提示学习**（Prompt Learning）作为一种可能的新范式受到了越来越多的关注，在 FewCLUE、SuperGLUE 等榜单的小样本任务上取得了远优于传统微调范式的结果。

**提示学习**的核心思想是将下游任务转化为预训练阶段的掩码预测（MLM）任务。实现思路包括通过模板（Template）定义的提示语句，将原有任务转化为预测掩码位置的词，以及通过标签词（Verbalizer）的定义，建立预测词与真实标签之间的映射关系。

以情感分类任务为例，“预训练-微调”范式和“预训练-提示”范式（以 [PET](https://arxiv.org/abs/2001.07676) 为例）之间的区别如下图所示

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/192727706-0a17b5ef-db6b-46be-894d-0ee315306776.png width=800 height=300 />
</div>

【微调学习】使用 `[CLS]` 来做分类，需要训练随机初始化的分类器，需要充分的训练数据来拟合。

【提示学习】通过提示语句和标签词映射的定义，转化为 MLM 任务，无需训练新的参数，适用于小样本场景。


Prompt API 提供了这类算法实现的基本模块，支持[PET](https://arxiv.org/abs/2001.07676)、[P-Tuning](https://arxiv.org/abs/2103.10385)、[WARP](https://aclanthology.org/2021.acl-long.381/)、[RGL](https://aclanthology.org/2022.findings-naacl.81/)等经典算法的快速实现。

**目录**

* [如何定义模板](#如何定义模板)
    * [离散型模板](#离散型模板)
    * [连续型模板](#连续型模板)
    * [快速定义模板](#快速定义模板)
* [如何定义标签词映射](#如何定义标签词映射)
    * [单掩码映射](#单掩码映射)
    * [多掩码映射](#多掩码映射)
    * [标签词映射分类](#标签词映射分类)
* [快速开始训练](#快速开始训练)
    * [数据准备](#数据准备)
    * [预训练参数准备](#预训练参数准备)
    * [定义提示学习模型](#定义提示学习模型)
    * [使用PromptTrainer训练](#使用PromptTrainer训练)
* [实践教程](#实践教程)
    * [文本分类示例](#文本分类示例)
    * 其他任务示例（待更新）
* [Reference](#Reference)

## 如何定义模板

**模板**（Template）的功能是在原有输入文本上增加提示语句，从而将原任务转化为 MLM 任务，可以分为离散型和连续型两种。Prompt API 中提供了统一的数据结构来构造不同类型的模板，输入相应格式的**字符串**，通过解析得到对应的输入模板，即字典构成的列表。

### 离散型模板

离散型模板 `ManualTemplate` 是直接将提示语句与原始输入文本拼接起来，二者的词向量矩阵共享，均为预训练模型学到的词向量矩阵。可用于实现 PET、RGL 等算法。

**模板关键字**

- ``text`` ：数据集中原始输入文本对应的关键字，包括`text_a`和`text_b`。[数据准备](#数据准备)中介绍了如何将自定义数据集转化为统一格式。
- ``hard`` ：自定义的文本提示语句。
- ``mask`` ：待预测词的占位符。
- ``sep`` ：用于区分不同的句子。`sep`前后的句子对应不同的`token_type_id`。

**模板定义**

```
{'hard': '“'}{'text': 'text_a'}{'hard': '”和“'}{'text': 'text_b'}{'hard': '”之间的逻辑关系是'}{'mask'}
```

或者使用简化方式定义，省略关键字 ``hard`` 后与上述模板等价。

```
“{'text': 'text_a'}”和“{'text': 'text_b'}”之间的逻辑关系是{'mask'}
```

**样本示例**

例如，对于自然语言推理任务，给定样本

```python
from paddlenlp.prompt import InputExample
sample = InputExample(uid=0,
                      text_a="心里有些生畏,又不知畏惧什么",
                      text_b="心里特别开心",
                      labels="矛盾")
```

按照模板修改拼接后，最终输入模型的文本数据为

```
“心里有些生畏,又不知畏惧什么”和“心里特别开心”之间的逻辑关系是[MASK]
```


**调用 API**

```python
from paddlenlp.prompt import ManualTemplate
from paddlenlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
template = ManualTemplate(tokenizer=tokenizer,
                          max_seq_length=512,
                          template="“{'text': 'text_a'}”和“{'text': 'text_b'}”之间的逻辑关系是{'mask'}")
input_dict = template.wrap_one_example(sample)
```

其中初始化参数定义如下

- ``tokenizer`` ：预训练模型的 tokenizer，用于文本编码。
- ``max_seq_length`` ：定义输入模型文本的最大长度，包括提示部分。当输入长度超过最大长度时，只会截断`text`关键字对应的输入文本，提示部分不做处理。
- ``template`` ：定义提示语句以及与输入文本组合方式的字符串。

**使用技巧**

不同模板定义对结果的影响很明显。一般来说，提示语句与原始输入文本拼接后，语句越通顺自然，模型效果越好。在实践中，对于不同的任务需要分析文本特点，尝试不同的模板以取得好的效果。


### 连续型模板

离散型模板的使用难点在于设计一个好的提示语句需要很多经验和语言专业知识。为了解决这一问题，连续型模板 `SoftTemplate` 尝试使用一组连续性 prompt 向量作为模板，这样模型训练时就无需人工给定提示语句。当然，`SoftTemplate` 也支持用人工构造的提示来初始化 prompt 向量。与离散型模板的区别在于连续型提示向量与输入文本的词向量矩阵不共享，二者在训练过程中分别进行参数更新。可用于实现 P-Tuning 等算法。

除此之外，连续型模板还支持混合模板定义，即在原始输入上同时拼接离散型提示和连续型提示向量。

**模板关键字**

- ``text`` ：数据集中原始输入文本对应的关键字，包括`text_a`和`text_b`。[数据准备](#数据准备)中介绍了如何将自定义数据集转化为统一格式。
- ``hard`` ：自定义的文本提示语句。
- ``mask`` ：待预测词的占位符。
- ``sep`` ：用于区分不同的句子。`sep`前后的句子对应不同的`token_type_id`。
- ``soft`` 表示连续型提示。若值为 ``None`` ，则使用对应数量的随机初始化向量作为提示；若值为文本，则使用对应长度的连续性向量作为提示，并预训练词向量中文本对应的向量进行初始化。

**模板定义**

- 定义长度为 1 的连续型提示，随机初始化：

```python
"{'soft': None}{'text': 'text_a'}{'sep'}{'text': 'text_b'}"
```

- 定义长度为 10 的连续型提示，随机初始化，其中 ``duplicate`` 参数表示连续型提示的长度（仅在随机初始化时有效，即`soft`值为`None`）：

```python
"{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': None, `duplicate`:10}{'mask'}"
```

- 定义长度为 15 的连续型提示，使用 `"请判断这两个句子间的逻辑关系："` 的预训练词向量逐一进行初始化：

```python
"{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': '请判断这两个句子间的逻辑关系：'}{'mask'}"
```

- 定义混合模板，这里`soft`关键字对应的提示和`hard`对应的提示对应两套不同的向量：

```python
"{'soft': '自然语言推理任务：'}{'text': 'text_a'}{'sep'}{'text': 'text_b'}这两个句子间的逻辑关系是{'mask'}"
```


**调用 API**

```python
from paddlenlp.prompt import SoftTemplate
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
template = SoftTemplate(tokenizer=tokenizer,
                        max_seq_length=512,
                        model=model,
                        template="{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': '请判断这两个句子间的逻辑关系：'}{'mask'}",
                        prompt_encoder='lstm',
                        encoder_hidden_size=200)
```

其中初始化参数定义如下

- ``tokenizer`` ：预训练模型的 tokenizer，用于文本编码。
- ``max_seq_length`` ：定义输入模型文本的最大长度，包括提示部分。当输入长度超过最大长度时，只会截断`text`关键字对应的输入文本，提示部分不做处理。
- ``model`` : 预训练语言模型，为了取预训练词向量用于连续型提示向量初始化。
- ``template`` ：定义连续型模板的提示语句、初始化以及与输入文本组合方式的字符串。
- ``prompt_encoder`` : 连续型提示向量的编码器，可选 ``mlp`` 和 ``lstm``。默认为 ``None`` ，即无编码器，直接使用向量。
- ``encoder_hidden_size`` : 连续型提示向量的维度。默认为 ``None`` ，即与预训练词向量维度相同。

**使用技巧**

- 对于分类任务，推荐的连续型提示长度一般为10-20。
- 对于随机初始化的连续性 prompt 向量，通常用比预训练模型微调更大的学习率来更新参数。
- 与离散型模板相似，连续型模板对初始化参数也比较敏感。自定义提示语句作为连续性 prompt 向量的初始化参数通常比随机初始化效果好。
- prompt_encoder 为已有论文中的策略，用于建模不同连续型提示向量之间的序列关系。在实际应用中推荐先去掉 prompt_encoder 调整向量初始化。


### 快速定义模板

PaddleNLP 提供了 ``AutoTemplate`` API 以便快速定义单句输入的手工初始化的连续型模板，同时支持直接按照模板类型自动切换离散型模板和离散型模板。

**模板定义**

- 只定义用于初始化连续型向量的文本提示，即可得到拼接到句尾的连续型模板输入。例如，

```python
"这篇文章表达了怎样的情感？"
```

等价于

```python
"{'text': 'text_a'}{'soft': '这篇文章表达了怎样的情感？'}{'mask'}"
```

- 当输入为完整模板字符串时，解析得到的模板与[离散型模板](#离散型模板)和[连续型模板](#连续型模板)中描述的一致。

**调用 API**

```python
from paddlenlp.prompt import AutoTemplate
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
# 离散型模板，返回值为 ManualTemplate 实例
template = AutoTemplate.create_from(template="{'text': 'text_a'}和{'text': 'text_b'}之间的逻辑关系是{'mask'}",
                                    tokenizer=tokenizer,
                                    max_seq_length=512)

# 连续型模板，返回值为 SoftTemplate 实例
template = AutoTemplate.create_from(template="{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': '请判断这两个句子间的逻辑关系：'}{'mask'}",
                                    tokenizer=tokenizer,
                                    max_seq_length=512,
                                    model=model,
                                    prompt_encoder='lstm',
                                    encoder_hidden_size=200)

# 快速定义单句连续型模板，返回值为 SoftTemplate 实例
template = AutoTemplate.create_from(template="这篇文章表达了怎样的情感？",
                                    tokenizer=tokenizer,
                                    max_seq_length=512,
                                    model=model,
                                    prompt_encoder='lstm',
                                    encoder_hidden_size=200)
```

其中初始化参数定义如下

- ``tokenizer`` ：预训练模型的 tokenizer，用于文本编码。
- ``max_seq_length`` ：定义输入模型文本的最大长度，包括提示部分。当输入长度超过最大长度时，只会截断`text`关键字对应的输入文本，提示部分不做处理。
- ``model`` ：预训练语言模型，为了取预训练词向量用于连续型提示向量初始化。
- ``template`` ：定义离散型/连续型提示、初始化以及和输入文本的组合方式。
- ``prompt_encoder`` ：连续型提示向量的编码器，可选 ``mlp`` 和 ``lstm`` 。默认为 ``None`` ，即无编码器，直接使用向量。
- ``encoder_hidden_size`` ：连续型提示向量的维度。默认为 ``None`` ，即与预训练词向量维度相同。


## 如何定义标签词映射

**标签词映射**（Verbalizer）也是提示学习中可选的重要模块，用于建立预测词和标签之间的映射，将“预训练-微调”模式中预测标签的任务转换为预测模板中掩码位置的词语，从而将下游任务统一为预训练任务的形式。目前框架支持了离散型标签词映射和 [Word-level Adversarial ReProgramming (WARP)](https://aclanthology.org/2021.acl-long.381/) 方法。


例如，在情感二分类任务中，微调方法和提示学习的标签体系如下

- **微调方式** : 数据集的标签为 ``负向`` 和 ``正向``，分别映射为 ``0`` 和 ``1`` ；

- **提示学习** : 通过下边的标签词映射建立原始标签与预测词之间的映射。

``` python
{'负向': '不', '正向': '很'}
```

具体来说，对于模板 ``{'text':'text_a'}这句话表示我{'mask'}满意。`` ，我们使用映射 ``{'负向': '不', '正向': '很'}`` 将标签 ``负向`` 映射为 ``不`` ，将标签 ``正向`` 映射为 ``很`` 。也就是说，我们期望对于正向情感的文本，预测结果为 ``...这句话表示我很满意。`` ，对于负向情感的文本，预测结果为 ``...这句话表示我不满意。``


### 单掩码映射

``ManualVerbalizer`` 支持构造简单的单 ``{'mask'}`` 标签词映射，直接作用于 ``AutoMaskedLM`` 模型结构。当标签对应的预测词长度大于 ``1`` 时取均值。

**调用 API**

```python
from paddlenlp.prompt import ManualVerbalizer
from paddlenlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
verbalizer = ManualVerbalizer(tokenizer=tokenizer,
                              labels=['负向', '正向'],
                              label_words={'负向': '不', '正向': '很'},
                              prefix=None)
```

其中初始化参数定义如下

- ``tokenizer`` : 预训练模型的 tokenizer，用于预测词的编码。
- ``labels`` : 数据集的原标签列表（可选）。
- ``label_words`` : 原标签到预测词之间的映射字典。如果同时定义了 ``labels`` ，二者的标签集合需要相同。
- ``prefix`` : 预测词解码前增加的前缀，用于 ``RoBERTa`` 等对前缀敏感的模型，例如 `roberta-large`， `good` 和 ` good` 经过 tokenize 会得到不同的 id。默认为 ``None`` ，无前缀。


### 多掩码映射

``MultiMaskVerbalizer`` 继承自 ``ManualVerbalizer`` ，支持多 ``{'mask'}`` 标签词映射。预测词长度需与 ``{'mask'}`` 长度一致。

**调用 API**

```python
from paddlenlp.prompt import MultiMaskVerbalizer
from paddlenlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
verbalizer = MultiMaskVerbalizer(tokenizer=tokenizer,
                                 labels=['负向', '正向'],
                                 label_words={'负向': '生气', '正向': '高兴'},
                                 prefix=None)
```


其中初始化参数定义同[单掩码映射](#单掩码映射) 。


### 标签词映射分类

标签词映射分类器 ``SoftVerbalizer`` 修改了原 ``AutoMaskedLM`` 的模型结构，将预训练模型最后一层“隐藏层-词表”替换为“隐藏层-标签”的映射。该层网络的初始化参数由标签词映射中的预测词词向量来决定，如果预测词长度大于 ``1`` ，则使用词向量均值进行初始化。当前支持的预训练模型包括 ``ErnieForMaskedLM`` 、 ``BertForMaskedLM`` 、 ``AlbertForMaskedLM`` 和 ``RobertaForMaskedLM`` 。可用于实现 WARP 算法。


**调用 API**

```python
from paddlenlp.prompt import SoftVerbalizer
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
verbalizer = SoftVerbalizer(tokenizer=tokenizer,
                            model=model,
                            labels=['负向', '正向'],
                            label_words={'负向': '生气', '正向': '高兴'},
                            prefix=None)
```

其中初始化参数定义同[单掩码映射](#单掩码映射) ，此外

- ``model`` ：预训练语言模型，用于取预训练词向量进行“隐藏层-标签”网络的修改和初始化。

## 快速开始训练

本节介绍了如何使用 ``PromptTrainer`` 快速搭建提示训练流程。

### 数据准备

Prompt 框架定义了统一的样本结构 ``InputExample`` 以便进行数据处理，数据集样本需要封装在 ``MapDataset`` 中。

例如，对于文本语义相似度 BUSTM 数据集中的原始样本

```python
data = [
    {'id': 3, 'sentence1': '你晚上吃了什么', 'sentence2': '你晚上吃啥了', 'label': 1},
    {'id': 4, 'sentence1': '我想打开滴滴叫的士', 'sentence2': '你叫小欧吗', 'label': 0},
    {'id': 5, 'sentence1': '女孩子到底是不是你', 'sentence2': '你不是女孩子吗', 'label': 1}
]
```


需要转换为统一格式

```python
from paddlenlp.datasets import MapDataset
from paddlenlp.prompt import InputExample

data_ds = MapDataset([InputExample(uid=example["id"],
                                   text_a=example["sentence1"],
                                   text_b=example["sentence2"],
                                   labels=example["label"]) for example in data])
```

### 预训练参数准备

如果使用标签词映射，用 ``AutoModelForMaskedLM`` 和 ``AutoTokenizer`` 加载预训练模型参数。如果不使用标签词映射，可将 ``AutoModelForMaskedLM`` 替换为任务对应的模型。

```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
```


### 定义提示学习模型

对于文本分类任务，我们将模板预处理和标签词映射封装为提示学习模型 ``PromptModelForSequenceClassification`` 。


```python
from paddlenlp.prompt import AutoTemplate
from paddlenlp.prompt import ManualVerbalizer
from paddlenlp.prompt import PromptModelForSequenceClassification

# 定义模板
template = AutoTemplate.create_from(template="{'text': 'text_a'}和{'text': 'text_b'}说的是{'mask'}同的事情。",
                                    tokenizer=tokenizer,
                                    max_seq_length=512)

# 定义标签词映射
verbalizer = ManualVerbalizer(tokenizer=tokenizer,
                              label_words={0: '不', 1: '相'})

# 定义文本分类提示模型
prompt_model = PromptModelForSequenceClassification(model,
                                                    template,
                                                    verbalizer,
                                                    freeze_plm=False,
                                                    freeze_dropout=False)
```

其中提示模型初始化参数如下

- ``model`` : 预训练模型实例，支持 ``AutoModelForMaskedLM`` 和 ``AutoModelForSequenceClassification`` 。
- ``template`` : 模板实例。
- ``verbalizer`` : 标签词映射实例。当设为 ``None`` 时，不使用标签词映射，模型输出及损失值计算由 ``model`` 类型定义。
- ``freeze_plm`` : 在训练时是否固定预训练模型参数。对于规模较小的预训练模型，推荐更新预训练模型参数。
- ``freeze_dropout`` : 在训练时是否固定预训练模型参数并关闭 ``dropout`` 。 当 ``freeze_dropout=True`` ，``freeze_plm`` 也为 ``True`` 。


### 使用PromptTrainer训练

``PromptTrainer`` 继承自 ``Trainer`` ， 封装了数据处理，模型训练、测试，训练策略等，便于训练流程的快速搭建。

**配置训练参数**

``PromptTuningArguments`` 继承自 ``TrainingArguments`` ，包含了提示学习的主要训练参数。其中 ``TrainingArguments`` 参数见 `Trainer API 文档 <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md>`_ ，其余参数详见 [Prompt Trainer参数列表](#PromptTrainer参数列表) 。推荐使用 **命令行** 的形式进行参数配置，即

```shell
python xxx.py --output_dir xxx --learning_rate xxx
```

除了训练参数，还需要自定义数据和模型相关的参数。最后用 ``PdArgumentParser`` 输出参数。

```python
from dataclasses import dataclass, field
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.prompt import PromptTuningArguments

@dataclass
class DataArguments:
    data_path : str = field(default="./data", metadata={"help": "The path to dataset."})

parser = PdArgumentParser((DataArguments, PromptTuningArguments))
data_args, training_args = parser.parse_args_into_dataclasses(
    args=["--output_dir", "./", "--do_train", "True"], look_for_args_file=False)
```

**初始化和训练**

除了上述准备，还需要定义损失函数和评估函数。

```python

import paddle
from paddle.metric import Accuracy
from paddlenlp.prompt import PromptTrainer

# 损失函数
criterion = paddle.nn.CrossEntropyLoss()

# 评估函数
def compute_metrics(eval_preds):
    metric = Accuracy()
    correct = metric.compute(paddle.to_tensor(eval_preds.predictions),
                             paddle.to_tensor(eval_preds.label_ids))
    metric.update(correct)
    acc = metric.accumulate()
    return {"accuracy": acc}

# 初始化
trainer = PromptTrainer(model=prompt_model,
                        tokenizer=tokenizer,
                        args=training_args,
                        criterion=criterion,
                        train_dataset=data_ds,
                        eval_dataset=None,
                        callbacks=None,
                        compute_metrics=compute_metrics)

# 训练模型
if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
```

## 实践教程

### 文本分类示例


- [多分类文本分类示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class/few-shot)

- [多标签文本分类示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label/few-shot)

- [多层次文本分类示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical/few-shot)


## Reference

- Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference. [[PDF]](https://arxiv.org/abs/2001.07676)
- GPT Understands, Too. [[PDF]](https://arxiv.org/abs/2103.10385)
- WARP: Word-level Adversarial ReProgramming. [[PDF]](https://aclanthology.org/2021.acl-long.381/)
- RGL: A Simple yet Effective Relation Graph Augmented Prompt-based Tuning Approach for Few-Shot Learning. [[PDF]](https://aclanthology.org/2022.findings-naacl.81/)
- R-Drop: Regularized Dropout for Neural Networks. [[PDF]](https://arxiv.org/abs/2106.14448)

### 附录


#### PromptTrainer参数列表


| 参数              |  类型  | 默认值   |   含义                                                   |
| ---------------- | ------ | ------- | ------------------------------------------------------- |
| max_seq_length   |  int   |  512    |    模型输入的最大长度，包括模板部分                          |
| freeze_plm       |  bool  |  False  |    是否在训练时固定预训练模型的参数                          |
| freeze_dropout   |  bool  |  False  |    是否在训练时固定预训练模型的参数，同时关闭 dropout         |
| use_rdrop        |  bool  |  False  |   是否使用 RDrop 策略，详见 [RDrop 论文](https://arxiv.org/abs/2106.14448) |
| alpha_rdrop      |  float |  5.0    |   RDrop Loss 的权重                                      |
| use_rgl          |  bool  |  False  |   是否使用 RGL 策略，详见 [RGL 论文](https://aclanthology.org/2022.findings-naacl.81/) |
| alpha_rgl        |  float |  0.5    |   RGL Loss 的权重                                        |
| ppt_learning_rate|  float |  1e-4   |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的学习率   |
| ppt_weight_decay |  float |  0.0    |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的衰减参数 |
| ppt_adam_beta1   |  float |  0.9    |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的 beta1  |
| ppt_adam_beta2   |  float |  0.999  |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的 beta2  |
| ppt_adam_epsilon |  float |  1e-8   |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的 epsilon|
