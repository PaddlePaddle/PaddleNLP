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
    * [前缀连续型模板](#前缀连续型模板)
    * [快速定义模板](#快速定义模板)
* [如何定义标签词映射](#如何定义标签词映射)
    * [离散型标签词映射](#离散型标签词映射)
    * [连续型标签词映射](#连续型标签词映射)
* [快速开始训练](#快速开始训练)
    * [数据准备](#数据准备)
    * [预训练参数准备](#预训练参数准备)
    * [定义提示学习模型](#定义提示学习模型)
    * [使用 PromptTrainer 训练](#使用 PromptTrainer 训练)
* [实践教程](#实践教程)
    * [文本分类示例](#文本分类示例)
    * 其他任务示例（待更新）
* [Reference](#Reference)

## 如何定义模板

**模板**（Template）的功能是在原有输入文本上增加提示语句，从而将原任务转化为 MLM 任务，可以分为离散型和连续型两种。Prompt API 中提供了统一的数据结构来构造不同类型的模板，输入相应格式的**字符串**，通过解析得到对应的输入模板。模板由不同字段构成，可任意组合。每个字段中的关键字定义了数据文本或者提示文本，即 `input_ids`，属性可定义该字段是否可截断，以及对应的 `position_ids`，`token_type_ids` 等。

### 离散型模板

离散型模板 `ManualTemplate` 是直接将提示语句与原始输入文本拼接起来，二者的词向量矩阵共享，均为预训练模型学到的词向量矩阵。可用于实现 PET、RGL 等算法。

**模板关键字及属性**

- ``text`` ：数据集中原始输入文本对应的关键字，例如，`text_a`、`text_b` 和 `content`。
- ``hard`` ：自定义的提示语句文本。
- ``mask`` ：待预测词的占位符。
    - ``length`` ：定义 ``mask`` 的数量。
- ``sep`` ：句间的标志符。不同句子的 `token_type_ids` 需使用 `token_type` 属性定义，默认相同。
- ``options`` ：数据集字典或者文件中的候选标签序列。
    - ``add_omask`` ：在每个标签前新增 `[O-MASK]` 字符，用于计算候选标签的预测值。支持实现 [UniMC](https://arxiv.org/pdf/2210.08590.pdf) 算法。
    - ``add_prompt`` ：给每个标签拼接固定的提示文本，标签位置由 `[OPT]` 标记。支持实现 [EFL](https://arxiv.org/pdf/2104.14690.pdf) 算法。

**模版通用属性**

- `position`: 定义当前字段的起始 `position id`。
- `token_type`: 定义当前字段及后续字段的 `token type id`。
- `truncate`: 定义当提示和文本总长度超过最大长度时，当前字段是否可截断。可选 `True` 和 `False`。

**模板定义**

```
{'hard': '“'}{'text': 'text_a'}{'hard': '”和“'}{'text': 'text_b'}{'hard': '”之间的逻辑关系是'}{'mask'}
```

或者使用简化方式定义，省略关键字 ``hard`` 后与上述模板等价。

```
“{'text': 'text_a'}”和“{'text': 'text_b'}”之间的逻辑关系是{'mask'}
```

```
{'options': './data/label.txt'}{'sep'}下边两句话间的逻辑关系是什么？{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}
```
其中 `label.txt` 为候选标签的本地文件路径，每行一个候选标签，例如

```
中立
蕴含
矛盾
```

**样本示例**

例如，对于自然语言推理任务，给定样本

```python
sample = {
    "text_a": "心里有些生畏,又不知畏惧什么", "text_b": "心里特别开心", "labels": "矛盾"
}
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
template = ManualTemplate(prompt="“{'text': 'text_a'}”和“{'text': 'text_b'}”之间的逻辑关系是{'mask'}",
                          tokenizer=tokenizer,
                          max_length=512)
input_dict = template(sample)
```

其中初始化参数定义如下

- ``prompt`` ：定义提示语句以及与输入文本组合方式的字符串。
- ``tokenizer`` ：预训练模型的 tokenizer，用于文本编码。
- ``max_length`` ：定义输入模型文本的最大长度，包括提示部分。

**使用技巧**

不同模板定义对结果的影响很明显。一般来说，提示语句与原始输入文本拼接后，语句越通顺自然，模型效果越好。在实践中，对于不同的任务需要分析文本特点，尝试不同的模板以取得好的效果。


### 连续型模板

离散型模板的使用难点在于设计一个好的提示语句需要很多经验和语言专业知识。为了解决这一问题，连续型模板 `SoftTemplate` 尝试使用一组连续性 prompt 向量作为模板，这样模型训练时就无需人工给定提示语句。当然，`SoftTemplate` 也支持用人工构造的提示来初始化 prompt 向量。与离散型模板的区别在于连续型提示向量与输入文本的词向量矩阵不共享，二者在训练过程中分别进行参数更新。可用于实现 P-Tuning 等算法。

除此之外，连续型模板还支持混合模板定义，即在原始输入上同时拼接离散型提示和连续型提示向量。

**模板关键字**

- ``text`` ：数据集中原始输入文本对应的关键字，例如，`text_a`和`text_b`。
- ``hard`` ：自定义的文本提示语句。
- ``mask`` ：待预测词的占位符。
- ``sep`` ：句间的标志符。不同句子的 `token_type_ids` 需使用 `token_type` 属性定义，默认相同。
- ``soft`` 表示连续型提示。若值为 ``None`` ，则随机初始化提示向量；若值为文本，则使用文本对应的预训练字向量初始化提示向量。
    - ``length`` ：定义 ``soft token`` 的数量。若定义文本长度小于该值，超过部分随机初始化。
    - ``encoder`` ：定义 `soft token` 的编码器类型，可选 `lstm`，`mlp`。默认为 `None`， 不使用编码器。
    - ``hidden_size`` ：定义编码器的隐藏层维度。默认与预训练词向量维度相同。
- ``options`` ：数据集字典或者文件中的候选标签序列。
    - ``add_omask`` ：在每个标签前新增 `[O-MASK]` 字符，用于计算候选标签的预测值。支持实现 [UniMC](https://arxiv.org/pdf/2210.08590.pdf) 算法。
    - ``add_prompt`` ：给每个标签拼接固定的提示文本，标签位置由 `[OPT]` 标记。支持实现 [EFL](https://arxiv.org/pdf/2104.14690.pdf) 算法。

**模版通用属性**

- `position`: 定义当前字段的起始 `position id`。
- `token_type`: 定义当前字段及后续字段的 `token type id`。
- `truncate`: 定义当提示和文本总长度超过最大长度时，当前字段是否可截断。可选 `True` 和 `False`。

**模板定义**

- 定义长度为 1 的连续型提示，随机初始化：

```python
"{'soft'}{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}"
```

- 定义长度为 10 的连续型提示，随机初始化，编码器为 `mlp`：

```python
"{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': None, 'length':10, 'encoder': 'mlp'}{'mask'}"
```

- 定义长度为 15 的连续型提示，使用 `请判断` 初始化前三个 soft token，其余随机初始化，编码器为隐藏层维度为 100 的双层 LSTM：

```python
"{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': '请判断：', 'length': 15, 'encoder': 'lstm', 'hidden_size': 100}{'mask'}"
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
template = SoftTemplate(prompt="{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': '请判断这两个句子间的逻辑关系：'}{'mask'}",
                        tokenizer=tokenizer,
                        max_length=512,
                        word_embeddings=model.get_input_embeddings())
```

其中初始化参数定义如下

- ``prompt`` ：定义连续型模板的提示语句、初始化以及与输入文本组合方式的字符串。
- ``tokenizer`` ：预训练模型的 tokenizer，用于文本编码。
- ``max_seq_length`` ：定义输入模型文本的最大长度，包括提示部分。
- ``word_embeddings`` ：预训练语言模型的词向量，用于连续型提示向量初始化。
- ``soft_embeddings`` ：连续型提示向量矩阵，可用于不同模板间的连续型参数共享。设置后将覆盖默认连续型向量矩阵。

**使用技巧**

- 对于分类任务，推荐的连续型提示长度一般为10-20。
- 对于随机初始化的连续性 prompt 向量，通常用比预训练模型微调更大的学习率来更新参数。
- 与离散型模板相似，连续型模板对初始化参数也比较敏感。自定义提示语句作为连续性 prompt 向量的初始化参数通常比随机初始化效果好。
- prompt_encoder 为已有论文中的策略，用于建模不同连续型提示向量之间的序列关系。


### 前缀连续型模板

`PrefixTemplate` 同样使用了连续型向量作为提示，与 `SoftTemplate` 的不同，该模版的提示向量不仅仅作用于输入层，每层都会有相应的提示向量。可用于实现 P-Tuning 等算法。

**模板关键字**

- ``text`` ：数据集中原始输入文本对应的关键字，例如，`text_a`和`text_b`。
- ``hard`` ：自定义的文本提示语句。
- ``mask`` ：待预测词的占位符。
- ``sep`` ：句间的标志符。不同句子的 `token_type_ids` 需使用 `token_type` 属性定义，默认相同。
- ``prefix`` 表示连续型提示，该字段**必须**位于模板首位。若值为 ``None`` ，则随机初始化提示向量；若值为文本，则使用文本对应的预训练字向量初始化提示向量。
    - ``length`` ：定义 ``soft token`` 的数量。若定义文本长度小于该值，超过部分随机初始化。
    - ``encoder`` ：定义 `soft token` 的编码器类型，可选 `lstm`，`mlp`。默认为 `None`， 不使用编码器。
    - ``hidden_size`` ：定义编码器的隐藏层维度。默认与预训练词向量维度相同。
- ``options`` ：数据集字典或者文件中的候选标签序列。
    - ``add_omask`` ：在每个标签前新增 `[O-MASK]` 字符，用于计算候选标签的预测值。支持实现 [UniMC](https://arxiv.org/pdf/2210.08590.pdf) 算法。
    - ``add_prompt`` ：给每个标签拼接固定的提示文本，标签位置由 `[OPT]` 标记。支持实现 [EFL](https://arxiv.org/pdf/2104.14690.pdf) 算法。

**模版通用属性**

- `position`: 定义当前字段的起始 `position id`。
- `token_type`: 定义当前字段及后续字段的 `token type id`。
- `truncate`: 定义当提示和文本总长度超过最大长度时，当前字段是否可截断。可选 `True` 和 `False`。

**模板定义**

- 定义长度为 15 的连续型提示，随机初始化：

```python
"{'prefix': '新闻类别', 'length': 10, 'encoder': 'lstm'}{'text': 'text_a'}"
```

- 定义混合模板，这里`prefix`关键字对应的提示和`hard`对应的提示对应两套不同的向量：

```python
"{'prefix': '自然语言推理任务：', 'encoder': 'mlp'}{'text': 'text_a'}{'sep'}{'text': 'text_b'}这两个句子间的逻辑关系是{'mask'}"
```


**调用 API**

```python
from paddlenlp.prompt import PrefixTemplate
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
template = PrefixTemplate(prompt="{'prefix': '任务描述'}{'text': 'text_a'}{'mask'}",
                          tokenizer=tokenizer,
                          max_length=512,
                          model=model,
                          prefix_dropout=0.1)
```

其中初始化参数定义如下

- ``prompt`` ：定义连续型模板的提示语句、初始化以及与输入文本组合方式的字符串。
- ``tokenizer`` ：预训练模型的 tokenizer，用于文本编码。
- ``max_length`` ：定义输入模型文本的最大长度，包括提示部分。
- ``model`` ：预训练语言模型，用于连续型提示向量初始化，以及根据模型结构生成每层对应的提示向量。
- ``prefix_dropout`` ：连续型提示向量的丢弃概率，用于正则化。


### 快速定义模板

PaddleNLP 提供了 ``AutoTemplate`` API 快速定义简化离散型模板，也可根据完整模板字符串自动切换 ManualTemplate、SoftTemplate 和 PrefixTemplate。

**模板定义**

- 快速定义离散型的文本提示。例如，

```python
"这篇文章表达了怎样的情感？"
```

等价于

```python
"{'text': 'text_a'}{'hard': '这篇文章表达了怎样的情感？'}{'mask'}"
```

- 当输入为完整模板字符串时，解析得到的模板与[离散型模板](#离散型模板)和[连续型模板](#连续型模板)中描述的一致。

**调用 API**

```python
from paddlenlp.prompt import AutoTemplate
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
# 离散型模板，返回值为 ManualTemplate 实例
template = AutoTemplate.create_from(prompt="这个句子表达了怎样的情感？",
                                    tokenizer=tokenizer,
                                    max_length=512)

template = AutoTemplate.create_from(prompt="这个句子表达了怎样的情感？{'text': 'text_a'}{'mask'}",
                                    tokenizer=tokenizer,
                                    max_length=512)

# 连续型模板，返回值为 SoftTemplate 实例
template = AutoTemplate.create_from(prompt="{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': '请判断这两个句子间的逻辑关系：'}{'mask'}",
                                    tokenizer=tokenizer,
                                    max_length=512,
                                    model=model)

# 前缀连续型模板，返回值为 PrefixTemplate 实例
template = AutoTemplate.create_from(prompt="{'prefix': None, 'encoder': 'mlp', 'hidden_size': 50}{'text': 'text_a'}",
                                    tokenizer=tokenizer,
                                    max_length=512,
                                    model=model)
```

其中初始化参数定义如下

- ``prompt`` ：定义离散型/连续型提示、初始化以及和输入文本的组合方式。
- ``tokenizer`` ：预训练模型的 tokenizer，用于文本编码。
- ``max_length`` ：定义输入模型文本的最大长度，包括提示部分。
- ``model`` ：预训练语言模型，为了取预训练词向量用于连续型提示向量初始化。

## 如何定义标签词映射

**标签词映射**（Verbalizer）也是提示学习中可选的重要模块，用于建立预测词和标签之间的映射，将“预训练-微调”模式中预测标签的任务转换为预测模板中掩码位置的词语，从而将下游任务统一为预训练任务的形式。目前框架支持了离散型标签词映射和连续型标签词映射 [Word-level Adversarial ReProgramming (WARP)](https://aclanthology.org/2021.acl-long.381/) 方法。


例如，在情感二分类任务中，微调方法和提示学习的标签体系如下

- **微调方式** : 数据集的标签为 ``负向`` 和 ``正向``，分别映射为 ``0`` 和 ``1`` ；

- **提示学习** : 通过下边的标签词映射建立原始标签与预测词之间的映射。

``` python
{'负向': '不', '正向': '很'}
```

具体来说，对于模板 ``{'text':'text_a'}这句话表示我{'mask'}满意。`` ，我们使用映射 ``{'负向': '不', '正向': '很'}`` 将标签 ``负向`` 映射为 ``不`` ，将标签 ``正向`` 映射为 ``很`` 。也就是说，我们期望对于正向情感的文本，预测结果为 ``...这句话表示我很满意。`` ，对于负向情感的文本，预测结果为 ``...这句话表示我不满意。``


### 离散型标签词映射

``ManualVerbalizer`` 支持构造 ``{'mask'}`` 对应的标签词映射，同一标签可对应多个不同长度的词，直接作用于 ``AutoMaskedLM`` 模型结构。当标签对应的预测词长度大于 ``1`` 时，默认取均值；当标签对应多个 `{'mask'}` 时，默认与单个 `{mask}` 效果等价。

**调用 API**

```python
from paddlenlp.prompt import ManualVerbalizer
from paddlenlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
verbalizer = ManualVerbalizer(tokenizer=tokenizer,
                              label_words={'负向': '不', '正向': '很'})
```

其中初始化参数定义如下

- ``label_words`` : 原标签到预测词之间的映射字典。
- ``tokenizer`` : 预训练模型的 tokenizer，用于预测词的编码。

``MaskedLMVerbalizer`` 同样支持构造 ``{'mask'}`` 对应的标签词映射，映射词与模板中的 `{'mask'}` 逐字对应，因此，映射词长度应与 `{'mask'}` 数量保持一致。当定义的标签词映射中同一标签对应多个词时，仅有第一个映射词生效。在自定义的 `compute_metric` 函数中需先调用 `verbalizer.aggregate_multiple_mask` 将多 `{'mask'}` 合并后再计算评估函数，默认使用乘积的方式。

**调用 API**
```python
from paddlenlp.prompt import MaskedLMVerbalizer
from paddlenlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
verbalizer = MaskedLMVerbalizer(tokenizer=tokenizer,
                                label_words={'负向': '不', '正向': '很'})
```

其中初始化参数定义如下

- ``label_words`` : 原标签到预测词之间的映射字典。
- ``tokenizer`` : 预训练模型的 tokenizer，用于预测词的编码。

### 连续型标签词映射

标签词映射分类器 ``SoftVerbalizer`` 修改了原 ``AutoMaskedLM`` 的模型结构，将预训练模型最后一层“隐藏层-词表”替换为“隐藏层-标签”的映射。该层网络的初始化参数由标签词映射中的预测词词向量来决定，如果预测词长度大于 ``1`` ，则使用词向量均值进行初始化。当前支持的预训练模型包括 ``ErnieForMaskedLM`` 、 ``BertForMaskedLM`` 、 ``AlbertForMaskedLM`` 和 ``RobertaForMaskedLM`` 。可用于实现 WARP 算法。


**调用 API**

```python
from paddlenlp.prompt import SoftVerbalizer
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
verbalizer = SoftVerbalizer(label_words={'负向': '生气', '正向': '高兴'},
                            tokenizer=tokenizer,
                            model=model)
```

- ``label_words`` : 原标签到预测词之间的映射字典。
- ``tokenizer`` : 预训练模型的 tokenizer，用于预测词的编码。
- ``model`` ：预训练语言模型，用于取预训练词向量进行“隐藏层-标签”网络的修改和初始化。

## 快速开始训练

本节介绍了如何使用 ``PromptTrainer`` 快速搭建提示训练流程。

### 数据准备

数据集封装为 ``MapDataset`` 类型。每条数据格式为字典结构，字典中关键字与模板中 `text` 定义的值相对应，统一使用 `labels` 关键字表示样本标签。

例如，文本语义相似度 BUSTM 数据集中的数据样本

```python
from paddlenlp.datasets import MapDataset

data_ds = MapDataset([
    {'id': 3, 'sentence1': '你晚上吃了什么', 'sentence2': '你晚上吃啥了', 'label': 1},
    {'id': 4, 'sentence1': '我想打开滴滴叫的士', 'sentence2': '你叫小欧吗', 'label': 0},
    {'id': 5, 'sentence1': '女孩子到底是不是你', 'sentence2': '你不是女孩子吗', 'label': 1}
])

def convert_label_keyword(input_dict):
    input_dict["labels"] = input_dict.pop("label")
    return input_dict

data_ds = data_ds.map(convert_label_keyword)
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
template = AutoTemplate.create_from(prompt="{'text': 'text_a'}和{'text': 'text_b'}说的是{'mask'}同的事情。",
                                    tokenizer=tokenizer,
                                    max_length=512)

# 定义标签词映射
verbalizer = ManualVerbalizer(label_words={0: '不', 1: '相'},
                              tokenizer=tokenizer)

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
- ``freeze_plm`` : 在训练时固定预训练模型参数，默认为 `False`。对于轻量级预训练模型，推荐使用默认值。
- ``freeze_dropout`` : 在训练时固定预训练模型参数并关闭 ``dropout`` 。 当 ``freeze_dropout=True`` ，``freeze_plm`` 也为 ``True`` 。


### 使用 PromptTrainer 训练

``PromptTrainer`` 继承自 ``Trainer`` ， 封装了数据处理，模型训练、测试，训练策略等，便于训练流程的快速搭建。

**配置训练参数**

``PromptTuningArguments`` 继承自 ``TrainingArguments`` ，包含了提示学习的主要训练参数。其中 ``TrainingArguments`` 参数见 `Trainer API 文档 <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md>`_ ，其余参数详见 [Prompt Trainer 参数列表](#PromptTrainer 参数列表) 。推荐使用 **命令行** 的形式进行参数配置，即

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


- [多分类文本分类示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class/few-shot)

- [多标签文本分类示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_label/few-shot)

- [多层次文本分类示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/hierarchical/few-shot)


## Reference

- Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference. [[PDF]](https://arxiv.org/abs/2001.07676)
- GPT Understands, Too. [[PDF]](https://arxiv.org/abs/2103.10385)
- WARP: Word-level Adversarial ReProgramming. [[PDF]](https://aclanthology.org/2021.acl-long.381/)
- RGL: A Simple yet Effective Relation Graph Augmented Prompt-based Tuning Approach for Few-Shot Learning. [[PDF]](https://aclanthology.org/2022.findings-naacl.81/)
- R-Drop: Regularized Dropout for Neural Networks. [[PDF]](https://arxiv.org/abs/2106.14448)
- Openprompt: An open-source framework for prompt-learning. [[PDF]](https://arxiv.org/abs/2111.01998)


### 附录


#### PromptTrainer 参数列表


| 参数              |  类型  | 默认值   |   含义                                                   |
| ---------------- | ------ | ------- | ------------------------------------------------------- |
| max_seq_length   |  int   |  512    |   模型输入的最大长度，包括模板部分                          |
| freeze_plm       |  bool  |  False  |   是否在训练时固定预训练模型的参数                          |
| freeze_dropout   |  bool  |  False  |   是否在训练时固定预训练模型的参数，同时关闭 dropout         |
| use_rdrop        |  bool  |  False  |   是否使用 RDrop 策略，详见 [RDrop 论文](https://arxiv.org/abs/2106.14448) |
| alpha_rdrop      |  float |  5.0    |   RDrop Loss 的权重                                      |
| use_rgl          |  bool  |  False  |   是否使用 RGL 策略，详见 [RGL 论文](https://aclanthology.org/2022.findings-naacl.81/) |
| alpha_rgl        |  float |  0.5    |   RGL Loss 的权重                                        |
| ppt_learning_rate|  float |  1e-4   |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的学习率   |
| ppt_weight_decay |  float |  0.0    |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的衰减参数 |
| ppt_adam_beta1   |  float |  0.9    |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的 beta1  |
| ppt_adam_beta2   |  float |  0.999  |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的 beta2  |
| ppt_adam_epsilon |  float |  1e-8   |   连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的 epsilon|
