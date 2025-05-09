# Prompt Learning: Prompt API

With the growth in scale of pre-trained language models, the "pre-train and fine-tune" paradigm has achieved increasingly better performance on downstream NLP tasks. However, this comes with correspondingly higher demands on training data volume and computational resources. To fully utilize the knowledge learned by pre-trained language models while reducing dependency on data and resources, **Prompt Learning** has emerged as a potential new paradigm gaining significant attention, achieving far superior results compared to traditional fine-tuning approaches on few-shot tasks in benchmarks like FewCLUE and SuperGLUE.

The core idea of **Prompt Learning** is to reformulate downstream tasks as masked language modeling (MLM) objectives from the pre-training phase. The implementation approach involves:
1. Using template-defined prompt statements to convert original tasks into predicting masked token positions
2. Establishing mappings between predicted words and true labels through verbalizer definitions

Taking sentiment classification as an example, the differences between the "pre-train and fine-tune" paradigm and the "pre-train and prompt" paradigm (using [PET](https://arxiv.org/abs/2001.07676) as example) are illustrated below:

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/192727706-0a17b5ef-db6b-46be-894d-0ee315306776.png width=800 height=300 />
</div>

[Fine-tuning Approach] uses `[CLS]`
To perform classification using `ChatGPT`, training a randomly initialized classifier requires sufficient training data to fit.

**[Prompt Learning]** By defining prompt statements and label word mappings, the task is transformed into an MLM task without training new parameters, making it suitable for few-shot scenarios.

The Prompt API provides fundamental modules for implementing such algorithms, supporting rapid implementation of classical algorithms like [PET](https://arxiv.org/abs/2001.07676), [P-Tuning](https://arxiv.org/abs/2103.10385), [WARP](https://aclanthology.org/2021.acl-long.381/), and [RGL](https://aclanthology.org/2022.findings-naacl.81/).

**Table of Contents**

* [How to Define Templates](#如何定义模板)
    * [Discrete Templates](#离散型模板)
    * [Continuous Templates](#连续型模板)
    * [Prefix Continuous Templates](#前缀连续型模板)
    * [Quick Template Definition](#快速定义模板)
* [How to Define Label Word Mappings](#如何定义标签词映射)
    * [Discrete Label Word Mappings](#离散型标签词映射)
    * [Continuous Label Word Mappings](#连续型标签词映射)
* [Quick Start Training](#快速开始训练)
    * [Data Preparation](#数据准备)
    * [Pre-trained Parameter Preparation](#预训练参数准备)
    * [Defining Prompt Learning Models](#定义提示学习模型)
    * [Training with PromptTrainer](#使用 PromptTrainer 训练)
* [Practical Tutorials](#实践教程)
    * [Text Classification Example](#文本分类示例)
    * Other Task Examples (to be updated)
* [Reference](#Reference)

## How to Define Templates

**Templates** add prompt statements to original input text, converting the original task into an MLM task. They can be categorized into discrete and continuous types. The Prompt API provides unified data structures to construct different template types by parsing formatted strings.

### Discrete Templates

Discrete templates `ManualTemplate` directly concatenate prompt statements with original input text. Both share the same word embedding matrix learned from pre-trained models. Suitable for implementing algorithms like PET and RGL.

**Template Keywords and Attributes**

- ``text``: Keywords corresponding to original input text in the dataset, e.g., `text_a`, `text_b`, and `content`.
- ``hard``: Custom prompt text.
- ``mask``: Placeholder for predicted words.
    - ``length``: Defines the number of ``mask`` tokens.
- ``sep``: Sentence separator. Use `token_type` attribute to define `token_type_ids` for different sentences (default: same).
- ``options``: Candidate label sequences from dataset dictionaries or files.
    - ``add_omask``: Adds `[O-MASK]` before each label for calculating prediction scores of candidate labels. Supports [UniMC](https://arxiv.org/pdf/2210.08590.pdf) algorithm.
    - ``add_prompt``
You are a professional NLP technical translator. Translate Chinese to English while:
1. Preserving EXACT formatting (markdown/rst/code)
2. Keeping technical terms in English
3. Maintaining code/math blocks unchanged
4. Using proper academic grammar
5. Keep code block in documents original
6. Keep the link in markdown/rst the same. E.g. [link](#这里) instead of [link](#here)
7. Keep the html tag in markdown/rst the same.
6. Just return the result of Translate. No additional messages.

**Add fixed prompt text for each tag, with tag positions marked by `[OPT]`. Supports implementation of [EFL](https://arxiv.org/pdf/2104.14690.pdf) algorithm.**

**Common Properties of Templates**

- `position`: Defines the starting `position id` for the current field.
- `token_type`: Defines `token type id` for current and subsequent fields.
- `truncate`: Determines whether the current field can be truncated when total prompt and text length exceeds max length. Options: `True` and `False`.

**Template Definition**

```
{'hard': '“'}{'text': 'text_a'}{'hard': '”和“'}{'text': 'text_b'}{'hard': '”之间的逻辑关系是'}{'mask'}
```

Or use simplified definition (equivalent to above by omitting ``hard`` keyword):

```
“{'text': 'text_a'}”和“{'text': 'text_b'}”之间的逻辑关系是{'mask'}
```

```
{'options': './data/label.txt'}{'sep'}下边两句话间的逻辑关系是什么？{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}
```
Where `label.txt` is the local file path containing candidate labels (one per line), e.g.

```
neutral
entailment
contradiction
```

**Sample Example**

For natural language inference task, given sample:

```python
sample = {
    "text_a": "心里有些生畏,又不知畏惧什么", "text_b": "心里特别开心", "labels": "contradiction"
}
```

After template modification and concatenation, the final input text to model is:

```
“心里有些生畏,又不知畏惧什么”和“心里特别开心”之间的逻辑关系是[MASK]
```

**API Call**

```python
from paddlenlp.prompt import ManualTemplate
from paddlenlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
template = ManualTemplate(prompt="“{'text': 'text_a'}”和“{'text': 'text_b'}”之间的逻辑关系是{'mask'}",
                          tokenizer=tokenizer,
                          max_length=512)
input_dict = template(sample)
```

Where initialization parameters are defined as:

- ``prompt``: String defining prompt statement and its combination with input text.
- ``tokenizer``: Pretrained model's tokenizer for text encoding.
- ``max_length``
Define the maximum length of the input model text, including the prompt section.

**Usage Tips**

The impact of different template definitions on results is significant. Generally, the more natural and fluent the concatenated prompt statements are with the original input text, the better the model performance. In practice, different templates should be analyzed and tried for different tasks to achieve optimal results.

### Continuous Template

The challenge in using discrete templates lies in the requirement of substantial experience and linguistic expertise to design good prompt statements. To address this, the continuous template `SoftTemplate` attempts to use a set of continuous prompt vectors as templates, eliminating the need for manual prompt design during model training. Of course, `SoftTemplate` also supports initializing prompt vectors with human-crafted prompts. The key difference from discrete templates is that the continuous prompt vectors and the input text's word embedding matrix are not shared; they are updated separately during training. This can be used to implement algorithms like P-Tuning.

Additionally, continuous templates support hybrid template definitions, i.e., concatenating both discrete prompts and continuous prompt vectors with the original input.

**Template Keywords**

- ``text``: Keywords corresponding to original input texts in the dataset, e.g., `text_a` and `text_b`.
- ``hard``: User-defined text prompts.
- ``mask``: Placeholder for tokens to be predicted.
- ``sep``: Sentence separator token. Different sentences' `token_type_ids` should be defined using the `token_type` attribute; they are identical by default.
- ``soft``: Represents continuous prompts. If the value is ``None``, prompt vectors are randomly initialized; if a text value is provided, the corresponding pretrained word embeddings are used for initialization.
    - ``length``: Defines the number of ``soft tokens``. If the defined text length is less than this value, the excess portion is randomly initialized.
    - ``encoder``: Defines the encoder type for `soft tokens`, options include `lstm` and `mlp`. Default is `None` (no encoder used).
    - ``hidden_size``: Defines the hidden layer dimension of the encoder. Defaults to the same dimension as the pretrained word embeddings.
- ``options``: Candidate label sequences from the dataset dictionary or files.
    - ``add_omask``: Adds `[O-MASK]` before each label for calculating prediction scores of candidate labels. Supports implementation of the [UniMC](https://arxiv.org/pdf/2210.08590.pdf) algorithm.
    - ``add_prompt``: Appends fixed prompt text to each label, with label positions marked by `[OPT]`. Supports implementation of the [EFL](https://arxiv.org/pdf/2104.14690.pdf) algorithm.

**General Template Attributes**

- `position`: Defines the starting `position id` for the current field.
- `token_type`: Defines the `token type id` for the current and subsequent fields.
- `truncate`: Determines whether the current field can be truncated when the total length of prompts and text exceeds the maximum length. Options: `True` or `False`.

**Template Definitions**

- Define a continuous prompt of length 1 with random initialization:

```python
"{'soft'}{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}"
```

- Define a continuous prompt of length 10 with random initialization and `mlp` encoder:
```python
"{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': None, 'length':10, 'encoder': 'mlp'}{'mask'}"
```

- Define a continuous prompt of length 15, initialize the first three soft tokens with `Please determine`, and randomly initialize the rest, with the encoder being a two-layer LSTM with hidden dimension 100:

```python
"{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': 'Please determine:', 'length': 15, 'encoder': 'lstm', 'hidden_size': 100}{'mask'}"
```

- Define a continuous prompt of length 15, initialized with the pretrained word vectors of `"Please determine the logical relationship between these two sentences:"` one by one:

```python
"{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': 'Please determine the logical relationship between these two sentences:'}{'mask'}"
```

- Define a hybrid template, where the `soft` keyword corresponds to one set of vectors and the `hard` keyword corresponds to another set of distinct vectors:

```python
"{'soft': 'Natural Language Inference Task:'}{'text': 'text_a'}{'sep'}{'text': 'text_b'} The logical relationship between these two sentences is {'mask'}"
```

**API Invocation**

```python
from paddlenlp.prompt import SoftTemplate
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
template = SoftTemplate(prompt="{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': 'Please determine the logical relationship between these two sentences:'}{'mask'}",
                        tokenizer=tokenizer,
                        max_length=512,
                        word_embeddings=model.get_input_embeddings())
```

The initialization parameters are defined as follows:

- ``prompt``: String defining the prompt statements for continuous templates, initialization methods, and composition with input text.
- ``tokenizer``: Pretrained model tokenizer for text encoding.
- ``max_seq_length``: Defines the maximum sequence length of model input text, including the prompt section.
- ``word_embeddings``: Word embeddings from the pretrained language model, used for continuous prompt vector initialization.
- ``soft_embeddings``
Continuous prompt vector matrix, which can be used for sharing continuous parameters between different templates. When set, it will override the default continuous vector matrix.

**Usage Tips**

- For classification tasks, the recommended length for continuous prompts is typically 10-20.
- For randomly initialized continuous prompt vectors, parameters are usually updated with a larger learning rate compared to fine-tuning pre-trained models.
- Similar to discrete templates, continuous templates are also sensitive to initialization. Using custom prompt statements as initialization parameters for continuous prompt vectors often yields better results than random initialization.
- The `prompt_encoder` refers to strategies from existing papers, used to model sequential relationships between different continuous prompt vectors.

### Prefix Continuous Template

`PrefixTemplate` also uses continuous vectors as prompts. Unlike `SoftTemplate`, the prompt vectors in this template not only operate at the input layer but also have corresponding prompt vectors at each layer. This can be used to implement algorithms like P-Tuning.

**Template Keywords**

- ``text``: Keywords corresponding to original input text in the dataset, e.g., `text_a` and `text_b`.
- ``hard``: Custom textual prompt statements.
- ``mask``: Placeholder for the word to be predicted.
- ``sep``: Sentence separator. Different sentences' `token_type_ids` should be defined using the `token_type` attribute, which is the same by default.
- ``prefix`` represents continuous prompts. This field **must** be placed first in the template. If the value is ``None``, prompt vectors are randomly initialized; if the value is text, the corresponding pre-trained word vectors are used to initialize the prompt vectors.
    - ``length``: Defines the number of ``soft tokens``. If the defined text length is less than this value, the excess part will be randomly initialized.
    - ``encoder``: Defines the encoder type for `soft tokens`, options include `lstm` and `mlp`. Default is `None` (no encoder used).
    - ``hidden_size``: Defines the hidden layer dimension of the encoder. Default is the same as the pre-trained word vector dimension.
- ``options``: Candidate label sequences from the dataset dictionary or files.
    - ``add_omask``: Adds `[O-MASK]` before each label for computing prediction scores of candidate labels. Supports implementation of the [UniMC](https://arxiv.org/pdf/2210.08590.pdf) algorithm.
    - ``add_prompt``: Appends fixed prompt text to each label, with label positions marked by `[OPT]`. Supports implementation of the [EFL](https://arxiv.org/pdf/2104.14690.pdf) algorithm.

**General Template Attributes**

- `position`: Defines the starting `position id` for the current field.
- `token_type`: Defines the `token type id` for the current field and subsequent fields.
- `truncate`: Determines whether the current field can be truncated when the total length of prompts and text exceeds the maximum length. Options are `True` or `False`.

**Template Definition**

- Define a continuous prompt of length 15 with random initialization:

```python
"{'prefix': 'news category', 'length': 10, 'encoder': 'lstm'}{'text': 'text_a'}"
```

- Define a hybrid template where the `prefix` keyword corresponds to one set of prompt vectors and `hard` corresponds to another distinct set:

```python
"{'prefix': None, 'length': 15, 'encoder': 'mlp'}{'hard': 'This is a text: '}{'text': 'text_a'}"
```
```python
"{'prefix': 'Natural Language Inference Task:', 'encoder': 'mlp'}{'text': 'text_a'}{'sep'}{'text': 'text_b'} The logical relationship between these two sentences is {'mask'}"
```

**API Call**

```python
from paddlenlp.prompt import PrefixTemplate
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
template = PrefixTemplate(prompt="{'prefix': 'Task Description'}{'text': 'text_a'}{'mask'}",
                          tokenizer=tokenizer,
                          max_length=512,
                          model=model,
                          prefix_dropout=0.1)
```

The initialization parameters are defined as follows:

- ``prompt``: Defines the prompt statement, initialization, and text composition method for the continuous template.
- ``tokenizer``: Pretrained model's tokenizer for text encoding.
- ``max_length``: Defines the maximum length of input text to the model, including the prompt section.
- ``model``: Pretrained language model for continuous prompt vector initialization and generating corresponding prompt vectors per layer based on model architecture.
- ``prefix_dropout``: Dropout probability for continuous prompt vectors, used for regularization.

### Quick Template Definition

PaddleNLP provides the ``AutoTemplate`` API for rapid definition of simplified discrete templates, and can automatically switch between ManualTemplate, SoftTemplate, and PrefixTemplate based on complete template strings.

**Template Definition**

- Quickly define discrete text prompts. For example:

```python
"What emotion does this article express?"
```

is equivalent to:

```python
"{'text': 'text_a'}{'hard': 'What emotion does this article express?'}{'mask'}"
```

- When input is a complete template string, the parsed template will be consistent with the descriptions in [Discrete Templates](#discrete-templates) and [Continuous Templates](#continuous-templates).
```python
from paddlenlp.prompt import AutoTemplate
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
# Discrete template, returns ManualTemplate instance
template = AutoTemplate.create_from(prompt="What emotion does this sentence express?",
                                    tokenizer=tokenizer,
                                    max_length=512)

template = AutoTemplate.create_from(prompt="What emotion does this sentence express? {'text': 'text_a'}{'mask'}",
                                    tokenizer=tokenizer,
                                    max_length=512)

# Continuous template, returns SoftTemplate instance
template = AutoTemplate.create_from(prompt="{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': 'Please determine the logical relationship between these two sentences:'}{'mask'}",
                                    tokenizer=tokenizer,
                                    max_length=512,
                                    model=model)

# Prefix continuous template, returns PrefixTemplate instance
template = AutoTemplate.create_from(prompt="{'prefix': None, 'encoder': 'mlp', 'hidden_size': 50}{'text': 'text_a'}",
                                    tokenizer=tokenizer,
                                    max_length=512,
                                    model=model)
```

The initialization parameters are defined as follows:

- ``prompt``: Defines discrete/continuous prompts, initialization, and combination with input text.
- ``tokenizer``: Tokenizer of the pre-trained model, used for text encoding.
- ``max_length``: Defines the maximum length of input text to the model, including the prompt section.
- ``model``
## How to Define Label Word Mapping

**Label Word Mapping** (Verbalizer) is an optional yet crucial module in prompt learning, establishing the mapping between predicted words and labels. It transforms the label prediction task from the traditional "pre-train fine-tune" paradigm into predicting masked words in templates, thereby unifying downstream tasks as pre-training objectives. The framework currently supports discrete label word mapping and continuous label word mapping via the [Word-level Adversarial ReProgramming (WARP)](https://aclanthology.org/2021.acl-long.381/) method.

For example, in a sentiment binary classification task:

- **Fine-tuning Approach**: Dataset labels are ``negative`` and ``positive``, mapped to ``0`` and ``1`` respectively.

- **Prompt Learning**: Uses the following label word mapping to bridge original labels and predicted words.

```python
{'negative': '不', 'positive': '很'}
```

Specifically, for the template ``{'text':'text_a'}这句话表示我{'mask'}满意。``, we use the mapping ``{'negative': '不', 'positive': '很'}`` to map label ``negative`` to ``不`` and label ``positive`` to ``很``. This means: for positive sentiment texts, the prediction should be ``...这句话表示我很满意。``; for negative sentiment texts, the prediction should be ``...这句话表示我不满意。``.

### Discrete Label Word Mapping

``ManualVerbalizer`` supports constructing label word mappings for ``{'mask'}`` positions. A single label can correspond to multiple words of varying lengths, directly applied to the ``AutoMaskedLM`` architecture. When a label's corresponding word exceeds length 1, the average is taken by default; when multiple `{'mask'}` tokens exist, it equivalently handles them as single `{mask}`.

**API Usage**

```python
from paddlenlp.prompt import ManualVerbalizer
from paddlenlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
verbalizer = ManualVerbalizer(tokenizer=tokenizer,
                              label_words={'negative': '不', 'positive': '很'})
```

Initialization parameters:

- ``label_words``: Dictionary mapping original labels to predicted words.
- ``tokenizer``: Pretrained model's tokenizer for word encoding.

``MaskedLMVerbalizer`` also supports label word mapping for ``{'mask'}`` positions. The mapped words must align with the number of `{'mask'}` tokens in the template. When multiple words are defined for the same label, only the first takes effect. The custom ``compute_metric`` function should call ``verbalizer.aggregate_multiple_mask`` to merge multiple `{'mask'}` positions before evaluation (product aggregation by default).

**API Usage**
```python
from paddlenlp.prompt import MaskedLMVerbalizer
from paddlenlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
verbalizer = MaskedLMVerbalizer(tokenizer=tokenizer,
                                label_words={'negative': 'not', 'positive': 'very'})
```

The initialization parameters are defined as follows:

- ``label_words``: A dictionary mapping original labels to prediction words.
- ``tokenizer``: The tokenizer from the pre-trained model, used for encoding prediction words.

### Continuous Label Word Mapping

The label word mapping classifier ``SoftVerbalizer`` modifies the original ``AutoMaskedLM`` model architecture by replacing the last "hidden layer-vocabulary" mapping with a "hidden layer-label" mapping. The initialization parameters of this layer are determined by the word vectors from the label word mapping. If the prediction word length exceeds ``1``, the average of word vectors is used for initialization. Currently supported pre-trained models include ``ErnieForMaskedLM``, ``BertForMaskedLM``, ``AlbertForMaskedLM``, and ``RobertaForMaskedLM``. This can be used to implement the WARP algorithm.

**API Call**

```python
from paddlenlp.prompt import SoftVerbalizer
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
verbalizer = SoftVerbalizer(label_words={'negative': 'angry', 'positive': 'happy'},
                            tokenizer=tokenizer,
                            model=model)
```

- ``label_words``: A dictionary mapping original labels to prediction words.
- ``tokenizer``: The tokenizer from the pre-trained model, used for encoding prediction words.
- ``model``: The pre-trained language model used to retrieve pre-trained word vectors for modifying and initializing the "hidden layer-label" network.

## Quick Start Training

This section explains how to use ``PromptTrainer`` to quickly set up a prompt training workflow.

### Data Preparation

The dataset is encapsulated as ``MapDataset`` type. Each data entry is formatted as a dictionary, where the dictionary keys correspond to the values defined in the template's `text`, and the `labels` keyword is uniformly used to represent sample labels.

For example, a data sample from the text semantic similarity BUSTM dataset:
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

### Pre-trained Parameter Preparation

If using label word mapping, load pre-trained parameters with ``AutoModelForMaskedLM`` and ``AutoTokenizer``. If not using label word mapping, you can replace ``AutoModelForMaskedLM`` with the corresponding model for the task.

```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
```

### Defining the Prompt Learning Model

For text classification tasks, we encapsulate template preprocessing and label word mapping into the prompt learning model ``PromptModelForSequenceClassification``.
```python
from paddlenlp.prompt import AutoTemplate
from paddlenlp.prompt import ManualVerbalizer
from paddlenlp.prompt import PromptModelForSequenceClassification

# Define template
template = AutoTemplate.create_from(prompt="{'text': 'text_a'} and {'text': 'text_b'} are talking about {'mask'} same thing.",
                                    tokenizer=tokenizer,
                                    max_length=512)

# Define label word mapping
verbalizer = ManualVerbalizer(label_words={0: 'not', 1: 'related'},
                              tokenizer=tokenizer)

# Define prompt-based sequence classification model
prompt_model = PromptModelForSequenceClassification(model,
                                                    template,
                                                    verbalizer,
                                                    freeze_plm=False,
                                                    freeze_dropout=False)

```

The initialization parameters for the prompt model are as follows:

- ``model`` : Pre-trained model instance, supporting ``AutoModelForMaskedLM`` and ``AutoModelForSequenceClassification``.
- ``template`` : Template instance.
- ``verbalizer`` : Label word mapping instance. When set to ``None``, label word mapping is not used, model output and loss calculation are defined by the ``model`` type.
- ``freeze_plm`` : Freeze pre-trained model parameters during training, default is `False`. For lightweight pre-trained models, the default value is recommended.
- ``freeze_dropout`` : Freeze pre-trained model parameters and disable ``dropout`` during training. When ``freeze_dropout=True``, ``freeze_plm`` also becomes ``True``.

### Training with PromptTrainer

``PromptTrainer`` inherits from ``Trainer``, encapsulating data processing, model training/evaluation, and training strategies for rapid setup of training workflow.

**Configuring Training Parameters**

``PromptTuningArguments`` inherits from ``TrainingArguments``, containing main training parameters for prompt learning. For ``TrainingArguments`` parameters, see
`Trainer API Documentation <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md>`_. For other parameters, please refer to [Prompt Trainer Parameter List](#PromptTrainer-Parameter-List). It is recommended to use **command line** for parameter configuration, i.e.

```shell
python xxx.py --output_dir xxx --learning_rate xxx
```

In addition to training parameters, custom parameters related to data and models need to be defined. Finally, use ``PdArgumentParser`` to output the parameters.

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

**Initialization and Training**

In addition to the above preparations, loss function and evaluation metrics need to be defined.
```python

import paddle
from paddle.metric import Accuracy
from paddlenlp.prompt import PromptTrainer

# Loss function
criterion = paddle.nn.CrossEntropyLoss()

# Evaluation function
def compute_metrics(eval_preds):
    metric = Accuracy()
    correct = metric.compute(paddle.to_tensor(eval_preds.predictions),
                             paddle.to_tensor(eval_preds.label_ids))
    metric.update(correct)
    acc = metric.accumulate()
    return {"accuracy": acc}

# Initialization
trainer = PromptTrainer(model=prompt_model,
                        tokenizer=tokenizer,
                        args=training_args,
                        criterion=criterion,
                        train_dataset=data_ds,
                        eval_dataset=None,
                        callbacks=None,
                        compute_metrics=compute_metrics)

# Training the model
if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
```
## Tutorials

### Text Classification Examples

- [Multi-class Text Classification Example](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class/few-shot)

- [Multi-label Text Classification Example](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_label/few-shot)

- [Hierarchical Text Classification Example](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/hierarchical/few-shot)

## Reference

- Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference. [[PDF]](https://arxiv.org/abs/2001.07676)
- GPT Understands, Too. [[PDF]](https://arxiv.org/abs/2103.10385)
- WARP: Word-level Adversarial ReProgramming. [[PDF]](https://aclanthology.org/2021.acl-long.381/)
- RGL: A Simple yet Effective Relation Graph Augmented Prompt-based Tuning Approach for Few-Shot Learning. [[PDF]](https://aclanthology.org/2022.findings-naacl.81/)
- R-Drop: Regularized Dropout for Neural Networks. [[PDF]](https://arxiv.org/abs/2106.14448)
- Openprompt: An open-source framework for prompt-learning. [[PDF]](https://arxiv.org/abs/2111.01998)

### Appendix

#### PromptTrainer Parameter List

| Parameter        | Type   | Default | Description                                             |
| ---------------- | ------ | ------- | ------------------------------------------------------- |
| max_seq_length   | int    | 512     | The maximum length of model input, including template part |
| freeze_plm       | bool   | False   | Whether to freeze PLM parameters during training        |
| freeze_dropout   | bool   | False   | Whether to freeze PLM parameters and disable dropout during training |
| use_rdrop        | bool   | False   | Whether to use RDrop strategy. See [RDrop Paper](https://arxiv.org/abs/2106.14448) |
| alpha_rdrop      | float  | 5.0     | Weight of RDrop loss                                    |
| use_rgl          | bool   | False   | Whether to use RGL strategy. See [RGL Paper](https://aclanthology.org/2022.findings-naacl.81/) |
| alpha_rgl        | float  | 0.5     | Weight of RGL loss                                      |
| ppt_learning_rate| float  | 1e-4    | Learning rate for continuous prompts and SoftVerbalizer's "hidden layer-label" layer parameters |
| ppt_weight_decay | float  | 0.0     | Weight decay for continuous prompts and SoftVerbalizer's parameters |
| ppt_adam_beta1   | float  | 0.9     | Beta1 for Adam optimizer of prompt-related parameters   |
| ppt_adam_beta2   | float  | 0.999   | Beta2 for Adam optimizer of prompt-related parameters   |
| ppt_adam_epsilon | float  | 1e-8    | Epsilon for Adam optimizer of prompt-related parameters |
