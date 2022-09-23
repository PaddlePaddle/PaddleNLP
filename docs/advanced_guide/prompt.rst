===========
Prompt API
===========

- \ `1. 提示学习是什么`_\

- \ `2. 如何定义模板`_\

  - \ `2.1 离散型模板`_\

  - \ `2.2 连续型模板`_\

  - \ `2.3 快速定义模板`_\

- \ `3. 如何定义答案空间映射`_\

  - \ `3.1 单掩码映射`_\

  - \ `3.2 多掩码映射`_\

  - \ `3.3 SoftVerbalizer`_\

- \ `4. 快速开始训练`_\

  - \ `4.1 数据准备`_\

  - \ `4.2 加载预训练模型`_\

  - \ `4.3 定义提示学习模型`_\

  - \ `4.4 构造 PromptTrainer 训练`_\

- \ `5. 实践教程`_\

  - \ `5.1 文本分类示例`_\

    5.2 其他任务示例（待更新）


1. 提示学习是什么
===============

**提示学习** （Prompt Learning）是在下游任务上使用预训练语言模型的一种新范式，通俗来讲，就是给预训练语言模型一个提示，来帮助它更好地理解学习的任务。该方法效果通常依赖于预训练语言模型的参数规模，主要用于小样本场景。

近年来，随着预训练语言模型规模的增长，其在各项自然语言处理任务上的表现越来越好，但对计算资源和存储资源的需求也越来越高。
与经典的“预训练-微调”范式不同，提示学习是根据模板（Template）修改输入文本，根据答案空间映射（Verbalizer）将标签映射为字典中的字词，将下游任务转化为预训练任务。
目的在于通过预训练任务与下游任务的统一，降低对预训练模型存储训练资源和数据资源的需求。

Prompt API 提供了这类算法实现的基本模块，支持PET、P-Tuning、RGL等经典算法的快速实现。


2. 如何定义模板
=============

**模板** （Template）用于定义如何将提示与输入文本相结合。目前模板主要分为离散型和连续型两种，构造方式也分为手工设计和自动学习。Prompt API 中提供了统一的数据结构来构造不同类型的模板，输入相应格式的字符串，通过解析得到对应的模板，即字典构成的列表。

2.1 离散型模板
------------------

离散型模板直接将预训练词表中的文本与输入文本拼接起来，可用于实现 PET、RGL 等算法。模板关键字 

- ``text`` ：指定数据集中原始输入文本对应的关键字。
- ``hard`` ：表示手工构造的提示。
- ``mask`` ：表示待预测的 token。
- ``sep`` 表示在句间增加 ``[SEP]`` 区分不同的句子。

模板定义
^^^^^^^

.. code-block:: python

    "{'text': 'text_a'}{'hard': '和'}{'text': 'text_b'}{'hard': '之间的逻辑关系是'}{'mask'}"


或者使用简化方式定义，省略关键字 ``hard`` 后与上述模板等价。

.. code-block:: python

    "{'text': 'text_a'}和{'text': 'text_b'}之间的逻辑关系是{'mask'}"

样本示例
^^^^^^^

例如，对于自然语言推理任务，给定样本

.. code-block:: python

    sample = InputExample(uid=0,
                          text_a="三种主要的抗组胺处方药(Allegra，Claritin，Zyrtec)似乎没有一种比其他的更有效",
                          text_b="抗组胺药的处方似乎都差不多",
                          labels="矛盾")

按照模板修改拼接后，最终输入模型的文本数据为

.. code-block::

    "三种主要的抗组胺处方药(Allegra，Claritin，Zyrtec)似乎没有一种比其他的更有效和抗组胺药的处方似乎都差不多之间的逻辑关系是[MASK]"


调用 API
^^^^^^^

.. code-block:: python

    from paddlenlp.prompt import ManualTemplate
    from paddlenlp.transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    template = ManualTemplate(tokenizer=tokenizer,
                              max_seq_length=512,
                              template="{'text': 'text_a'}和{'text': 'text_b'}之间的逻辑关系是{'mask'}")

其中初始化参数定义如下

- ``tokenizer`` ：预训练模型的 tokenizer，用于文本编码。
- ``max_seq_length`` ：定义输入模型文本的最大长度，包括提示部分。
- ``template`` ：定义手工模板和输入文本组合方式的字符串。

2.2 连续型模板
----------------

与离散型模板直接使用预训练词向量不同，连续型模板额外定义了一组连续向量作为提示，这一方法的优点是不需要很多实验经验和语言专业知识。可用于实现 P-Tuning 等算法。模板关键字 

- ``text`` ：指定数据集中原始输入文本对应的关键字。
- ``hard`` ：表示手工构造的提示。
- ``mask`` ：表示待预测的 token。
- ``sep`` 表示在句间增加 ``[SEP]`` 区分不同的句子。
- ``soft`` 表示连续向量。若值为 ``None`` ，则使用随机初始化；若值为文本，则使用预训练词向量中对应的向量进行初始化。

模板定义
^^^^^^^

- 使用单个提示向量：

.. code-block:: python

    "{'soft': None}{'text': 'text_a'}{'sep'}{'text': 'text_b'}"

- 使用 ``duplicate`` 参数快速定义多个提示向量：

.. code-block:: python

    "{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': None, `duplicate`:10}{'mask'}"

- 直接用 ``soft`` 的值定义多个提示向量并手工初始化：

.. code-block:: python

    "{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'soft': '请判断这两个句子间的逻辑关系：'}{'mask'}"

- 混合定义离散型和连续型提示：

.. code-block:: python

    "{'soft': '#自然语言推理#'}{'text': 'text_a'}{'sep'}{'text': 'text_b'}这两个句子间的逻辑关系是{'mask'}"


调用 API
^^^^^^^

.. code-block:: python

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

其中初始化参数定义如下

- ``tokenizer`` : 预训练模型的 tokenizer，用于文本编码。
- ``max_seq_length`` : 定义输入模型文本的最大长度，包括提示部分。
- ``model`` : 预训练语言模型，为了取预训练词向量用于连续型提示向量初始化。
- ``template`` : 定义连续型模板和输入文本的组合方式。
- ``prompt_encoder`` : 连续型提示向量的编码器，可选 ``mlp`` 和 ``lstm``。默认为 ``None`` ，即无编码器，直接使用向量。
- ``encoder_hidden_size`` : 连续型提示向量的维度。默认为 ``None`` ，即与预训练词向量维度相同。


2.3 快速定义模板
----------------

我们提供了 ``AutoTemplate`` API 以便快速定义单句输入的手工初始化的连续型模板，同时支持直接按照模板类型自动切换 ``ManualTemplate`` 和 ``SoftTemplate``。

模板定义
^^^^^^^

直接输入用于初始化连续型向量的文本，即可得到拼接到句尾的连续型模板输入。例如，

.. code-block:: python

    "这篇文章表达了怎样的情感？"

等价于

.. code-block:: python

    "{'text': 'text_a'}{'soft': '这篇文章表达了怎样的情感？'}{'mask'}"


调用 API
^^^^^^^

.. code-block:: python

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


其中初始化参数定义如下

- ``tokenizer`` ：预训练模型的 tokenizer，用于文本编码。
- ``max_seq_length`` ：定义输入模型文本的最大长度，包括提示部分。
- ``model`` ：预训练语言模型，为了取预训练词向量用于连续型提示向量初始化。
- ``template`` ：定义离散型/连续型模板和输入文本的组合方式。
- ``prompt_encoder`` ：连续型提示向量的编码器，可选 ``mlp`` 和 ``lstm`` 。默认为 ``None`` ，即无编码器，直接使用向量。
- ``encoder_hidden_size`` ：连续型提示向量的维度。默认为 ``None`` ，即与预训练词向量维度相同。


3. 如何定义答案空间映射
=====================

**答案空间映射** （Verbalizer）也是提示学习中可选的重要模块，用于建立预测词和标签之间的映射，将“预训练-微调”模式中预测标签的任务转换为预测模板中掩码位置的词语，从而将下游任务统一为预训练任务的形式。目前框架支持了离散型答案空间映射和 WARP 方法。


例如，在情感二分类任务中，微调方法和提示学习的标签体系如下

- **微调方式** : 数据集的标签为 ``负向`` 和 ``正向``，分别映射为 ``0`` 和 ``1`` ；

- **提示学习** : 通过下边的答案空间映射建立原始标签与预测词之间的映射。

.. code-block:: python
    
    {'负向': '不', '正向': '很'}


具体来说，对于模板 ``{'text':'text_a'}这句话表示我{'mask'}满意。`` ，我们使用映射 ``{'负向': '不', '正向': '很'}`` 将标签 ``负向`` 映射为 ``不`` ，将标签 ``正向`` 映射为 ``很`` 。也就是说，我们期望对于正向情感的文本，预测结果为 ``...这句话表示我很满意。`` ，对于负向情感的文本，预测结果为 ``...这句话表示我不满意。``


3.1 单掩码映射
-------------

``ManualVerbalizer`` 支持构造简单的单 ``{'mask'}`` 答案空间映射，直接作用于 ``AutoMaskedLM`` 模型结构。当标签对应的预测词长度大于 ``1`` 时取均值。

调用 API
^^^^^^^

.. code-block:: python

    from paddlenlp.prompt import ManualVerbalizer
    from paddlenlp.transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    verbalizer = ManualVerbalizer(tokenizer=tokenizer,
                                  labels=['负向', '正向'],
                                  label_words={'负向': '不', '正向': '很'},
                                  prefix=None)

其中初始化参数定义如下

- ``tokenizer`` : 预训练模型的 tokenizer，用于预测词的编码。
- ``labels`` : 数据集的原标签列表（可选）。
- ``label_words`` : 原标签到预测词之间的映射字典。如果同时定义了 ``labels`` ，二者的标签集合需要相同。
- ``prefix`` : 预测词解码前增加的前缀，用于 ``RoBERTa`` 等对前缀敏感的模型。默认为 ``None`` ，无前缀。


3.2 多掩码映射
-------------

``MultiMaskVerbalizer`` 继承自 ``ManualVerbalizer`` ，支持多 ``{'mask'}`` 答案空间映射。预测词长度需与 ``{'mask'}`` 长度一致。

调用 API
^^^^^^^

.. code-block:: python

    from paddlenlp.prompt import MultiMaskVerbalizer
    from paddlenlp.transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    verbalizer = MultiMaskVerbalizer(tokenizer=tokenizer,
                                     labels=['负向', '正向'],
                                     label_words={'负向': '生气', '正向': '高兴'},
                                     prefix=None)

其中初始化参数定义同 \ `3.1 单掩码映射`_\ 。


3.3 SoftVerbalizer
------------------

``SoftVerbalizer`` 修改了原 ``AutoMaskedLM`` 的模型结构，将预训练模型最后一层“隐藏层-词表”替换为“隐藏层-标签”的映射。该层网络的初始化参数由答案空间映射中的预测词词向量来决定，如果预测词长度大于 ``1`` ，则使用词向量均值进行初始化。当前支持的预训练模型包括 ``ErnieForMaskedLM`` 、 ``BertForMaskedLM`` 、 ``AlbertForMaskedLM`` 和 ``RobertaForMaskedLM`` 。


调用 API
^^^^^^^

.. code-block:: python

    from paddlenlp.prompt import MultiMaskVerbalizer
    rom paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    verbalizer = SoftVerbalizer(tokenizer=tokenizer,
                                model=model,
                                labels=['负向', '正向'],
                                label_words={'负向': '生气', '正向': '高兴'},
                                prefix=None)

其中初始化参数定义同 \ `3.1 单掩码映射`_\ ，此外

- ``model`` ：预训练语言模型，为了取预训练词向量用于“隐藏层-标签”网络的修改和初始化。

4. 快速开始训练
=============

本节介绍了如何使用 ``PromptTrainer`` 快速搭建提示训练流程。

4.1 数据准备
-----------

Prompt 框架定义了统一的样本结构 ``InputExample`` 以便进行数据处理，数据集样本需要封装在 ``MapDataset`` 中。

例如，对于文本语义相似度 BUSTM 数据集中的原始样本

.. code-block:: python

    data = [
        {'id': 3, 'sentence1': '你晚上吃了什么', 'sentence2': '你晚上吃啥了', 'label': '1'},
        {'id': 4, 'sentence1': '我想打开滴滴叫的士', 'sentence2': '你叫小欧吗', 'label': '0'},
        {'id': 5, 'sentence1': '女孩子到底是不是你', 'sentence2': '你不是女孩子吗', 'label': '1'}
    ]

需要转换为统一格式

.. code-block:: python
    
    from paddlenlp.datasets import MapDataset
    from paddlenlp.prompt import InputExample

    data_ds = MapDataset([InputExample(uid=example["id"],
                                       text_a=example["sentence1"],
                                       text_b=example["sentence2"],
                                       labels=example["label"]) for example in data])

4.2 加载预训练模型
----------------

如果使用答案空间映射，用 ``AutoModelForMaskedLM`` 和 ``AutoTokenizer`` 加载预训练模型参数。如果不使用答案空间映射，可将 ``AutoModelForMaskedLM`` 替换为任务对应的模型。

.. code-block:: python

    from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained("ernie-3.0-base-zh")
    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")

4.3 定义提示学习模型
------------------

对于文本分类任务，我们将模板预处理和答案空间映射封装为提示学习模型 ``PromptModelForSequenceClassification`` 。


.. code-block:: python

    # 定义模板
    template = AutoTemplate.create_from(template="{'text': 'text_a'}和{'text': 'text_b'}说的是{'mask'}同的事情。",
                                        tokenizer=tokenizer,
                                        max_seq_length=512)

    # 定义答案空间映射
    verbalizer = ManualVerbalizer(tokenizer=tokenizer,
                                  label_words={'0': '不', '1': '相'})

    # 定义文本分类提示模型
    prompt_model = PromptModelForSequenceClassification(
        model,
        template,
        verbalizer,
        freeze_plm=False,
        freeze_dropout=False)


其中提示模型初始化参数如下

- ``model`` : 预训练模型实例，支持 ``AutoModelForMaskedLM`` 和 ``AutoModelForSequenceClassification`` 。
- ``template`` : 模板实例。
- ``verbalizer`` : 答案空间映射实例。当设为 ``None`` 时，不使用答案空间映射，模型输出及损失值计算由 ``model`` 类型定义。
- ``freeze_plm`` : 在训练时是否固定预训练模型参数。对于规模较小的预训练模型，推荐更新预训练模型参数。
- ``freeze_dropout`` : 在训练时是否固定预训练模型参数并关闭 ``dropout`` 。 当 ``freeze_dropout=True`` ，``freeze_plm`` 也为 ``True`` 。


4.4 构造 PromptTrainer 训练
--------------------------

``PromptTrainer`` 继承自 ``Trainer`` ， 封装了数据处理，模型训练、测试，训练策略等，便于训练流程的快速搭建。

配置训练参数
^^^^^^^^^^

``PromptTuningArguments`` 继承自 ``TrainingArguments`` ，包含了提示学习的主要训练参数。其中 ``TrainingArguments`` 参数见 `Trainer API 文档 <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md>`_ ，其余参数详见 \ `Prompt 参数列表`_\ 。推荐使用 **命令行** 的形式进行参数配置，即

.. code-block:: shell

    python xxx.py --output_dir xxx --learning_rate


除了训练参数，还需要自定义数据和模型相关的参数。最后用 ``PdArgumentParser`` 输出参数。

.. code-block:: python
    
    from dataclasses import dataclass
    from paddlenlp.trainer import PdArgumentParser
    from paddlenlp.prompt import PromptTuningArguments

    @dataclass
    class DataArguments:
        data_path : str = field(default="./data", metadata={"help": "The path to dataset."})

    parser = PdArgumentParser((DataArguments, PromptTuningArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()


初始化和训练
^^^^^^^^^^^

除了上述准备，还需要定义损失函数和评估函数。

.. code-block:: python

    import paddle
    from paddle.metric import Accuracy

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
                            eval_dataset=data_ds,
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

5. 实践教程
==========

5.1 文本分类示例
--------------

- `多分类文本分类示例 <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class/few-shot>`_

- `多标签文本分类示例 <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label/few-shot>`_

- `多层次文本分类示例 <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical/few-shot>`_


附录  
----

Prompt 参数列表
^^^^^^^^^^^^^^

.. table:: Prompt 参数列表

=================== ======= ========= ========================================================
参数                 类型    默认值     含义
=================== ======= ========= ========================================================
max_seq_length      int     512       模型输入的最大长度，包括模板部分        
freeze_plm          bool    False     是否在训练时固定预训练模型的参数
freeze_dropout      bool    False     是否在训练时固定预训练模型的参数，同时关闭 dropout
use_rdrop           bool    False     是否使用 RDrop 策略，详见 `RDrop 论文 <https://arxiv.org/abs/2106.14448>`_
alpha_rdrop         float   5.0       RDrop Loss 的权重
use_rgl             bool    False     是否使用 RGL 策略，详见 `RGL 论文 <https://aclanthology.org/2022.findings-naacl.81/>`_
alpha_rgl           float   0.5       RGL Loss 的权重
ppt_learning_rate   float   1e-4      连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的学习率
ppt_weight_decay    float   0.0       连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的衰减参数
ppt_adam_beta1      float   0.9       连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的 beta1
ppt_adam_beta2      float   0.999     连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的 beta2
ppt_adam_epsilon    float   1e-8      连续型提示以及 SoftVerbalizer “隐藏层-标签”层参数的 epsilon
=================== ======= ========= ========================================================