==========================================
模型格式转换
==========================================

0. 前言
------------------------------------------
本文将介绍如何进行不同框架下的模型权重转换（以模型权重从PyTorch框架到Paddle框架的格式转换为例）。

模型格式转换的过程需要用户对模型结构有一个较详细的了解，成功完成模型格式转换也会有助于加深用户对该模型结构的理解。
让我们开始这个有趣的过程吧！

1. 模型权重文件概述
------------------------------------------
不管在什么框架下，当我们保存训练好的模型时，我们都需要将模型的参数权重持久化保存下来；
当我们加载一个保存好的模型时，我们都需要将参数权重加载并重新赋值给相应的模型。

PyTorch和Paddle都是通过序列化和反序列化模型的 ``state dict`` （状态字典）来进行参数权重的存储和加载的。
``state dict`` 从数据结构上来看就是一个字典（比如Python中的dict），
其中key是模型参数的名称（数据类型为string），而value则为key所对应的值（数据类型为Tensor）。
参数存储时，先获取目标对象的 ``state dict`` ，然后将 ``state dict`` 存储至磁盘；
参数载入时，先从磁盘载入保存的 ``state dict`` ，然后通过 ``set_state_dict()`` 方法配置到目标对象中。

按照约定俗成的命名规则，Paddle框架保存的模型文件名一般后缀为 `'.pdparams'` ，
PyTorch框架保存的模型文件名一般后缀为 `'.pt'` 、 `'.pth'` 或者 `'.bin'` 。
虽然后缀并不影响模型的保存和加载，但我们一般都会遵循这个命名规范。

2. 模型的 ``state dict`` 概述
------------------------------------------
刚刚我们简单介绍了一下模型文件和其中存储的 ``state dict`` ，
下面让我们来看一个具体的例子来对 ``state dict`` 有更进一步的了解。

``LeNet`` 是由Yann LeCun等人在1998年提出的一个CNN网络模型，并且成功应用于手写数字识别系统。
Paddle集成了 ``LeNet`` 这个简单的模型，我们可以一键进行模型加载，
下面的代码实现了该模型的加载和对应 ``state dict`` 的输出：

.. code:: python

    >>> import paddle
    >>> from paddle.vision.models import LeNet
    >>> model = LeNet()
    >>> model.state_dict().keys()  # 输出state_dict的所有keys
    odict_keys(['features.0.weight', 'features.0.bias', 'features.3.weight', 'features.3.bias',
                'fc.0.weight', 'fc.0.bias', 'fc.1.weight', 'fc.1.bias', 'fc.2.weight', 'fc.2.bias'])

    >>> model.state_dict()['features.0.weight']  # 输出 'features.0.weight' 对应的value
    Parameter containing:
    Tensor(shape=[6, 1, 3, 3], dtype=float32, place=CPUPlace, stop_gradient=False,
           [[[[-0.31584871,  0.27280194, -0.43816274],
              [ 0.06681869,  0.44526964,  0.80944657],
              [ 0.05796078,  0.57411081,  0.15335406]]],
            ...
            ...
            [[[-0.07211500, -0.14458601, -1.11733580],
              [ 0.53036308, -0.19761689,  0.56962037],
              [-0.09760553, -0.02011104, -0.50577533]]]])


我们可以通过 ``model.state_dict().keys()`` 来获取模型的所有参数名称。
可以看到 ``LeNet`` 一共有10组参数，分别为：*'features.0.weight'*、*'features.0.bias'*、*'features.3.weight'*
、*'features.3.bias'*、*'fc.0.weight'*、*'fc.0.bias'*、*'fc.1.weight'*、*'fc.1.bias'*、*'fc.2.weight'* 和 *'fc.2.bias'*。

通过查询 ``model.state_dict()['features.0.weight']`` 可以查看 **'features.0.weight'** 这个参数的具体权重数值。
上述输出显示该权重是一个dtype=float32，shape=[6, 1, 3, 3]的Tensor。

3. 利用 ``state dict`` 进行权重格式转换
------------------------------------------
了解了模型的存储和加载以及相关的 ``state dict`` 之后，我们来看一下模型格式的转换的具体步骤。
一般来说，我们可以通过 ``state dict`` 的相互转换来帮助我们进行模型格式的转换。

以从PyTorch框架到Paddle框架的模型权重转换为例，转换的具体流程为：

1. 加载PyTorch模型得到 ``state dict``
2. PyTorch下的 ``state dict`` 转换为Paddle下的 ``state dict``
3. 保存Paddle下的 ``state dict`` 得到Paddle模型。

下面我们来看一个具体的例子：``'bert-base-uncased'`` 是一个谷歌开源的12层的bert英文模型。
PaddleNLP（Paddle框架）和HuggingFace的transformers（PyTorch框架）里都集成了这个模型，
两者参数量和具体参数数值是完全一致的。我们可以来加载对比这两个模型的 ``state dict`` 来了解转换的细节。

3.1 PyTorch框架下的 ``state dict``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
首先加载transformers下的 ``'bert-base-uncased'`` 模型，

.. code:: python

    >>> import torch
    >>> model_name = "bert-base-uncased"
    >>> # 模型下载地址： https://huggingface.co/bert-base-uncased/blob/main/pytorch_model.bin
    >>> model_file = "pytorch_model.bin"
    >>> pytorch_state_dict = torch.load(model_file)
    >>> pytorch_state_dict.keys()
    odict_keys(['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight',
                'bert.embeddings.LayerNorm.gamma', 'bert.embeddings.LayerNorm.beta',
                'bert.encoder.layer.0.attention.self.query.weight', 'bert.encoder.layer.0.attention.self.query.bias',
                'bert.encoder.layer.0.attention.self.key.weight', 'bert.encoder.layer.0.attention.self.key.bias',
                'bert.encoder.layer.0.attention.self.value.weight', 'bert.encoder.layer.0.attention.self.value.bias',
                'bert.encoder.layer.0.attention.output.dense.weight', 'bert.encoder.layer.0.attention.output.dense.bias',
                'bert.encoder.layer.0.attention.output.LayerNorm.gamma', 'bert.encoder.layer.0.attention.output.LayerNorm.beta',
                'bert.encoder.layer.0.intermediate.dense.weight', 'bert.encoder.layer.0.intermediate.dense.bias',
                'bert.encoder.layer.0.output.dense.weight', 'bert.encoder.layer.0.output.dense.bias',
                'bert.encoder.layer.0.output.LayerNorm.gamma', 'bert.encoder.layer.0.output.LayerNorm.beta',
                'bert.encoder.layer.1'...
                'bert.encoder.layer.2'...
                .
                .
                .
                'bert.encoder.layer.9'...
                'bert.encoder.layer.10'...
                'bert.encoder.layer.11.attention.self.query.weight', 'bert.encoder.layer.11.attention.self.query.bias',
                'bert.encoder.layer.11.attention.self.key.weight', 'bert.encoder.layer.11.attention.self.key.bias',
                'bert.encoder.layer.11.attention.self.value.weight', 'bert.encoder.layer.11.attention.self.value.bias',
                'bert.encoder.layer.11.attention.output.dense.weight', 'bert.encoder.layer.11.attention.output.dense.bias',
                'bert.encoder.layer.11.attention.output.LayerNorm.gamma', 'bert.encoder.layer.11.attention.output.LayerNorm.beta',
                'bert.encoder.layer.11.intermediate.dense.weight', 'bert.encoder.layer.11.intermediate.dense.bias',
                'bert.encoder.layer.11.output.dense.weight', 'bert.encoder.layer.11.output.dense.bias',
                'bert.encoder.layer.11.output.LayerNorm.gamma', 'bert.encoder.layer.11.output.LayerNorm.beta',
                'bert.pooler.dense.weight', 'bert.pooler.dense.bias',
                'cls.predictions.bias', 'cls.predictions.transform.dense.weight',
                'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.gamma',
                'cls.predictions.transform.LayerNorm.beta', 'cls.predictions.decoder.weight',
                'cls.seq_relationship.weight', 'cls.seq_relationship.bias'])

\**odict_keys**（ordered_dict keys）所显示的是PyTorch模型文件所对应的 ``state dict`` 的keys:
我们仔细观察一下可以发现参数可以分成几大模块：**embeddings** 模块，
**encoder_layers** 模块, **pooler** 模块和 **cls** 模块。

我们可以结合bert的具体结构来解读一下各个模块：

- **embeddings** 模块

  *'bert.embeddings'* 开头的各个参数是embeddings模块的参数，
  包括word_embeddings矩阵，position_embeddings矩阵，token_type_embeddings矩阵以及embeddings模块的LayerNorm层参数等。
- **encoder_layers** 模块

  *'bert.encoder.layer'*开头的各个参数是各encoder层的参数，
  可以看到 ``'bert-base-uncased'`` 模型一共有12层encoder（编号0-11），每一层encoder的结构都相同。
  每一层encoder主要由一个*self-attention*模块和一个*feed-forward*模块构成。
  我们具体来看一下第1层encoder的参数（编号为0，'bert.encoder.layer.0'开头的参数）：

  首先是*self-attention*模块：

  * *'attention.self.query'*，*'attention.self.key'* 和 *'attention.self.value'*
    分别代表self-attention结构里面的query矩阵，key矩阵和value矩阵。
  * *'attention.output.dense'* 是self-attention结构的线性层。
  * *'attention.output.LayerNorm'* 则是self-attention结构后的LayerNorm层。

  接下来是*feed-forward*模块，对应 'intermediate.dense' 和 'output.dense' 开头的参数
  。*feed-forward*之后还有一个*LayerNorm*层，对应的是 'output.LayerNorm' 开头的参数。
- **pooler** 模块

  pooler模块在最后一层encoder之后，是我们对最后一层encoder输出的池化操作，
- **cls** 模块

  cls模块是我们计算mlm（masked language model）和next sentence prediction（nsp）任务的结构。
  'cls.predictions'开头的参数是我们做mlm任务时的参数，'cls.seq_relationship'开头的参数是我们做nsp预测任务时的参数。

3.2 Paddle框架下的 ``state dict``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
相信到现在，我们已经对bert这个模型的结构以及相应的具体参数有了更进一步的了解。
接下来我们来加载PaddleNLP下的模型：

.. code:: python

    >>> import paddle
    >>> model_name = "bert-base-uncased"
    >>> # 模型下载地址： https://bj.bcebos.com/paddlenlp/models/transformers/bert-base-uncased.pdparams
    >>> model_file = "bert-base-uncased.pdparams"
    >>> paddle_state_dict = paddle.load(model_file)
    >>> paddle_state_dict.keys()
    dict_keys(['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight',
                'bert.embeddings.layer_norm.weight', 'bert.embeddings.layer_norm.bias',
                'bert.encoder.layers.0.self_attn.q_proj.weight', 'bert.encoder.layers.0.self_attn.q_proj.bias',
                'bert.encoder.layers.0.self_attn.k_proj.weight', 'bert.encoder.layers.0.self_attn.k_proj.bias',
                'bert.encoder.layers.0.self_attn.v_proj.weight', 'bert.encoder.layers.0.self_attn.v_proj.bias',
                'bert.encoder.layers.0.self_attn.out_proj.weight', 'bert.encoder.layers.0.self_attn.out_proj.bias',
                'bert.encoder.layers.0.linear1.weight', 'bert.encoder.layers.0.linear1.bias',
                'bert.encoder.layers.0.linear2.weight', 'bert.encoder.layers.0.linear2.bias',
                'bert.encoder.layers.0.norm1.weight', 'bert.encoder.layers.0.norm1.bias',
                'bert.encoder.layers.0.norm2.weight', 'bert.encoder.layers.0.norm2.bias',
                'bert.encoder.layers.1'...
                ...
                ...
                ...
                'bert.encoder.layers.10'...
                'bert.encoder.layers.11.self_attn.q_proj.weight', 'bert.encoder.layers.11.self_attn.q_proj.bias',
                'bert.encoder.layers.11.self_attn.k_proj.weight', 'bert.encoder.layers.11.self_attn.k_proj.bias',
                'bert.encoder.layers.11.self_attn.v_proj.weight', 'bert.encoder.layers.11.self_attn.v_proj.bias',
                'bert.encoder.layers.11.self_attn.out_proj.weight', 'bert.encoder.layers.11.self_attn.out_proj.bias',
                'bert.encoder.layers.11.linear1.weight', 'bert.encoder.layers.11.linear1.bias',
                'bert.encoder.layers.11.linear2.weight', 'bert.encoder.layers.11.linear2.bias',
                'bert.encoder.layers.11.norm1.weight', 'bert.encoder.layers.11.norm1.bias',
                'bert.encoder.layers.11.norm2.weight', 'bert.encoder.layers.11.norm2.bias',
                'bert.pooler.dense.weight', 'bert.pooler.dense.bias',
                'cls.predictions.decoder_weight', 'cls.predictions.decoder_bias',
                'cls.predictions.transform.weight', 'cls.predictions.transform.bias',
                'cls.predictions.layer_norm.weight', 'cls.predictions.layer_norm.bias',
                'cls.seq_relationship.weight', 'cls.seq_relationship.bias'])

Paddle模型的 ``state dict`` 是通过一个dict来进行存储，可以看到，两者的 ``state dict`` 是十分相似的。
我们对比一下两者：

- 两者的存储是相似的，PyTorch里使用的是python中的ordered_dict来存储模型的参数状态，
  在Paddle中则使用的是python中的dict来来进行存储。
- 两者的结构也是相似的，都可以分成embeddings，encoder_layer, pooler, cls等
  模块（当然这也很直观，毕竟两者的模型结构和模型参数是完全一致的）。
- 同时两者也存在一些区别，两者的 ``state dict`` 的keys有一些细微的差异，这是由于模型代码的具体实现的参数命名差异所造成的。

3.3 PyTorch和Paddle的 ``state dict`` 对比
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
我们接下来对上述两个 ``state dict`` 的参数名称以及对应权重来做一一对应。
下面的表格是整理好的 ``state_dict`` 对应关系表格（同一行代表着相对应的参数）：

+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| Keys (PyTorch)                                         | Shape (PyTorch)            | Keys (Paddle)                                    | Shape (Paddle)            |
+========================================================+============================+==================================================+===========================+
| bert.embeddings.word_embeddings.weight                 | [30522, 768]               | bert.embeddings.word_embeddings.weight           | [30522, 768]              |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.embeddings.position_embeddings.weight             | [512, 768]                 | bert.embeddings.position_embeddings.weight       | [512, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.embeddings.token_type_embeddings.weight           | [2, 768]                   | bert.embeddings.token_type_embeddings.weight     | [2, 768]                  |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.embeddings.LayerNorm.gamma                        | [768]                      | bert.embeddings.layer_norm.weight                | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.embeddings.LayerNorm.beta                         | [768]                      | bert.embeddings.layer_norm.bias                  | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.query.weight       | [768, 768]                 | bert.encoder.layers.0.self_attn.q_proj.weight    | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.query.bias         | [768]                      | bert.encoder.layers.0.self_attn.q_proj.bias      | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.key.weight         | [768, 768]                 | bert.encoder.layers.0.self_attn.k_proj.weight    | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.key.bias           | [768]                      | bert.encoder.layers.0.self_attn.k_proj.bias      | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.value.weight       | [768, 768]                 | bert.encoder.layers.0.self_attn.v_proj.weight    | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.value.bias         | [768]                      | bert.encoder.layers.0.self_attn.v_proj.bias      | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.output.dense.weight     | [768, 768]                 | bert.encoder.layers.0.self_attn.out_proj.weight  | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.output.dense.bias       | [768]                      | bert.encoder.layers.0.self_attn.out_proj.bias    | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.output.LayerNorm.gamma  | [768]                      | bert.encoder.layers.0.norm1.weight               | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.output.LayerNorm.beta   | [768]                      | bert.encoder.layers.0.norm1.bias                 | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.intermediate.dense.weight         | [3072, 768]                | bert.encoder.layers.0.linear1.weight             | [768, 3072]               |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.intermediate.dense.bias           | [3072]                     | bert.encoder.layers.0.linear1.bias               | [3072]                    |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.output.dense.weight               | [768, 3072]                | bert.encoder.layers.0.linear2.weight             | [3072, 768]               |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.output.dense.bias                 | [768]                      | bert.encoder.layers.0.linear2.bias               | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.output.LayerNorm.gamma            | [768]                      | bert.encoder.layers.0.norm2.weight               | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.output.LayerNorm.beta             | [768]                      | bert.encoder.layers.0.norm2.bias                 | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.pooler.dense.weight                               | [768, 768]                 | bert.pooler.dense.weight                         | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.pooler.dense.bias                                 | [768]                      | bert.pooler.dense.bias                           | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.bias                                   | [30522]                    | cls.predictions.decoder_bias                     | [30522]                   |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.transform.dense.weight                 | [768, 768]                 | cls.predictions.transform.weight                 | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.transform.dense.bias                   | [768]                      | cls.predictions.transform.bias                   | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.transform.LayerNorm.gamma              | [768]                      | cls.predictions.layer_norm.weight                | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.transform.LayerNorm.beta               | [768]                      | cls.predictions.layer_norm.bias                  | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.decoder.weight                         | [30522, 768]               | cls.predictions.decoder_weight                   | [30522, 768]              |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.seq_relationship.weight                            | [2, 768]                   | cls.seq_relationship.weight                      | [768, 2]                  |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.seq_relationship.bias                              | [2]                        | cls.seq_relationship.bias                        | [2]                       |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+

正确地对应好 ``state dict`` 的参数以及权重有助于我们正确地进行 ``state dict`` 的转换。

我们从参数名称上能看出基本的一个对应关系，例如：

* `bert.embeddings.LayerNorm.gamma` 对应 `bert.embeddings.layer_norm.weight` ；
* `bert.embeddings.LayerNorm.beta` 对应 `bert.embeddings.layer_norm.bias` ；
* `bert.encoder.layer.0.attention.self.query.weight` 对应 `bert.encoder.layers.0.self_attn.q_proj.weight` ；
* `bert.encoder.layer.0.attention.self.query.bias` 对应 `bert.encoder.layers.0.self_attn.q_proj.bias`。

两者的顺序是基本一致的，但也有一些例外，例如：

* `bert.encoder.layers.0.norm1.weight` 对应 `bert.encoder.layer.0.attention.output.LayerNorm.gamma` ；
* `bert.encoder.layers.0.norm1.bias` 对应 `bert.encoder.layer.0.attention.output.LayerNorm.beta` ；
* `bert.encoder.layer.0.intermediate.dense.weight` 对应 `bert.encoder.layers.0.linear1.weight` ；
* `bert.encoder.layer.0.output.dense.weight` 对应 `bert.encoder.layers.0.linear2.weight` ；
* `bert.encoder.layer.0.output.LayerNorm.gamma` 对应 `bert.encoder.layers.0.norm2.weight`。

正确的参数对应关系可能需要我们阅读具体的代码进行判断。
在上面的表格中我们已经将两者的keys准确地一一对应了。建立好了keys的对应关系之后，我们可以进行values的对应。

如果你仔细观察表格，会发现有些参数对应的values形状存在差异。
比如 ``bert.encoder.layer.0.intermediate.dense.weight`` 和 ``bert.encoder.layers.0.linear1.weight``
这两个keys是相对应的一组参数名，但是他们的values形状却不相同；前者是 ``[3072, 768]`` ，
后者是 ``[768, 3072]`` ，两者刚好是一个转置的关系。这是因为PyTorch对于 ``nn.Linear`` 模块的保存是将权重的shape进行转置后保存的。
所以在我们进行 ``state dict`` 转换的时候，需要注意做好shape的转换（例如将PyTorch模型里
nn.Linear层对应的参数权重转置处理后生成Paddle的参数权重）。

另外还需要注意其他一些细节，这里列出来几个可能会遇到的情景以供参考：

- 有些模型结构可能在实现时对参数的处理有差异导致存在参数的拆分或者合并等操作，
  此时我们需要进行参数多对一或者一对多的映射，同时将对应的values拆分或者合并。
- 还有存在batch norm层时，我们需要注意todo。

3.4 bert模型转换代码
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
下一步就是进行最关键的模型转换环节。
这一步十分关键，正确地进行 ``state dict`` 的转换才能确保我们通过精度验证。

下面是进行模型转换的代码（PyTorch转换为Paddle）：

.. code:: python

    import paddle
    import torch
    import numpy as np

    torch_model_path = "pytorch_model.bin"
    torch_state_dict = torch.load(torch_model_path)

    paddle_model_path = "bert_base_uncased.pdparams"
    paddle_state_dict = {}

    # State_dict's keys mapping: from torch to paddle
    keys_dict = {
        # about embeddings
        "embeddings.LayerNorm.gamma": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.beta": "embeddings.layer_norm.bias",

        # about encoder layer
        'encoder.layer': 'encoder.layers',
        'attention.self.query': 'self_attn.q_proj',
        'attention.self.key': 'self_attn.k_proj',
        'attention.self.value': 'self_attn.v_proj',
        'attention.output.dense': 'self_attn.out_proj',
        'attention.output.LayerNorm.gamma': 'norm1.weight',
        'attention.output.LayerNorm.beta': 'norm1.bias',
        'intermediate.dense': 'linear1',
        'output.dense': 'linear2',
        'output.LayerNorm.gamma': 'norm2.weight',
        'output.LayerNorm.beta': 'norm2.bias',

        # about cls predictions
        'cls.predictions.transform.dense': 'cls.predictions.transform',
        'cls.predictions.decoder.weight': 'cls.predictions.decoder_weight',
        'cls.predictions.transform.LayerNorm.gamma': 'cls.predictions.layer_norm.weight',
        'cls.predictions.transform.LayerNorm.beta': 'cls.predictions.layer_norm.bias',
        'cls.predictions.bias': 'cls.predictions.decoder_bias'
    }


    for torch_key in torch_state_dict:
        paddle_key = torch_key
        for k in keys_dict:
            if k in paddle_key:
                paddle_key = paddle_key.replace(k, keys_dict[k])

        if ('linear' in paddle_key) or ('proj' in  paddle_key) or ('vocab' in  paddle_key and 'weight' in  paddle_key) or ("dense.weight" in paddle_key) or ('transform.weight' in paddle_key) or ('seq_relationship.weight' in paddle_key):
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy().transpose())
        else:
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())

        print("torch: ", torch_key,"\t", torch_state_dict[torch_key].shape)
        print("paddle: ", paddle_key, "\t", paddle_state_dict[paddle_key].shape, "\n")

    paddle.save(paddle_state_dict, paddle_model_path)


我们来看一下这份转换代码：
我们需要下载好待转换的PyTorch模型，并加载模型得到**torch_state_dict**
；**paddle_state_dict** 和 **paddle_model_path** 则定义了转换后的 ``state dict`` 和模型文件路径；
代码中 **keys_dict** 定义了两者keys的映射关系（可以通过上面的表格对比得到）。

下一步就是最关键的 *paddle_state_dict* 的构建，我们对 *torch_state_dict* 里的每一个key都进行映射，
得到对应的 *paddle_state_dict* 的key。获取 *paddle_state_dict* 的key之后我们需要
对 *torch_state_dict* 的value进行转换，如果key对应的结构是 ``nn.Linear`` 模块的话，
我们还需要进行value的transpose操作。

最后我们保存得到的 *paddle_state_dict* 就能得到对应的Paddle模型。
至此我们已经完成了模型的转换工作，得到了Paddle框架下的模型 ``"model_state.pdparams"`` 。

4. 模型权重验证
------------------------------------------
得到了模型权重后，我们还需要进行精度的对齐来验证我们上述转换的正确性。
我们可以通过前向推理和下游任务fine-tuning这两个任务进行精度对齐验证。

4.1 对齐前向精度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
前向精度的对齐十分简单，我们只需要保证两者输入是一致的前提下，观察得到的输出是否一致。
这里有几个注意事项，我们运行推理时需要打开eval模式，设置dropout为0等操作去除随机性造成的影响。

除了得到的模型权重文件，我们还需要准备模型配置文件。将模型权重文件（model_state.pdparams）和模型配置文件（model_config.json）
这两个文件放在同一个路径下，我们就可以进行模型前向精度的对齐验证，下面提供了bert模型对齐前向精度的代码示例：

.. code:: python

    text = "Welcome to use paddle paddle and paddlenlp!"
    torch_model_name = "bert-base-uncased"
    paddle_model_name = "bert-base-uncased"

    # torch output
    import torch
    import transformers
    from transformers.models.bert import *

    # torch_model = BertForPreTraining.from_pretrained(torch_model_name)
    torch_model = BertModel.from_pretrained(torch_model_name)
    torch_tokenizer = BertTokenizer.from_pretrained(torch_model_name)
    torch_model.eval()

    torch_inputs = torch_tokenizer(text, return_tensors="pt")
    torch_outputs = torch_model(**torch_inputs)

    torch_logits = torch_outputs[0]
    torch_array = torch_logits.cpu().detach().numpy()
    print("torch_prediction_logits shape:{}".format(torch_array.shape))
    print("torch_prediction_logits:{}".format(torch_array))


    # paddle output
    import paddle
    import paddlenlp
    from paddlenlp.transformers.bert.modeling import *
    import numpy as np

    # paddle_model = BertForPretraining.from_pretrained(paddle_model_name)
    paddle_model = BertModel.from_pretrained(paddle_model_name)
    paddle_tokenizer = BertTokenizer.from_pretrained(paddle_model_name)
    paddle_model.eval()

    paddle_inputs = paddle_tokenizer(text)
    paddle_inputs = {k:paddle.to_tensor([v]) for (k, v) in paddle_inputs.items()}
    paddle_outputs = paddle_model(**paddle_inputs)

    paddle_logits = paddle_outputs[0]
    paddle_array = paddle_logits.numpy()
    print("paddle_prediction_logits shape:{}".format(paddle_array.shape))
    print("paddle_prediction_logits:{}".format(paddle_array))


    # the output logits should have the same shape
    assert torch_array.shape == paddle_array.shape, "the output logits should have the same shape, but got : {} and {} instead".format(torch_array.shape, paddle_array.shape)
    diff = torch_array - paddle_array
    print(np.amax(abs(diff)))

代码最后会打印模型输出矩阵的每个元素最大差值，根据这个差值可以判定我们是否对齐了前向精度。

4.2 下游任务fine-tuning验证（可选）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
当我们对齐前向精度时，一般来说我们的模型转换就已经成功了。我们还可以运行下游任务fine-tuning进行double check。
同样的，我们需要设置相同的训练数据，相同的训练参数，相同的训练环境进行fine-tuning来对比两者的收敛性以及收敛指标。

5. 写在最后
------------------------------------------
恭喜你成功完成了模型权重的格式转换工作！欢迎向PaddleNLP提PR共享你的模型，
这样每一个使用PaddleNLP的用户都能使用你共享的模型哦～
