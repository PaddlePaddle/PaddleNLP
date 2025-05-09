====================================================================================
贡献预训练模型权重
====================================================================================

1. 模型网络结构类型
------------------------------------------------------------------------------------
PaddleNLP目前已支持绝大多数主流的预训练模型网络结构，既包括百度自研的预训练模型（如ERNIE系列），
也涵盖业界主流的预训练模型（如BERT，ALBERT，GPT，RoBERTa，XLNet等）。

PaddleNLP目前支持的预训练模型结构类型汇总可见
`Transformer预训练模型汇总 <https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html>`_
（持续增加中，也非常欢迎进行新模型贡献：`如何贡献新模型 <https://paddlenlp.readthedocs.io/zh/latest/community/contribute_models/contribute_new_models.html>`_ ）。

2. 模型参数权重类型
------------------------------------------------------------------------------------
非常欢迎大家贡献优质模型参数权重。
参数权重类型包括但不限于（以BERT模型网络为例）：

- PaddleNLP还未收录的BERT预训练模型参数权重
  （如 `bert-base-japanese-char <https://huggingface.co/cl-tohoku/bert-base-japanese-char>`_ ，`danish-bert-botxo <https://huggingface.co/Maltehb/danish-bert-botxo>`_ 等）；
- BERT模型在其他垂类领域（如数学，金融，法律，医学等）的预训练模型参数权重
  （如 `MathBERT <https://huggingface.co/tbs17/MathBERT>`_ ，`finbert <https://huggingface.co/ProsusAI/finbert>`_ 等）；
- 基于BERT在下游具体任务进行fine-tuning后的模型参数权重
  （如 `bert-base-multilingual-uncased-sentiment <https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment>`_ ，
  `bert-base-NER <https://huggingface.co/dslim/bert-base-NER>`_ 等）；
- 其他模型参数权重（任何你觉得有价值的模型参数权重）；

3. 参数权重格式转换
------------------------------------------------------------------------------------
当我们想要贡献github上开源的某模型权重时，但是发现该权重保存为其他的深度学习框架（PyTorch，TensorFlow等）的格式，
这就需要我们进行不同深度学习框架间的模型格式转换，下面的链接给出了一份详细的关于Pytorch到Paddle模型格式转换的教程：
`Pytorch到Paddle模型格式转换文档 <./convert_pytorch_to_paddle.rst>`_ 。

4. 进行贡献
------------------------------------------------------------------------------------
4.1 准备权重相关文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
一般来说，我们需要准备 **model_state.pdparams** ，**vocab.txt**，**tokenizer_config.json**
以及 **model_config.json** 这四个文件进行参数权重贡献。

- model_state.pdparams 文件可以通过上述的参数权重格式转换过程得到；
- vocab.txt 文件可以直接使用原始模型对应的vocab文件（根据模型对应tokenizer类型的不同，该文件名可能为spiece.model等）；
- model_config.json 文件可以参考对应 model.save_pretrained() 接口保存的model_config.json文件；
- tokenizer_config.json 文件可以参考对应 tokenizer.save_pretrained() 接口保存的model_config.json文件；

4.2 创建个人目录
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
如果你是首次进行权重贡献，那么你需要在 ``PaddleNLP/community/`` 下新建一个目录。
目录名称使用你的github名称，比如新建目录 ``PaddleNLP/community/yingyibiao/`` 。
如果已有个人目录，则可以跳过此步骤。

4.3 创建权重目录
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
在步骤4.2的个人目录下新建一个权重目录，权重目录名为本次贡献的模型权重名称。
比如我想贡献 ``bert-base-uncased-sst-2-finetuned`` 这个模型，
则新建权重目录 ``PaddleNLP/community/yingyibiao/bert-base-uncased-sst-2-finetuned/`` 。

4.4 在权重目录下添加PR（pull request）相关文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
在步骤4.3的目录下加入两个文件，分别为 ``README.md`` 和 ``files.json`` 。

- ``README.md`` 是对你贡献的权重的详细介绍，使用示例，权重来源等。
- ``files.json`` 为步骤4.1所得的权重相关文件以及对应地址。files.json文件内容示例如下，只需将地址中的 *yingyibiao* 和
  *bert-base-uncased-sst-2-finetuned* 分别更改为你的github用户名和权重名称。

.. code:: python

  {
    "model_config_file": "https://bj.bcebos.com/paddlenlp/models/transformers/community/yingyibiao/bert-base-uncased-sst-2-finetuned/model_config.json",
    "model_state": "https://bj.bcebos.com/paddlenlp/models/transformers/community/yingyibiao/bert-base-uncased-sst-2-finetuned/model_state.pdparams",
    "tokenizer_config_file": "https://bj.bcebos.com/paddlenlp/models/transformers/community/yingyibiao/bert-base-uncased-sst-2-finetuned/tokenizer_config.json",
    "vocab_file": "https://bj.bcebos.com/paddlenlp/models/transformers/community/yingyibiao/bert-base-uncased-sst-2-finetuned/vocab.txt"
  }

4.5 在github上提PR进行贡献
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- 第一次进行开源贡献的同学可以参考 `first-contributions <https://github.com/firstcontributions/first-contributions>`_ 。
- 模型权重贡献PR示例请参考 `bert-base-uncased-sst-2-finetuned PR <.>`_ 。