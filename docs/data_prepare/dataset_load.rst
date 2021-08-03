============
加载数据集
============

快速加载内置数据集
---------------------

目前PaddleNLP内置20余个NLP数据集，涵盖阅读理解，文本分类，序列标注，机器翻译等多项任务。目前提供的数据集可以在 :doc:`数据集列表 <./dataset_list>` 中找到。

以 **msra_ner** 数据集为例:

.. code-block::

    >>> from paddlenlp.datasets import load_dataset
    >>> train_ds, test_ds = load_dataset("msra_ner", splits=("train", "test"))

:func:`load_dataset` 方法会从 :obj:`paddlenlp.datasets` 下找到msra_ner数据集对应的数据读取脚本（默认路径：paddlenlp/datasets/msra_ner.py），并调用脚本中 :class:`DatasetBuilder` 类的相关方法生成数据集。

生成数据集可以以 :class:`MapDataset` 和 :class:`IterDataset` 两种类型返回，分别是对 :class:`paddle.io.Dataset` 和 :class:`paddle.io.IterableDataset` 的扩展，只需在 :func:`load_dataset` 时设置 :attr:`lazy` 参数即可获取相应类型。:obj:`Flase` 对应返回 :class:`MapDataset` ，:obj:`True` 对应返回 :class:`IterDataset`，默认值为None，对应返回 :class:`DatasetBuilder` 默认的数据集类型，大多数为 :class:`MapDataset` 。

.. code-block::

    >>> from paddlenlp.datasets import load_dataset
    >>> train_ds = load_dataset("msra_ner", splits="train")  
    >>> print(type(train_ds))
    <class 'paddlenlp.datasets.dataset.MapDataset'> # Default
    >>> train_ds = load_dataset("msra_ner", splits="train", lazy=True) 
    >>> print(type(train_ds))
    <class 'paddlenlp.datasets.dataset.IterDataset'>

关于 :class:`MapDataset` 和 :class:`IterDataset` 功能和异同可以参考API文档 :doc:`datasets <../source/paddlenlp.datasets.dataset>`。

选择子数据集
^^^^^^^^^^^^^^^^^^^^^^^

有些数据集是很多子数据集的集合，每个子数据集都是一个独立的数据集。例如 **GLUE** 数据集就包含COLA, SST2, MRPC, QQP等10个子数据集。

:func:`load_dataset` 方法提供了一个 :attr:`name` 参数用来指定想要获取的子数据集。使用方法如下：

.. code-block::

    >>> from paddlenlp.datasets import load_dataset
    >>> train_ds, dev_ds = load_dataset("glue", name="cola", splits=("train", "dev"))  

以内置数据集格式读取本地数据集
-----------------------------

有的时候，我们希望使用数据格式与内置数据集相同的本地数据替换某些内置数据集的数据（例如参加SQuAD竞赛，对训练数据进行了数据增强）。 :func:`load_dataset` 方法提供的 :attr:`data_files` 参数可以实现这个功能。以 **SQuAD** 为例。

.. code-block::

    >>> from paddlenlp.datasets import load_dataset
    >>> train_ds, dev_ds = load_dataset("squad", data_files=("my_train_file.json", "my_dev_file.json"))
    >>> test_ds = load_dataset("squad", data_files="my_test_file.json")

.. note::

    对于某些数据集，不同的split的读取方式不同。对于这种情况则需要在 :attr:`splits` 参数中以传入与 :attr:`data_files` **一一对应** 的split信息。
    
    此时 :attr:`splits` 不再代表选取的内置数据集，而代表以何种格式读取本地数据集。
    
    下面以 **COLA** 数据集为例：

    .. code-block::

        >>> from paddlenlp.datasets import load_dataset
        >>> train_ds, test_ds = load_dataset("glue", "cola", splits=["train", "test"], data_files=["my_train_file.csv", "my_test_file.csv"])

    **另外需要注意数据集的是没有默认加载选项的，**:attr:`splits` **和**:attr:`data_files` **必须至少指定一个。**