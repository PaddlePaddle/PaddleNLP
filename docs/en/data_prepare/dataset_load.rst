============
Loading Datasets
============

Quickly Load Built-in Datasets
------------------------------

PaddleNLP currently provides 20+ built-in NLP datasets covering tasks like reading comprehension, text classification, sequence labeling, machine translation, etc. All available datasets can be found in :doc:`Dataset List <./dataset_list>`.

Take the **msra_ner** dataset as an example:

.. code-block::

    >>> from paddlenlp.datasets import load_dataset
    >>> train_ds, test_ds = load_dataset("msra_ner", splits=("train", "test"))

The :func:`load_dataset` method will locate the corresponding data loading script for msra_ner dataset in :obj:`paddlenlp.datasets` (default path: paddlenlp/datasets/msra_ner.py), and call the relevant methods of the :class:`DatasetBuilder` class in the script to generate the dataset.

The generated dataset can be returned as either :class:`MapDataset` or :class:`IterDataset`, which are extensions of :class:`paddle.io.Dataset` and :class:`paddle.io.IterableDataset` respectively. Simply set the :attr:`lazy` parameter in :func:`load_dataset` to get the corresponding type. :obj:`False` returns :class:`MapDataset` while :obj:`True` returns :class:`IterDataset`. The default value is None, which returns the dataset type predefined by :class:`DatasetBuilder` (mostly :class:`MapDataset`).

.. code-block::

    >>> from paddlenlp.datasets import load_dataset
    >>> train_ds = load_dataset("msra_ner", splits="train")
    >>> print(type(train_ds))
    <class 'paddlenlp.datasets.dataset.MapDataset'> # Default
    >>> train_ds = load_dataset("msra_ner", splits="train", lazy=True)
    >>> print(type(train_ds))
    <class 'paddlenlp.datasets.dataset.IterDataset'>

For details about :class:`MapDataset` and :class:`IterDataset` features and differences, please refer to the API documentation :doc:`datasets <../source/paddlenlp.datasets.dataset>`.

Selecting Subsets
^^^^^^^^^^^^^^^^^

Some datasets are collections of multiple subsets, where each subset is an independent dataset. For example, the **GLUE** dataset contains 10 subsets like COLA, SST2, MRPC, QQP, etc.

The :func:`load_dataset` method provides a :attr:`name` parameter to specify subsets. For example, to load the SQuAD dataset from the XTREME benchmark:

.. code-block::

    >>> from paddlenlp.datasets import load_dataset
    >>> squad_train = load_dataset('xtreme', name='squad', splits='train')

The data loading script will automatically add the ``name`` parameter to the dataset file path. For example, the files for the SQuAD subset are typically stored in the ``xtreme/squad`` directory.
The `splits` parameter is used to specify the subsets of the dataset to retrieve. Usage example:

.. code-block::

    >>> from paddlenlp.datasets import load_dataset
    >>> train_ds, dev_ds = load_dataset("glue", name="cola", splits=("train", "dev"))

Reading Local Datasets in Built-in Dataset Format
-------------------------------------------------

Sometimes we may want to use local data that shares the same format as built-in datasets to replace some built-in data (e.g., for data augmentation in SQuAD competition training). The :func:`load_dataset` method's :attr:`data_files` parameter enables this functionality. Taking **SQuAD** as an example:

.. code-block::

    >>> from paddlenlp.datasets import load_dataset
    >>> train_ds, dev_ds = load_dataset("squad", data_files=("my_train_file.json", "my_dev_file.json"))
    >>> test_ds = load_dataset("squad", data_files="my_test_file.json")

.. note::

    For some datasets, different splits may require different reading approaches. In such cases, corresponding split information must be provided in the :attr:`splits` parameter, which should **exactly match** the :attr:`data_files` entries.

    In this scenario, :attr:`splits` no longer represents selected built-in datasets, but rather specifies the format for reading local data.

    Here's an example using the **COLA** dataset:

    .. code-block::

        >>> from paddlenlp.datasets import load_dataset
        >>> train_ds, test_ds = load_dataset("glue", "cola", splits=["train", "test"], data_files=["my_train_file.csv", "my_test_file.csv"])

    **Important note:** The dataset has no default loading options - you must specify at least one of :attr:`splits` or :attr:`data_files`.