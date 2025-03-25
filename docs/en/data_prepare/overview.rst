============
Overview
============

Datasets and data processing are among the most critical components in NLP tasks. To help users complete this phase with lower learning costs, PaddleNLP offers the following features:

- **Powerful APIs**: Assist users in handling data processing workflows for most common NLP tasks.
- **Flexible Encapsulation**: Maintains low coupling and high cohesion between modules, allowing users to meet specific data processing requirements through inheritance and customization.
- **Built-in Datasets**: Cover most NLP tasks, paired with concise and easy-to-use dataset loading protocols and contribution guidelines. More friendly for beginners and community contributors.

Core APIs
----------

- :func:`load_dataset`: A quick dataset loading interface that generates datasets by calling methods of :class:`DatasetBuilder` subclasses with dataset reader script names and other parameters. For detailed instructions, refer to :doc:`Loading Datasets <./dataset_load>`.
- :class:`DatasetBuilder`: A base class inherited by all built-in datasets. Its main functionality includes downloading dataset files and generating Dataset objects. Most methods are encapsulated and not exposed to contributors. Contributors can submit datasets to the community by overriding methods like :func:`_get_data` and :func:`_read`. For details, see :doc:`How to Contribute Datasets </community/contribute_dataset>`.
- :class:`MapDataset/IterDataset`: Built-in dataset types in PaddleNLP, extending :class:`paddle.io.Dataset` and :class:`paddle.io.IterableDataset` respectively. They include NLP-specific data processing features like :func:`map` and :func:`filter`, and help users easily create custom datasets. Refer to *** and :doc:`How to Create Custom Datasets <./dataset_self_defined>` for more information.

Data Processing Workflow Design
-------------------------------

The general data processing workflow in PaddleNLP is as follows:

1. Load dataset (built-in or custom dataset, returning **raw data**).
2. Define :func:`trans_func` (including tokenization, token-to-ID conversion, etc.) and apply it via the dataset's :func:`map` method to convert raw data into *features*.
3. Define **batchify** methods and :class:`BatchSampler` based on the processed data.
4. Define :class:`DataLoader` with :class:`BatchSampler` and :func:`batchify_fn`.

Below is the data processing flowchart for a BERT-based text classification task:

.. image:: ../imgs/data_preprocess_pipline.png

For detailed information about data processing, refer to :doc:`./data_preprocess`.