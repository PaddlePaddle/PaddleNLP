========================
Sharing Your Dataset
========================

In addition to using built-in datasets in PaddleNLP, we also encourage users to contribute their own datasets to PaddleNLP.

Below we detail the workflow for contributing a dataset:

Environment Setup
---------------

#. Writing and testing PaddleNLP code requires Python 3.6+ and the latest version of PaddlePaddle. Please ensure these dependencies are properly installed.
#. Click the Fork button on PaddleNLP's GitHub page to create a copy of the PaddleNLP repo under your GitHub account.
#. Clone your forked repo locally and add the official repo as a remote.

   .. code-block::

       git clone https://github.com/USERNAME/PaddleNLP
       cd PaddleNLP
       git remote add upstream https://github.com/PaddlePaddle/PaddleNLP.git

#. Install the pre-commit hook, which helps format source code and automatically check for issues before submission. PRs that fail hook checks **cannot** be merged into PaddleNLP.

   .. code-block::

       pip install pre-commit
       pre-commit install

Adding a :class:`DatasetBuilder`
----------------------------------

#. Create a new local branch, typically branched from develop.

   .. code-block::

       git checkout -b my-new-dataset

#. Navigate to the `PaddleNLP/paddlenlp/datasets/` directory in your local repo - all dataset code is stored here.

   .. code-block::

       cd paddlenlp/datasets

#. Determine a `name` for your dataset (e.g., `squad`, `chnsenticorp`). This `name` will be used when loading your dataset.
    
   .. note::

       - To facilitate usage, ensure the `name` is **concise and semantically meaningful**.
       - The dataset `name` should follow snake_case convention.

#. Create a Python file named after the dataset `name` (e.g., `squad.py`) in this directory. Implement your dataset's :class:`DatasetBuilder` class in this file.

   Refer to the tutorial :doc:`How to Write a DatasetBuilder <./how_to_write_a_DatasetBuilder>` for detailed guidelines on implementing :class:`DatasetBuilder`.

   We also recommend referencing existing :class:`DatasetBuilder` implementations. The following examples may be helpful:

   -  `squad.py`
`iwslt15.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/iwslt15.py>`__ Translation dataset containing vocabulary files.
   -  `glue.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/glue.py>`__ GLUE dataset containing multiple sub-datasets, file format is TSV.
   -  `squad.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/squad.py>`__ Reading comprehension dataset, file format is JSON.
   -  `imdb.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/imdb.py>`__ IMDB dataset, each split contains multiple files.
   -  `ptb.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/ptb.py>`__ Corpus dataset.
   -  `msra_ner.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/msra_ner.py>`__ Sequence labeling dataset.

#. After development, you can use :attr:`load_dataset` to test whether the splits in your created dataset can be correctly identified. You can also use :attr:`
Check if the dataset reading format meets your expectations:

.. code-block::

    from paddlenlp.datasets import load_dataset

    ds = load_dataset('your_dataset_name', splits='your_split')
    print(ds[0])

Submit Your Work
---------------

#. When you confirm the dataset code is ready, commit your changes locally:
   
   .. code-block::
       
       git add PaddleNLP/paddlenlp/datasets/your_dataset_name.py
       git commit

#. Before submitting, it's recommended to fetch the latest upstream code and update current branch:

   .. code-block::
       
       git fetch upstream
       git pull upstream develop

#. Push local changes to GitHub and submit a Pull Request to PaddleNLP:

   .. code-block::
       
       git push origin my-new-dataset

The above is the complete process for contributing datasets to PaddleNLP. We will review your PR promptly and provide feedback if needed. If everything looks good, we will merge it into the PaddleNLP repo, making your dataset available to others.

If you have any questions about contributing datasets, feel free to join our official QQ technical group: 973379845. We'll address your inquiries promptly.