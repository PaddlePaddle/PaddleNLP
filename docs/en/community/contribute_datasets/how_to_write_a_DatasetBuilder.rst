==============
Creating :class:`DatasetBuilder`
==============

Dataset contributions are achieved by defining a subclass of :class:`DatasetBuilder`. A qualified :class:`DatasetBuilder` needs to follow certain protocols and specifications.

Let's take :obj:`LCQMC` as an example to understand the typical methods and parameters required in a :class:`DatasetBuilder`.

Member Variables
---------------

.. code-block::

    from paddle.dataset.common import md5file
    from paddle.utils.download import get_path_from_url
    from paddlenlp.utils.env import DATA_HOME

    class LCQMC(DatasetBuilder):
        """
        LCQMC: A Large-scale Chinese Question Matching Corpus
        More information please refer to `https://www.aclweb.org/anthology/C18-1166/`

        """
        lazy = False
        URL = "https://bj.bcebos.com/paddlehub-dataset/lcqmc.tar.gz"
        MD5 = "62a7ba36f786a82ae59bbde0b0a9af0c"
        META_INFO = collections.namedtuple('META_INFO', ('file', 'md5'))
        SPLITS = {
            'train': META_INFO(
                os.path.join('lcqmc', 'train.tsv'),
                '2193c022439b038ac12c0ae918b211a1'),
            'dev': META_INFO(
                os.path.join('lcqmc', 'dev.tsv'),
                'c5dcba253cb4105d914964fd8b3c0e94'),
            'test': META_INFO(
                os.path.join('lcqmc', 'test.tsv'),
                '8f4b71e15e67696cc9e112a459ec42bd'),
        }

First, the contributed dataset needs to inherit from the :class:`paddlenlp.datasets.DatasetBuilder` class, with the class name in camel case format. Then you should add a docstring briefly describing the dataset's origin and other information. The following member variables need to be defined:

- :attr:`lazy`: The default dataset type. :obj:`False` corresponds to :class:`MapDataset`, :obj:`True` corresponds to :class:`IterDataset`.
- :attr:`URL
- :attr:`URL`: The download URL for the dataset archive, must provide a valid and stable link. If the dataset is not archived, this may be omitted.
- :attr:`MD5`: MD5 checksum of the dataset archive for file validation. If the dataset is not archived, this may be omitted.
- :attr:`META_INFO`: The format of dataset split information.
- :attr:`SPLITS`: Split information of the dataset, containing file locations, filenames, MD5 values, etc. after decompression. For non-archived datasets, download URLs are typically provided here. May also include parameters like file reading configurations.

Additionally, some datasets may require other member variables like :attr:`VOCAB_INFO` (refer to `iwslt15.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/iwslt15.py>`__). Member variables may have different formats. Contributors can adjust accordingly based on actual requirements.

.. note::

    - If the contributed dataset has no subsets, the :class:`DatasetBuilder` **must include** the :attr:`SPLITS` member variable, which must be a dictionary with keys corresponding to the dataset's splits.
    - If the contributed dataset contains subsets, the :class:`DatasetBuilder` **must include** the :attr:`BUILDER_CONFIGS` member variable. This must be a dictionary with keys corresponding to the subset's :attr:`name`. The values should be dictionaries containing split information for the subset, with keys being `splits`. For specific formats, refer to `glue.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/glue.py>`__.

:func:`_get_data` Method
------------------------

.. code-block::

    def _get_data(self, mode, **kwargs):
        ''' Check and download Dataset '''
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

The :func:`_get_data` method locates the specific dataset file based on the input :attr:`mode` and split information. It first performs MD5 checksum validation on local files. If validation fails, it calls :func:`
`paddle.utils.download.get_path_from_url` method downloads and verifies dataset files, finally returns the local path of dataset file.

:func:`_read` method
-----------------------

.. code-block::

    def _read(self, filename):
        """Reads data."""
        with open(filename, 'r', encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    query, title, label = data
                    yield {"query": query, "title": title, "label": label}

The :func:`_read` method reads data from given file path. This method must be a generator to ensure :class:`DatasetBuilder` can construct both :class:`MapDataset` and :class:`IterDataset`. When different splits require distinct data reading approaches, this method should additionally support :attr:`split` parameter and handle different split configurations.

.. note::

    - Each example provided by this method should be a :class:`Dictionary` object.
    - :class:`DatasetBuilder` provides label-to-id conversion during Dataset generation. To use this feature, users must set the label key in examples as **"label"** or **"labels"**, and properly implement :func:`get_labels` method in the class.

:func:`get_labels` method
-----------------------

.. code-block::

    def get_labels(self):
        """
        Return labels of the LCQMC object.
        """
        return ["0", "1"]

The :func:`get_labels` method returns a list containing all labels in the dataset. This is used to convert class labels to ids, and this list will be passed as an instance variable to the generated dataset.

:func:`get_vocab` method
-----------------------

If the dataset provides vocabulary files, the :func:`get_vocab` method and :attr:`VOCAB_INFO` variable need to be added.

This method returns a :class:`Dictionary` object containing dataset vocabulary information based on :attr:`VOCAB_INFO`, which is passed as an instance variable to the generated dataset. Used to initialize :class:`paddlenlp.data.Vocab` object during training. Refer to official implementation for method details.
iwslt15.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/iwslt15.py>`__

.. note::

    - When contributing a dataset, the :func:`get_labels` and :func:`get_vocab` methods are optional, depending on the specific dataset content. The :func:`_read` and :func:`_get_data` methods are **required**.
    - If you do not wish to perform an md5 check during data retrieval, you may omit the relevant member variables and validation code.