==============
创建 :class:`DatasetBuilder`
==============

数据集的贡献通过定义一个 :class:`DatasetBuilder` 的子类来实现。一个合格的 :class:`DatasetBuilder` 需要遵循一些协议和规范。

下面我们以 :obj:`LCQMC` 为例了解一下 :class:`DatasetBuilder` 通常需要包含哪些方法和参数。

成员变量
---------------

.. code-block::

    from paddle.dataset.common import md5file
    from paddle.utils.download import get_path_from_url
    from paddlenlp.utils.env import DATA_HOME

    class LCQMC(DatasetBuilder):
        """
        LCQMC:A Large-scale Chinese Question Matching Corpus
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
    
首先贡献的数据集需要继承 :class:`paddlenlp.datasets.DatasetBuilder` 类，类名格式为camel case。之后应该添加一段注释，简要说明数据集的来源等信息。之后需定义以下成员变量：

- :attr:`lazy` ：数据集的默认类型。:obj:`False` 对应 :class:`MapDataset` ，:obj:`True` 对应 :class:`IterDataset` 。
- :attr:`URL` ：数据集压缩包下载地址，需提供有效并稳定的下载链接。如果数据集不是压缩包，可以不再这里提供。
- :attr:`MD5` ：数据集压缩包的md5值，用于文件校验，如果数据集文件不是压缩包，可以不再这里提供。
- :attr:`META_INFO` ：数据集split信息格式。
- :attr:`SPLITS` ：数据集的split信息，包含数据集解压后的不同文件的具体位置，文件名，md5值等，如果数据集不是压缩包则通常在这里提供下载地址，还可以包含诸如不同文件对应的文件读取参数等信息。

除此之外，不同的数据集可能还需要诸如 :attr:`VOCAB_INFO` 等其他成员变量（参见 `iwslt15.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/iwslt15.py>`__ ）。或者成员变量会有其他格式。贡献者可以根据实际情况自行调整。

.. note::

    - 如果贡献的数据集没有子数据集，那么 :class:`DatasetBuilder` **必须包含** :attr:`SPLITS` 成员变量，且该变量必须是一个字典，字典的key是该数据集包含的splits。
    - 如果贡献的数据集有子数据集，那么 :class:`DatasetBuilder` **必须包含** :attr:`BUILDER_CONFIGS` 成员变量，且该变量必须是一个字典，字典的key是该数据集包含的子数据集的 :attr:`name` 。字典的value是包含该数据集的子数据集split信息的字典，key值必须是 `splits` 。具体格式（参见 `glue.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/glue.py>`__ ）

:func:`_get_data` 方法
-----------------------

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

:func:`_get_data` 方法根据传入的 :attr:`mode` 和数据集的split信息定位到具体数据集文件。首先进行md5值校验本地文件，若校验失败则调用 :func:`paddle.utils.download.get_path_from_url` 方法下载并校验数据集文件，最后返回数据集文件的本地地址。

:func:`_read` 方法
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

:func:`_read` 方法根据传入的文件地址读取数据。该方法必须是一个生成器，以确保 :class:`DatasetBuilder` 可以构造 :class:`MapDataset` 和  :class:`IterDataset` 两种数据集。 
当不同split对应的数据文件读取方式不同时，该方法还需要支持 :attr:`split` 参数，并支持不同split下的读取方式。

.. note::

    - 该方法提供的每条example都应是一个 :class:`Dictionary` 对象。
    - :class:`DatasetBuilder` 在生成Dataset时提供了将class label转换为id的功能。如果用户需要此功能，需要将example中label对应的key设置为 **"label"** 或 **"labels"** ，并在类中正确添加 :func:`get_labels` 方法。

:func:`get_labels` 方法
-----------------------

.. code-block::

    def get_labels(self):
        """
        Return labels of the LCQMC object.
        """
        return ["0", "1"]

:func:`get_labels` 方法返回一个由该数据集中所有label组成的list。用于将数据集中的class label转换为id，并且这个list之后会作为实例变量传给生成的数据集。

:func:`get_vocab` 方法
-----------------------

如果数据集提供词典文件，则需要加入 :func:`get_vocab` 方法和 :attr:`VOCAB_INFO` 变量。

该方法会根据 :attr:`VOCAB_INFO` 变量返回一个包含数据集词典信息的 :class:`Dictionary` 对象并作为实例变量传给生成的数据集。用于在训练过程中初始化 :class:`paddlenlp.data.Vocab` 对象。
该方法的写法请参考 `iwslt15.py  <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/iwslt15.py>`__ 。

.. note::

    - 贡献数据集时 :func:`get_labels` 和 :func:`get_vocab` 方法是可选的，视具体数据集内容而定。 :func:`_read` 和 :func:`_get_data` 方法是 **必须包含** 的。
    - 如果您不希望在数据获取过程中进行md5值校验，可以不用给出相关成员变量和校验代码。

