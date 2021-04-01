============
如何自定义数据集
============

通过使用PaddleNLP提供的 :class:`MapDataset` 和 :class:`IterDataset` 。任何人都可以方便的定义属于自己的数据集。

从本地文件创建数据集
-------------------

从本地文件创建数据集时，我们 **推荐** 根据本地数据集的格式给出读取function并传入 :class:`MapDataset` 或 :class:`IterDataset` 中创建数据集。

以 :obj:`waybill_ie` 快递单信息抽取任务中的数据为例：

.. code-block::

    from paddlenlp.datasets import MapDataset, IterDataset

    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            # 跳过列名
            next(f)
            for line in f():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield {'tokens': words, 'labels': labels}
    
    # MapDataset需要__getitem__()方法和__len__()方法，需要将生成器转为list
    map_ds = MapDataset(list(read(data_path))) 

    # IterDataset需要__iter__()方法，生成器满足需求
    iter_ds = IterDataset(read(data_path))

我们推荐将数据读取代码写成生成器(generator)的形式，这样可以更好的构建 :class:`MapDataset` 和 :class:`IterDataset` 两种数据集。同时我们也推荐将单条数据写成字典的格式，这样可以更方便的监测数据流向。

.. note::

    需要注意的是，只有从 :class:`DatasetBuilder` 初始化的数据集具有将数据中的label自动转为id的功能（详细条件参见 :doc:`如何贡献数据集 <../community/contribute_dataset>`）。
    
    像上例中的自定义数据集需要在自定义的convert to feature方法中添加label转id的功能。

从 :class:`paddle.io.Dataset/IterableDataset` 创建数据集 
-------------------

虽然PaddlePddle内置的 :class:`Dataset` 和 :class:`IterableDataset` 是可以直接接入 :class:`DataLoader` 用于模型训练的，但有时我们希望更方便的使用一些数据处理（例如convert to feature, 数据清洗，数据增强等）。而PaddleNLP内置的 :class:`MapDataset` 和 :class:`IterDataset` 正好提供了能实现以上功能的API。

所以如果您习惯使用 :class:`paddle.io.Dataset/IterableDataset` 创建数据集的话。只需要在原来的数据集上套上一层 :class:`MapDataset` 或 :class:`IterDataset` 就可以把原来的数据集对象转换成PaddleNLP的数据集。

下面举一个简单的小例子。:class:`IterDataset` 的用法基本相同。

.. code-block::

    from paddle.io import Dataset
    from paddlenlp.datasets import MapDataset

    class MyDataset(Dataset):
        def __init__(self, path):

            def load_data_from_source(path):
                ...
                ...
                return data

            self.data = load_data_from_source(path)

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return len(self.data)
    
    ds = MyDataset(data_path)      # paddle.io.Dataset
    new_ds = MapDataset(MyDataset) # paddlenlp.datasets.MapDataset

从其他python对象创建数据集
-------------------

理论上，我们可以使用任何包含 :func:`__getitem__` 方法和 :func:`__len__` 方法的python对象创建 :class:`MapDataset`。包括 :class:`List` ，:class:`Tuple` ，:class:`DataFrame` 等。只要将符合条件的python对象作为初始化参数传入 :class:`MapDataset` 即可完成创建。

.. code-block::

    from paddlenlp.datasets import MapDataset

    data_source_1 = [1,2,3,4,5]
    data_source_2 = ('a', 'b', 'c', 'd')

    list_ds = MapDataset(data_source_1)
    tuple_ds = MapDataset(data_source_2)

    print(list_ds[0])  # 1
    print(tuple_ds[0]) # a

同样的，我们也可以使用包含 :func:`__iter__` 方法的python对象创建 :class:`IterDataset` 。例如 :class:`List`， :class:`Generator` 等。创建方法与 :class:`MapDataset` 相同。

.. code-block::

    from paddlenlp.datasets import IterDataset

    data_source_1 = ['a', 'b', 'c', 'd']
    data_source_2 = (i for i in range(5))

    list_ds = IterDataset(data_source_1)
    gen_ds = IterDataset(data_source_2)

    print([data for data in list_ds]) # ['a', 'b', 'c', 'd']
    print([data for data in gen_ds])  # [0, 1, 2, 3, 4]

与常规的python对象一样，只要满足以上的条件，我们也可以使用同样的方法从第三方数据集创建PaddleNLP数据集。

例如HuggingFace Dataset：

.. code-block::

    from paddlenlp.datasets import MapDataset
    from datasets import load_dataset
    
    hg_ds = load_dataset('msra_ner', split='train')
    print(tpye(hg_ds)) # <class 'datasets.arrow_dataset.Dataset'>

    ppnlp_ds = MapDataset(hg_ds)
    print(tpye(ppnlp_ds)) # <class 'paddlenlp.datasets.dataset.MapDataset'>

    print(ppnlp_ds[0]) # {'id': '0', 
                       #  'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                       #               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                       #               0, 0, 0, 0, 0, 0, 0, 0], 
                       #  'tokens': ['当', '希', '望', '工', '程', '救', '助', '的', '百', '万', '儿', 
                       #             '童', '成', '长', '起', '来', '，', '科', '教', '兴', '国', '蔚', 
                       #             '然', '成', '风', '时', '，', '今', '天', '有', '收', '藏', '价', 
                       #             '值', '的', '书', '你', '没', '买', '，', '明', '日', '就', '叫', 
                       #             '你', '悔', '不', '当', '初', '！']}

