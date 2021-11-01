============
如何自定义数据集
============

通过使用PaddleNLP提供的 :func:`load_dataset` ， :class:`MapDataset` 和 :class:`IterDataset` 。任何人都可以方便的定义属于自己的数据集。

从本地文件创建数据集
-------------------

从本地文件创建数据集时，我们 **推荐** 根据本地数据集的格式给出读取function并传入 :func:`load_dataset` 中创建数据集。

以 `waybill_ie <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/waybill_ie>`__ 快递单信息抽取任务中的数据为例：

.. code-block::

    from paddlenlp.datasets import load_dataset

    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            # 跳过列名
            next(f)
            for line in f:
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield {'tokens': words, 'labels': labels}

    # data_path为read()方法的参数
    map_ds = load_dataset(read, data_path='train.txt',lazy=False) 
    iter_ds = load_dataset(read, data_path='train.txt',lazy=True) 

我们推荐将数据读取代码写成生成器(generator)的形式，这样可以更好的构建 :class:`MapDataset` 和 :class:`IterDataset` 两种数据集。同时我们也推荐将单条数据写成字典的格式，这样可以更方便的监测数据流向。

事实上，:class:`MapDataset` 在绝大多数时候都可以满足要求。一般只有在数据集过于庞大无法一次性加载进内存的时候我们才考虑使用 :class:`IterDataset` 。任何人都可以方便的定义属于自己的数据集。

.. note::

    需要注意的是，只有PaddleNLP内置的数据集具有将数据中的label自动转为id的功能（详细条件参见 :doc:`创建DatasetBuilder <../community/contribute_datasets/how_to_write_a_DatasetBuilder>`）。
    
    像上例中的自定义数据集需要在自定义的convert to feature方法中添加label转id的功能。

    自定义数据读取function中的参数可以直接以关键字参数的的方式传入 :func:`load_dataset` 中。而且对于自定义数据集，:attr:`lazy` 参数是 **必须** 传入的。

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
    
    ds = MyDataset(data_path)  # paddle.io.Dataset
    new_ds = MapDataset(ds)    # paddlenlp.datasets.MapDataset

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

.. note::

    需要注意，像上例中直接将 **生成器** 对象传入 :class:`IterDataset` 所生成的数据集。其数据只能迭代 **一次** 。

与常规的python对象一样，只要满足以上的条件，我们也可以使用同样的方法从第三方数据集创建PaddleNLP数据集。

例如HuggingFace Dataset：

.. code-block::

    from paddlenlp.datasets import MapDataset
    from datasets import load_dataset
    
    hf_train_ds = load_dataset('msra_ner', split='train')
    print(type(train_ds)) # <class 'datasets.arrow_dataset.Dataset'>

    train_ds = MapDataset(train_ds)
    print(type(train_ds)) # <class 'paddlenlp.datasets.dataset.MapDataset'>

    print(train_ds[2]) # {'id': '2', 
                       #  'ner_tags': [0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       #               0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                       #  'tokens': ['因', '有', '关', '日', '寇', '在', '京', '掠', '夺', '文', '物',
                       #             '详', '情', '，', '藏', '界', '较', '为', '重', '视', '，', '也', 
                       #             '是', '我', '们', '收', '藏', '北', '京', '史', '料', '中', '的',
                       #             '要', '件', '之', '一', '。']}

    hf_train_ds = load_dataset('cmrc2018', split='train')
    train_ds = MapDataset(hf_train_ds)
    print(train_ds[1818]) # {'answers': {'answer_start': [9], 'text': ['字仲可']}, 
                          #  'context': '徐珂（），原名昌，字仲可，浙江杭县（今属杭州市）人。光绪举人。
                          #              后任商务印书馆编辑。参加南社。1901年在上海担任了《外交报》、
                          #              《东方杂志》的编辑，1911年，接管《东方杂志》的“杂纂部”。与潘仕成、
                          #              王晋卿、王辑塘、冒鹤亭等友好。编有《清稗类钞》、《历代白话诗选》、
                          #              《古今词选集评》等。光绪十五年（1889年）举人。后任商务印书馆编辑。
                          #              参加南社。曾担任袁世凯在天津小站练兵时的幕僚，不久离去。', 
                          #  'id': 'TRAIN_113_QUERY_0', 
                          #  'question': '徐珂字什么？'}
    
    hf_train_ds = load_dataset('glue', 'sst2', split='train')
    train_ds = MapDataset(hf_train_ds)
    print(train_ds[0]) # {'idx': 0, 'label': 0, 'sentence': 'hide new secretions from the parental units '}

    hf_train_ds = load_dataset('ptb_text_only', split='train')
    train_ds = MapDataset(hf_train_ds)
    print(train_ds[1]) # {'sentence': 'pierre <unk> N years old will join the board as a nonexecutive director nov. N'}
