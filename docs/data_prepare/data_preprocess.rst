================
数据处理
================

Dataset中通常为原始数据，需要经过一定的数据处理并进行采样组batch，而后通过 :class:`paddle.io.DataLoader` 为训练或预测使用，PaddleNLP中为其中各环节提供了相应的功能支持。

基于预训练模型的数据处理
------------------------

在使用预训练模型做NLP任务时，需要加载对应的Tokenizer，PaddleNLP在 :class:`PreTrainedTokenizer` 中内置的 :func:`__call__` 方法可以实现基础的数据处理功能。PaddleNLP内置的所有预训练模型的Tokenizer都继承自 :class:`PreTrainedTokenizer` ，下面以BertTokenizer举例说明：

.. code-block::

    from paddlenlp.transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 单句转换（单条数据）
    print(tokenizer(text='天气不错')) # {'input_ids': [101, 1921, 3698, 679, 7231, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0]}

    # 句对转换（单条数据）
    print(tokenizer(text='天气',text_pair='不错')) # {'input_ids': [101, 1921, 3698, 102, 679, 7231, 102], 'token_type_ids': [0, 0, 0, 0, 1, 1, 1]}

    # 单句转换（多条数据）
    print(tokenizer(text=['天气','不错'])) # [{'input_ids': [101, 1921, 3698, 102], 'token_type_ids': [0, 0, 0, 0]}, 
                                          #  {'input_ids': [101, 679, 7231, 102], 'token_type_ids': [0, 0, 0, 0]}]
 
关于 :func:`__call__` 方法的其他参数和功能，请查阅PreTrainedTokenizer。

paddlenlp内置的 :class:`paddlenlp.datasets.MapDataset` 的 :func:`map` 方法支持传入一个函数，对数据集内的数据进行统一转换。下面我们以 :obj:`LCQMC` 的数据处理流程为例：

.. code-block::

    from paddlenlp.transformers import BertTokenizer
    from paddlenlp.datasets import load_dataset

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_ds = load_dataset('lcqmc', splits='train')

    print(train_ds[0]) # {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生', 'label': 1}

可以看到， :obj:`LCQMC` 是一个句对匹配任务，即判断两个句子的意思是否相似的2分类任务。我们需要处理的是key为 **query** 和 **title** 的文本数据，我们编写基于 :class:`PreTrainedTokenizer` 的数据处理函数并传入数据集的 :func:`map` 方法。

.. code-block::

    def convert_example(example, tokenizer):
        tokenized_example = tokenizer(
                                text=example['query'], 
                                text_pair=example['title'])
        # 加上label用于训练
        tokenized_example['label'] = [example['label']]
        return tokenized_example
    
    from functools import partial

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer)
    
    train_ds.map(trans_func)
    print(train_ds[0]) # {'input_ids': [101, 1599, 3614, 2802, 5074, 4413, 4638, 4511, 4495, 
                       #                1599, 3614, 784, 720, 3416, 4638, 1957, 4495, 102, 
                       #                4263, 2802, 5074, 4413, 4638, 4511, 4495, 1599, 3614, 
                       #                784, 720, 3416, 4638, 1957, 4495, 102], 
                       #  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                       #                     0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       #  'label': [1]}

可以看到，数据集中的文本数据已经被处理成了模型可以接受的 *feature* 。

:func:`map` 方法有一个重要的参数 :attr:`batched`，当设置为 :obj:`True` 时（默认为 :obj:`False` ），数据处理函数 :func:`trans_func` 的输入不再是单条数据，而是数据集的所有数据：

.. code-block::

    def convert_examples(examples, tokenizer):
        querys = [example['query'] for example in examples]
        titles = [example['title'] for example in examples]
        tokenized_examples = tokenizer(text=querys, text_pair=titles)

        # 加上label用于训练
        for idx in range(len(tokenized_examples)):
            tokenized_examples[idx]['label'] = [examples[idx]['label']]
        
        return tokenized_examples
    
    from functools import partial

    trans_func = partial(convert_examples, tokenizer=tokenizer)
    
    train_ds.map(trans_func, batched=True)
    print(train_ds[0]) # {'input_ids': [101, 1599, 3614, 2802, 5074, 4413, 4638, 4511, 4495, 
                       #                1599, 3614, 784, 720, 3416, 4638, 1957, 4495, 102, 
                       #                4263, 2802, 5074, 4413, 4638, 4511, 4495, 1599, 3614, 
                       #                784, 720, 3416, 4638, 1957, 4495, 102], 
                       #  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                       #                     0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       #  'label': [1]}

可以看到，在本例中两种实现的结果是相同的。但是在诸如阅读理解，对话等任务中，一条原始数据可能会产生多个 *feature* 的情况（参见 `run_squad.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_reading_comprehension/SQuAD/run_squad.py>`__ ）通常需要将 :attr:`batched` 参数设置为 :obj:`True` 。 

:func:`map` 方法还有一个 :attr:`num_workers` 参数，当其大于0时进行多进程数据处理，可以提高处理速度。但是需要注意如果在数据处理的函数中用到了 **数据index** 的相关信息，多进程处理可能会导致错误的结果。

关于 :func:`map` 方法的其他参数和 :class:`paddlenlp.datasets.MapDataset` 的其他数据处理方法，请查阅 :doc:`dataset <../source/paddlenlp.datasets.dataset>` 。

Batchify
-----------

PaddleNLP内置了多种collate function，配合 :class:`paddle.io.BatchSampler` 可以协助用户简单的完成组batch的操作。

我们继续以 :obj:`LCQMC` 的数据处理流程为例。从上一节最后可以看到，处理后的单条数据是一个 **字典** ，包含 `input_ids` ， `token_type_ids` 和 `label` 三个key。

其中 `input_ids` 和 `token_type_ids` 是需要进行 **padding** 操作后输入模型的，而 `label` 是需要 **stack** 之后传入loss function的。

因此，我们使用PaddleNLP内置的 :func:`Dict` ，:func:`Stack` 和 :func:`Pad` 函数整理batch中的数据。最终的 :func:`batchify_fn` 如下：

.. code-block::

    from paddlenlp.data import Dict, Stack, Pad 

    # 使用Dict函数将Pad，Stack等函数与数据中的键值相匹配
    train_batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        'label': Stack(dtype="int64")
    }): fn(samples)

之后使用 :class:`paddle.io.BatchSampler` 和 :func:`batchify_fn` 构建 :class:`paddle.io.DataLoader` ：

.. code-block::

    from paddle.io import DataLoader, BatchSampler

    train_batch_sampler = BatchSampler(train_ds, batch_size=2, shuffle=True)

    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=train_batchify_fn)

到此，一个完整的数据准备流程就完成了。关于更多batchify方法，请查阅 :doc:`collate <../source/paddlenlp.data.collate>`。

.. note::

    - 当需要进行 **单机多卡** 训练时，需要将 :class:`BatchSampler` 更换为 :class:`DistributedBatchSampler` 。更多有关 :class:`paddle.io.BatchSampler` 的信息，请查阅 `BatchSampler <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dataloader/batch_sampler/BatchSampler_cn.html>`_。

    - 当需要诸如batch内排序，按token组batch等更复杂的组batch功能时。可以使用PaddleNLP内置的 :class:`SamplerHelper` 。相关用例请参考 `reader.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_translation/transformer/reader.py>`__。

基于非预训练模型的数据处理
-------------------------

在使用非预训练模型做NLP任务时，我们可以借助PaddleNLP内置的 :class:`JiebaTokenizer` 和 :class:`Vocab` 完成数据处理的相关功能，整体流程与使用预训练模型基本相似。我们以中文情感分析 :obj:`ChnSentiCorp` 数据集为例：

.. code-block::

    from paddlenlp.data import JiebaTokenizer, Vocab
    from paddlenlp.datasets import load_dataset

    train_ds = load_dataset('chnsenticorp', splits='train')
    
    print(train_ds[0]) # {'text': '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。
                       #  酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 
                       #  服务吗，一般', 'label': 1}

    # 从本地词典文件构建Vocab
    vocab = Vocab.load_vocabulary('./senta_word_dict.txt', unk_token='[UNK]', pad_token='[PAD]')

    # 使用Vocab初始化JiebaTokenizer
    tokenizer = JiebaTokenizer(vocab)

.. note::

    - :class:`Vocab` 除了可以从本地词典文件初始化之外，还提供多种初始化方法，包括从 :class:`dictionary` 创建、从数据集创建等。详情请查阅Vocab。
    - 除了使用内置的 :class:`JiebaTokenizer` 外，用户还可以使用任何自定义的方式或第三方库进行分词，之后使用 :func:`Vocab.to_indices` 方法将token转为id。

之后与基于预训练模型的数据处理流程相似，编写数据处理函数并传入 :func:`map` 方法：

.. code-block::

    def convert_example(example, tokenizer):
        input_ids = tokenizer.encode(example["text"])
        valid_length = [len(input_ids)]
        label = [example["label"]]
        return input_ids, valid_length, label

    trans_fn = partial(convert_example, tokenizer=tokenizer)
    train_ds.map(trans_fn)

    print(train_ds[0]) # ([417329, 128448, 140437, 173188, 118001, 213058, 595790, 1106339, 940533, 947744, 169206,
                       #   421258, 908089, 982848, 1106339, 35413, 1055821, 4782, 377145, 4782, 238721, 4782, 642263,
                       #   4782, 891683, 767091, 4783, 672971, 774154, 1250380, 1106339, 340363, 146708, 1081122, 
                       #   4783, 1, 943329, 1008467, 319839, 173188, 909097, 1106339, 1010656, 261577, 1110707, 
                       #   1106339, 770761, 597037, 1068649, 850865, 4783, 1, 993848, 173188, 689611, 1057229, 1239193, 
                       #   173188, 1106339, 146708, 427691, 4783, 1, 724601, 179582, 1106339, 1250380], 
                       #  [67], 
                       #  [1])


可以看到，原始数据已经被处理成了 *feature* 。但是这里我们发现单条数据并不是一个 **字典** ，而是 **元组** 。所以我们的 :func:`batchify_fn` 也要相应的做一些调整：

.. code-block::

    from paddlenlp.data import Tuple, Stack, Pad 

    # 使用Tuple函数将Pad，Stack等函数与数据中的键值相匹配
    train_batchify_fn = lambda samples, fn=Tuple((
        Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    )): fn(samples)

可以看到，:func:`Dict` 函数是将单条数据中的键值与 :func:`Pad` 等函数进行对应，适用于单条数据是字典的情况。而 :func:`Tuple` 是通过单条数据中不同部分的index进行对应的。

所以需要 **注意** 的是 :func:`convert_example` 方法和 :func:`batchify_fn` 方法的匹配。

之后的流程与基于预训练模型的数据处理相同。
