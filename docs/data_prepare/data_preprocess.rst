================
数据处理
================

为方便用户进行数据处理。PaddleNLP的各个模块都封装了支持NLP任务的数据处理功能。

基于预训练模型的数据处理
------------------------

在使用预训练模型做NLP任务时，需要加载对应的Tokenizer，PaddleNLP在 :class:`PreTrainedTokenizer` 中内置的 :func:`__call__` 方法可以实现基础的数据处理功能。下面以BertTokenizer举例说明：

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

关于 :func:`__call__` 方法的其他参数和功能，请移步PreTrainedTokenizer。

paddlenlp内置的 :class:`paddlenlp.datasets.MapDataset` 的 :func:`map` 方法支持传入一个函数，对数据集内的数据进行统一转换。下面我们以 :obj:`LCQMC` 的数据处理流程为例：

.. code-block::

    from paddlenlp.transformers import BertTokenizer
    from paddlenlp.datasets import load_dataset

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_ds = load_dataset('lcqmc', splits='train')

    print(train_ds[0]) # {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生', 'label': 1}

可以看到，我们需要处理的是key为 **query** 和 **title** 的文本数据，我们编写基于 :class:`PreTrainedTokenizer` 的数据处理函数并传入数据集的 :func:`map` 方法。

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

可以看到，数据集中的文本数据已经被处理成了模型可以接受的 *feature* 。:func:`map` 方法有一个重要的参数 :attr:`batched`，当设置为 :obj:`True` 时（默认为 :obj:`False` ），数据处理函数 :func:`trans_func` 的输入不再是单条数据，而是数据集的所有数据：

.. code-block::

    def convert_examples(examples, tokenizer):
        querys = [example['query'] for example in examples]
        titles = [example['title'] for example in examples]
        tokenized_examples = tokenizer(text=querys, text_pair=titles])

        # 加上label用于训练
        for idx in range(len(tokenized_examples)):
            tokenized_examples[idx]['label'] = [examples[idx]['label']]
        
        return tokenized_examples
    
    from functools import partial

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer)
    
    train_ds.map(trans_func, batched=True)
    print(train_ds[0]) # {'input_ids': [101, 1599, 3614, 2802, 5074, 4413, 4638, 4511, 4495, 
                       #                1599, 3614, 784, 720, 3416, 4638, 1957, 4495, 102, 
                       #                4263, 2802, 5074, 4413, 4638, 4511, 4495, 1599, 3614, 
                       #                784, 720, 3416, 4638, 1957, 4495, 102], 
                       #  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                       #                     0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       #  'label': [1]}

可以看到，在本例中两种实现的结果是相同的。但是在诸如阅读理解，对话等任务中，一条原始数据可能会产生多个 *feature* 的情况（参见 `run_squad.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_reading_comprehension/SQuAD/run_squad.py>`__ ）通常需要将 :attr:`batched` 参数设置为 :obj:`True` 。 

关于 :func:`map` 方法的其他参数和 :class:`paddlenlp.datasets.MapDataset` 的其他数据处理方法，请移步MapDataset。

Batchify
-----------

PaddleNLP内置了多种collate function，可以协助用户简单的完成组batch的操作。

我们继续以 :obj:`LCQMC` 的数据处理流程为例。从上一节最后可以看到，处理后的单条数据是一个 **字典** ，包含 `input_ids` ， `token_type_ids` 和 `label` 三个key。

其中 `input_ids` 和 `token_type_ids` 是需要进行 **padding** 操作后输入模型的，而 `label` 是需要 **stack** 之后传入loss function的。因此最终的 :func:`batchify_fn` 如下：

.. code-block::

    from paddlenlp.data import Dict, Stack, Pad 

    train_batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        'labels': Stack(dtype="int64")
    }): fn(samples)

之后使用处理好的数据集和 :func:`batchify_fn` 构建 :class:`paddle.io.DataLoader` ：

.. code-block::

    from paddle.io import DataLoader

    train_data_loader = DataLoader(dataset=train_ds, batch_size=2, collate_fn=train_batchify_fn)

到此，一个完整的数据准备流程就完成了。关于更多batchify方法，请移步collate。