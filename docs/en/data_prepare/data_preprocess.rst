================
Data Processing
================

Datasets typically contain raw data, which requires specific processing and sampling to form batches. These are then fed through :class:`paddle.io.DataLoader` for training or inference. PaddleNLP provides corresponding functionalities to support each step in this pipeline.

Data Processing Based on Pretrained Models
------------------------------------------

When using pretrained models for NLP tasks, loading the corresponding Tokenizer is essential. PaddleNLP's :class:`PreTrainedTokenizer` implements basic data processing capabilities through its built-in :func:`__call__` method. All pretrained model Tokenizers in PaddleNLP inherit from :class:`PreTrainedTokenizer`. Here's an example using BertTokenizer:

.. code-block::

    from paddlenlp.transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # Single sentence conversion (single data instance)
    print(tokenizer(text='天气不错')) # {'input_ids': [101, 1921, 3698, 679, 7231, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0]}

    # Sentence pair conversion (single data instance)
    print(tokenizer(text='天气',text_pair='不错')) # {'input_ids': [101, 1921, 3698, 102, 679, 7231, 102], 'token_type_ids': [0, 0, 0, 0, 1, 1, 1]}

    # Single sentence conversion (multiple data instances)
    print(tokenizer(text=['天气','不错'])) # [{'input_ids': [101, 1921, 3698, 102], 'token_type_ids': [0, 0, 0, 0]}, 
                                          #  {'input_ids': [101, 679, 7231, 102], 'token_type_ids': [0, 0, 0, 0]}]

For additional parameters and functionalities of the :func:`__call__` method, please refer to PreTrainedTokenizer.

The :func:`map` method in PaddleNLP's built-in :class:`paddlenlp.datasets.MapDataset` supports applying a function to uniformly transform dataset entries. Below we demonstrate this using :obj:`LCQMC`
Take the data processing pipeline of `LCQMC` as an example:

.. code-block::

    from paddlenlp.transformers import BertTokenizer
    from paddlenlp.datasets import load_dataset

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_ds = load_dataset('lcqmc', splits='train')

    print(train_ds[0]) # {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生', 'label': 1}

As can be seen, :obj:`LCQMC` is a sentence pair matching task, i.e., a 2-classification task to determine whether two sentences are semantically similar. We need to process the text data with keys **query** and **title**. We will implement a data processing function based on :class:`PreTrainedTokenizer` and pass it to the dataset's :func:`map` method.
.. code-block::

    def convert_example(example, tokenizer):
        tokenized_example = tokenizer(
                                text=example['query'], 
                                text_pair=example['title'])
        # Add label for training
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

As can be seen, the text data in the dataset has been processed into *features* that the model can accept.

The :func:`map` method has an important parameter :attr:`batched`. When set to :obj:`True` (default is :obj:`False`), the data processing function :func:`trans_func`
.. code-block::

    def convert_examples(examples, tokenizer):
        queries = [example['query'] for example in examples]
        titles = [example['title'] for example in examples]
        tokenized_examples = tokenizer(text=queries, text_pair=titles, return_dict=False)

        # Add label for training
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

As can be seen, the results of the two implementations in this example are the same. However, in tasks such as machine reading comprehension and dialogue, where a single raw data instance may generate multiple *features* (refer to `run_squad.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_reading_comprehension/SQuAD/run_squad.py>`__), it is usually necessary to set the :attr:`batched` parameter to :obj:`True`.

:func:`map
The :func:`map` method also has a :attr:`num_workers` parameter. When it's greater than 0, multi-process data processing will be enabled, which can improve processing speed. However, please note that if the data processing function uses information related to **data index**, multi-processing may lead to incorrect results.

For other parameters of the :func:`map` method and other data processing methods of :class:`paddlenlp.datasets.MapDataset`, please refer to :doc:`dataset <../source/paddlenlp.datasets.dataset>`.

Batchify
-----------

PaddleNLP provides various built-in collate functions that work with :class:`paddle.io.BatchSampler` to simplify batch creation.

Let's continue with the :obj:`LCQMC` data processing example. As shown in the previous section, each processed data sample is a **dictionary** containing three keys: `input_ids`, `token_type_ids`, and `label`.

Among these, `input_ids` and `token_type_ids` need to be **padded** before being fed into the model, while `label` needs to be **stacked** before being passed to the loss function.

Therefore, we use PaddleNLP's built-in :func:`Dict`, :func:`Stack`, and :func:`Pad` functions to organize batch data. The final :func:`batchify_fn` is as follows:

.. code-block::

    from paddlenlp.data import Dict, Stack, Pad 

    # Use Dict to match Pad/Stack functions with dictionary keys
    train_batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        'label': Stack(dtype="int64")
    }): fn(samples)

Then we use :class:`paddle.io.BatchSampler` and :func:`batchify_fn` to build :class:`paddle.io.DataLoader`:

.. code-block::

    from paddle.io import DataLoader, BatchSampler

    train_batch_sampler = BatchSampler(train_ds, batch_size=2, shuffle=True)

    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=train_batchify_fn)

This completes the entire data preparation pipeline. For more batchify methods, please refer to :doc:`collate <../source/paddlenlp.data.collate>`.
.. note::

    - When performing **single-machine multi-GPU** training, replace :class:`BatchSampler` with :class:`DistributedBatchSampler`. For more information about :class:`paddle.io.BatchSampler`, please refer to `BatchSampler <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dataloader/batch_sampler/BatchSampler_cn.html>`_.

    - For more complex batching functionalities such as in-batch sorting or token-based batching, you can use PaddleNLP's built-in :class:`SamplerHelper`. Example usage can be found in `reader.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_translation/transformer/reader.py>`__.

Data Processing Based on Non-Pretrained Models
-----------------------------------------------

When using non-pretrained models for NLP tasks, we can leverage PaddleNLP's built-in :class:`JiebaTokenizer` and :class:`Vocab` for data processing. The overall workflow is similar to using pretrained models. We demonstrate this using the Chinese sentiment analysis dataset :obj:`ChnSentiCorp`:

.. code-block::

    from paddlenlp.data import JiebaTokenizer, Vocab
    from paddlenlp.datasets import load_dataset

    train_ds = load_dataset('chnsenticorp', splits='train')
    
    print(train_ds[0]) # {'text': '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。
                       #  酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 
                       #  服务吗，一般', 'label': 1}

    # Build Vocab from local vocab file
    vocab = Vocab.load_vocabulary('./senta_word_dict.txt', unk_token='[UNK]', pad_token='[PAD]')

    # Initialize JiebaTokenizer with Vocab
    tokenizer = JiebaTokenizer(vocab)

.. note::

    - In addition to initializing from a local vocabulary file, :class:`Vocab` provides multiple initialization methods, including creating from :class:`dictionary` and datasets. For details, refer to the Vocab documentation.
    - Besides the built-in :class:`JiebaTokenizer`, users can implement custom tokenization approaches or use third-party libraries, then convert tokens to indices via :func:`Vocab.to_indices`.
The `convert_example` method converts tokens to ids.

Following similar data processing flow as pre-trained model based approaches, we write data processing functions and pass them to the :func:`map` method:

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

We can observe that the raw data has been processed into *features*. However, here we notice that a single data entry is not a **dictionary** but a **tuple**. Therefore, our :func:`batchify_fn` needs to be adjusted accordingly.
.. code-block::

    from paddlenlp.data import Tuple, Stack, Pad 

    # Use Tuple function to align Pad, Stack etc. functions with key-value pairs in data
    train_batchify_fn = lambda samples, fn=Tuple((
        Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    )): fn(samples)

It can be observed that the :func:`Dict` function maps the key-values in individual data instances to corresponding :func:`Pad` etc. functions, which is suitable when each data instance is a dictionary. Whereas :func:`Tuple` aligns different components through indices in individual data instances.

Therefore, special attention should be paid to the correspondence between the :func:`convert_example` method and the :func:`batchify_fn` method.

The subsequent workflow remains consistent with the data processing approach based on pretrained models.