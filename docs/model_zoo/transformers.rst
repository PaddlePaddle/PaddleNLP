PaddleNLP Transformer API
====================================

随着深度学习的发展，NLP领域涌现了一大批高质量的Transformer类预训练模型，多次刷新各种NLP任务SOTA（State of the Art）。
PaddleNLP为用户提供了常用的 ``BERT``、``ERNIE``、``ALBERT``、``RoBERTa``、``XLNet`` 等经典结构预训练模型，
让开发者能够方便快捷应用各类Transformer预训练模型及其下游任务。

------------------------------------
Transformer预训练模型汇总
------------------------------------

下表汇总了介绍了目前PaddleNLP支持的各类预训练模型以及对应预训练权重。我们目前提供了 **62** 种预训练的参数权重供用户使用，
其中包含了 **32** 种中文语言模型的预训练权重。

+--------------------+-------------------------------------+----------------------+---------------------------------+
| Model              | Pretrained Weight                   |  Chinese Language    | Details of the model            |
+====================+=====================================+======================+=================================+
|ALBERT_             |``albert-base-v1``                   |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-large-v1``                  |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-xlarge-v1``                 |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-xxlarge-v1``                |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-base-v2``                   |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-large-v2``                  |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-xlarge-v2``                 |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-xxlarge-v2``                |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-chinese-tiny``              |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-chinese-small``             |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-chinese-base``              |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-chinese-large``             |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-chinese-xlarge``            |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``albert-chinese-xxlarge``           |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|BERT_               |``bert-base-uncased``                |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``bert-large-uncased``               |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``bert-base-multilingual-uncased``   |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``bert-base-cased``                  |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``bert-base-chinese``                |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``bert-base-multilingual-cased``     |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``bert-large-cased``                 |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``bert-wwm-chinese``                 |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``bert-wwm-ext-chinese``             |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|BigBird_            |``bigbird-base-uncased``             |x                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|DistilBert_         |``distilbert-base-uncased``          |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``distilbert-base-cased``            |x                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|ELECTRA_            |``electra-small``                    |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``electra-base``                     |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``electra-large``                    |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``chinese-electra-small``            |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``chinese-electra-base``             |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|ERNIE_              |``ernie-1.0``                        |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``ernie-tiny``                       |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``ernie-2.0-en``                     |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``ernie-2.0-large-en``               |x                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|ERNIE-CTM           |``ernie-ctm``                        |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|ERNIE-GEN_          |``ernie-gen-base-en``                |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``ernie-gen-large-en``               |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``ernie-gen-large-en-430g``          |x                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|ERNIE-GRAM_         |``ernie-gram-zh``                    |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|GPT_                |``gpt-cpm-large-cn``                 |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``gpt-cpm-small-cn-distill``         |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``gpt2-medium-en``                   |x                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|NeZha_              |``nezha-base-chinese``               |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``nezha-large-chinese``              |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``nezha-base-wwm-chinese``           |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``nezha-large-wwm-chinese``          |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|RoBERTa_            |``roberta-wwm-ext``                  |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``roberta-wwm-ext-large``            |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``rbt3``                             |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``rbtl3``                            |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|SKEP_               |``skep_ernie_1.0_large_ch``          |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``skep_ernie_2.0_large_en``          |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``skep_roberta_large_en``            |x                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|TinyBert_           |``tinybert-4l-312d``                 |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``tinybert-6l-768d``                 |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``tinybert-4l-312d-v2``              |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``tinybert-6l-768d-v2``              |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``tinybert-4l-312d-zh``              |√                     |                                 |
+                    +-------------------------------------+----------------------+---------------------------------+
|                    |``tinybert-6l-768d-zh``              |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|Transformer_        |                                     |                      |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|UnifiedTransformer_ |``unified_transformer-12L-cn``       |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``unified_transformer-12L-cn-luge``  |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``plato-mini``                       |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+
|XLNet_              |``xlnet-base-cased``                 |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``xlnet-large-cased``                |x                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``chinese-xlnet-base``               |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``chinese-xlnet-mid``                |√                     |                                 |
|                    +-------------------------------------+----------------------+---------------------------------+
|                    |``chinese-xlnet-large``              |√                     |                                 |
+--------------------+-------------------------------------+----------------------+---------------------------------+


------------------------------------
Transformer预训练模型适用任务汇总
------------------------------------

+--------------------+-------------------------+----------------------+--------------------+-----------------+
| Model              | Sequence Classification | Token Classification | Question Answering | Text Generation |
+====================+=========================+======================+====================+=================+
|ALBERT_             |√                        |√                     |√                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|BERT_               |√                        |√                     |√                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|BigBird_            |√                        |x                     |x                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|DistilBert_         |√                        |√                     |√                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|ELECTRA_            |√                        |√                     |x                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|ERNIE_              |√                        |√                     |√                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|ERNIE-CTM           |x                        |√                     |x                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|ERNIE-GEN_          |x                        |x                     |x                   |√                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|ERNIE-GRAM_         |√                        |√                     |√                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|GPT_                |x                        |x                     |x                   |√                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|NeZha_              |√                        |√                     |√                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|RoBERTa_            |√                        |√                     |√                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|SKEP_               |√                        |√                     |x                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|TinyBert_           |√                        |x                     |x                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|Transformer_        |x                        |x                     |x                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|UnifiedTransformer_ |x                        |x                     |x                   |√                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+
|XLNet_              |√                        |√                     |x                   |x                |
+--------------------+-------------------------+----------------------+--------------------+-----------------+

.. _ALBERT: https://arxiv.org/abs/1909.11942
.. _BERT: https://arxiv.org/abs/1810.04805
.. _BigBird: https://arxiv.org/abs/2007.14062
.. _DistilBert: https://arxiv.org/abs/1910.01108
.. _ELECTRA: https://arxiv.org/abs/2003.10555
.. _ERNIE: https://arxiv.org/abs/1904.09223
.. _ERNIE-GEN: https://arxiv.org/abs/2001.11314
.. _ERNIE-GRAM: https://arxiv.org/abs/2010.12148
.. _GPT: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
.. _NeZha: https://arxiv.org/abs/1909.00204
.. _RoBERTa: https://arxiv.org/abs/1907.11692
.. _SKEP: https://arxiv.org/abs/2005.05635
.. _TinyBert: https://arxiv.org/abs/1909.10351
.. _Transformer: https://arxiv.org/abs/1706.03762
.. _UnifiedTransformer: https://arxiv.org/abs/2006.16779
.. _XLNet: https://arxiv.org/abs/1906.08237

------------------------------------
预训练模型使用方法
------------------------------------

PaddleNLP Transformer API在提丰富预训练模型的同时，也降低了用户的使用门槛。
只需十几行代码，用户即可完成模型加载和下游任务Fine-tuning。

.. code:: python

    from functools import partial
    import numpy as np

    import paddle
    from paddlenlp.datasets import load_dataset
    from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer

    train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])

    model = BertForSequenceClassification.from_pretrained("bert-wwm-chinese", num_classes=len(train_ds.label_list))

    tokenizer = BertTokenizer.from_pretrained("bert-wwm-chinese")

    def convert_example(example, tokenizer):
        encoded_inputs = tokenizer(text=example["text"], max_seq_len=512, pad_to_max_seq_len=True)
        return tuple([np.array(x, dtype="int64") for x in [
                encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], [example["label"]]]])
    train_ds = train_ds.map(partial(convert_example, tokenizer=tokenizer))

    batch_sampler = paddle.io.BatchSampler(dataset=train_ds, batch_size=8, shuffle=True)
    train_data_loader = paddle.io.DataLoader(dataset=train_ds, batch_sampler=batch_sampler, return_list=True)

    optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())

    criterion = paddle.nn.loss.CrossEntropyLoss()

    for input_ids, token_type_ids, labels in train_data_loader():
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = paddle.nn.functional.softmax(logits, axis=1)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

上面的代码给出使用预训练模型的简要示例，更完整详细的示例代码，
可以参考：`使用预训练模型Fine-tune完成中文文本分类任务 <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/pretrained_models/>`_

1. 加载数据集：PaddleNLP内置了多种数据集，用户可以一键导入所需的数据集。
2. 加载预训练模型：PaddleNLP的预训练模型可以很容易地通过 ``from_pretrained()`` 方法加载。
   第一个参数是汇总表中对应的 ``Pretrained Weight``，可加载对应的预训练权重。
   ``BertForSequenceClassification`` 初始化 ``__init__`` 所需的其他参数，如 ``num_classes`` 等，
   也是通过 ``from_pretrained()`` 传入。``Tokenizer`` 使用同样的 ``from_pretrained`` 方法加载。
3. 通过 ``Dataset`` 的 ``map`` 函数，使用 ``tokenizer`` 将 ``dataset`` 从原始文本处理成模型的输入。
4. 定义 ``BatchSampler`` 和 ``DataLoader``，shuffle数据、组合Batch。
5. 定义训练所需的优化器，loss函数等，就可以开始进行模型fine-tune任务。

------------------------------------
Reference
------------------------------------
- 部分中文预训练模型来自：
  `brightmart/albert_zh <https://github.com/brightmart/albert_zh>`_,
  `ymcui/Chinese-BERT-wwm <https://github.com/ymcui/Chinese-BERT-wwm>`_,
  `huawei-noah/Pretrained-Language-Model/TinyBERT <https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT>`_,
  `ymcui/Chinese-XLNet <https://github.com/ymcui/Chinese-XLNet>`_,
  `huggingface/xlnet_chinese_large <https://huggingface.co/clue/xlnet_chinese_large>`_,
  `Knover/luge-dialogue <https://github.com/PaddlePaddle/Knover/tree/luge-dialogue/luge-dialogue>`_,
  `huawei-noah/Pretrained-Language-Model/NEZHA-PyTorch/ <https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch>`_
- Lan, Zhenzhong, et al. "Albert: A lite bert for self-supervised learning of language representations." arXiv preprint arXiv:1909.11942 (2019).
- Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
- Zaheer, Manzil, et al. "Big bird: Transformers for longer sequences." arXiv preprint arXiv:2007.14062 (2020).
- Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).
- Clark, Kevin, et al. "Electra: Pre-training text encoders as discriminators rather than generators." arXiv preprint arXiv:2003.10555 (2020).
- Sun, Yu, et al. "Ernie: Enhanced representation through knowledge integration." arXiv preprint arXiv:1904.09223 (2019).
- Xiao, Dongling, et al. "Ernie-gen: An enhanced multi-flow pre-training and fine-tuning framework for natural language generation." arXiv preprint arXiv:2001.11314 (2020).
- Xiao, Dongling, et al. "ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding." arXiv preprint arXiv:2010.12148 (2020).
- Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.
- Wei, Junqiu, et al. "NEZHA: Neural contextualized representation for chinese language understanding." arXiv preprint arXiv:1909.00204 (2019).
- Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).
- Tian, Hao, et al. "SKEP: Sentiment knowledge enhanced pre-training for sentiment analysis." arXiv preprint arXiv:2005.05635 (2020).
- Vaswani, Ashish, et al. "Attention is all you need." arXiv preprint arXiv:1706.03762 (2017).
- Jiao, Xiaoqi, et al. "Tinybert: Distilling bert for natural language understanding." arXiv preprint arXiv:1909.10351 (2019).
- Bao, Siqi, et al. "Plato-2: Towards building an open-domain chatbot via curriculum learning." arXiv preprint arXiv:2006.16779 (2020).
- Yang, Zhilin, et al. "Xlnet: Generalized autoregressive pretraining for language understanding." arXiv preprint arXiv:1906.08237 (2019).
- Cui, Yiming, et al. "Pre-training with whole word masking for chinese bert." arXiv preprint arXiv:1906.08101 (2019).
