

PaddleNLP Transformer预训练模型
====================================

随着深度学习的发展，NLP领域涌现了一大批高质量的Transformer类预训练模型，多次刷新了不同NLP任务的SOTA（State of the Art），极大地推动了自然语言处理的进展。
PaddleNLP为用户提供了常用的预训练模型及其相应权重，如 ``BERT``、``ERNIE``、``ALBERT``、``RoBERTa``、``XLNet`` 等，采用统一的API进行加载、训练和调用，
让开发者能够方便快捷地应用各种Transformer类预训练模型及其下游任务，且相应预训练模型权重下载速度快、稳定。

------------------------------------
预训练模型使用方法
------------------------------------

PaddleNLP Transformer API在提供丰富预训练模型的同时，也降低了用户的使用门槛。
使用Auto模块，可以加载不同网络结构的预训练模型，无需查找模型对应的类别。只需十几行代码，用户即可完成模型加载和下游任务Fine-tuning。

.. code:: python

    from functools import partial
    import numpy as np

    import paddle
    from paddlenlp.datasets import load_dataset
    from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

    train_ds = load_dataset("chnsenticorp", splits=["train"])

    model = AutoModelForSequenceClassification.from_pretrained("bert-wwm-chinese", num_classes=len(train_ds.label_list))

    tokenizer = AutoTokenizer.from_pretrained("bert-wwm-chinese")

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
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

上面的代码给出使用预训练模型的简要示例，更完整详细的示例代码，
可以参考：`使用预训练模型Fine-tune完成中文文本分类任务 <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/pretrained_models/>`_

1. 加载数据集：PaddleNLP内置了多种数据集，用户可以一键导入所需的数据集。
2. 加载预训练模型：PaddleNLP的预训练模型可以很容易地通过 ``from_pretrained()`` 方法加载。
   Auto模块（包括AutoModel, AutoTokenizer, 及各种下游任务类）提供了方便易用的接口，
   无需指定类别，即可调用不同网络结构的预训练模型。
   第一个参数是汇总表中对应的 ``Pretrained Weight``，可加载对应的预训练权重。
   ``AutoModelForSequenceClassification`` 初始化 ``__init__`` 所需的其他参数，如 ``num_classes`` 等，
   也是通过 ``from_pretrained()`` 传入。``Tokenizer`` 使用同样的 ``from_pretrained`` 方法加载。
3. 通过 ``Dataset`` 的 ``map`` 函数，使用 ``tokenizer`` 将 ``dataset`` 从原始文本处理成模型的输入。
4. 定义 ``BatchSampler`` 和 ``DataLoader``，shuffle数据、组合Batch。
5. 定义训练所需的优化器，loss函数等，就可以开始进行模型fine-tune任务。

------------------------------------
Transformer预训练模型汇总
------------------------------------

PaddleNLP的Transformer预训练模型包含从 `huggingface.co`_ 直接转换的模型权重和百度自研模型权重，方便社区用户直接迁移使用。
目前共包含了40多个主流预训练模型，500多个模型权重。

.. _huggingface.co: https://huggingface.co/models

.. toctree::
   :maxdepth: 3

   ALBERT <transformers/ALBERT/contents>
   BART <transformers/BART/contents>
   BERT <transformers/BERT/contents>
   BigBird <transformers/BigBird/contents>
   Blenderbot <transformers/Blenderbot/contents>
   Blenderbot-Small <transformers/Blenderbot-Small/contents>
   ChineseBert <transformers/ChineseBert/contents>
   ConvBert <transformers/ConvBert/contents>
   CTRL <transformers/CTRL/contents>
   DistilBert <transformers/DistilBert/contents>
   ELECTRA <transformers/ELECTRA/contents>
   ERNIE <transformers/ERNIE/contents>
   ERNIE-CTM <transformers/ERNIE-CTM/contents>
   ERNIE-DOC <transformers/ERNIE-DOC/contents>
   ERNIE-GEN <transformers/ERNIE-GEN/contents>
   ERNIE-GRAM <transformers/ERNIE-GRAM/contents>
   ERNIE-M <transformers/ERNIE-M/contents>
   FNet <transformers/FNet/contents>
   Funnel <transformers/Funnel/contents>
   GPT <transformers/GPT/contents>
   LayoutLM <transformers/LayoutLM/contents>
   LayoutLMV2 <transformers/LayoutLMV2/contents>
   LayoutXLM <transformers/LayoutXLM/contents>
   Luke <transformers/Luke/contents>
   MBart <transformers/MBart/contents>
   MegatronBert <transformers/MegatronBert/contents>
   MobileBert <transformers/MobileBert/contents>
   MPNet <transformers/MPNet/contents>
   NeZha <transformers/NeZha/contents>
   PPMiniLM <transformers/PPMiniLM/contents>
   ProphetNet <transformers/ProphetNet/contents>
   Reformer <transformers/Reformer/contents>
   RemBert <transformers/RemBert/contents>
   RoBERTa <transformers/RoBERTa/contents>
   RoFormer <transformers/RoFormer/contents>
   SKEP <transformers/SKEP/contents>
   SqueezeBert <transformers/SqueezeBert/contents>
   T5 <transformers/T5/contents>
   TinyBert <transformers/TinyBert/contents>
   UnifiedTransformer <transformers/UnifiedTransformer/contents>
   UNIMO <transformers/UNIMO/contents>
   XLNet <transformers/XLNet/contents>



------------------------------------
Transformer预训练模型适用任务汇总
------------------------------------

+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
| Model              | Sequence Classification | Token Classification | Question Answering | Text Generation | Multiple Choice |
+====================+=========================+======================+====================+=================+=================+
|ALBERT_             | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|BART_               | ✅                      | ✅                   | ✅                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|BERT_               | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|BigBird_            | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|Blenderbot_         | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|Blenderbot-Small_   | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ChineseBert_        | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ConvBert_           | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|CTRL_               | ✅                      | ❌                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|DistilBert_         | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ELECTRA_            | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE-CTM_          | ❌                      | ✅                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE-DOC_          | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE-GEN_          | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE-GRAM_         | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE-M_            | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|FNet_               | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|Funnel_             | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|GPT_                | ✅                      | ✅                   | ❌                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|LayoutLM_           | ✅                      | ✅                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|LayoutLMV2_         | ❌                      | ✅                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|LayoutXLM_          | ❌                      | ✅                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|Luke_               | ❌                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|MBart_              | ✅                      | ❌                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|MegatronBert_       | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|MobileBert_         | ✅                      | ❌                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|MPNet_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|NeZha_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|PPMiniLM_           | ✅                      | ❌                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ProphetNet_         | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|Reformer_           | ✅                      | ❌                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|RemBert_            | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|RoBERTa_            | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|RoFormer_           | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|SKEP_               | ✅                      | ✅                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|SqueezeBert_        | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|T5_                 | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|TinyBert_           | ✅                      | ❌                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|UnifiedTransformer_ | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|XLNet_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+

.. _ALBERT: https://arxiv.org/abs/1909.11942
.. _BART: https://arxiv.org/abs/1910.13461
.. _BERT: https://arxiv.org/abs/1810.04805
.. _BERT-Japanese: https://arxiv.org/abs/1810.04805
.. _BigBird: https://arxiv.org/abs/2007.14062
.. _Blenderbot: https://arxiv.org/pdf/2004.13637.pdf
.. _Blenderbot-Small: https://arxiv.org/pdf/2004.13637.pdf
.. _ChineseBert: https://arxiv.org/abs/2106.16038
.. _ConvBert: https://arxiv.org/abs/2008.02496
.. _CTRL: https://arxiv.org/abs/1909.05858
.. _DistilBert: https://arxiv.org/abs/1910.01108
.. _ELECTRA: https://arxiv.org/abs/2003.10555
.. _ERNIE: https://arxiv.org/abs/1904.09223
.. _ERNIE-CTM: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm
.. _ERNIE-DOC: https://arxiv.org/abs/2012.15688
.. _ERNIE-GEN: https://arxiv.org/abs/2001.11314
.. _ERNIE-GRAM: https://arxiv.org/abs/2010.12148
.. _ERNIE-M: https://arxiv.org/abs/2012.15674
.. _FNet: https://arxiv.org/abs/2105.03824
.. _Funnel: https://arxiv.org/abs/2006.03236
.. _GPT: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
.. _LayoutLM: https://arxiv.org/abs/1912.13318
.. _LayoutLMV2: https://arxiv.org/abs/2012.14740
.. _LayoutXLM: https://arxiv.org/abs/2104.08836
.. _Luke: https://arxiv.org/abs/2010.01057
.. _MBart: https://arxiv.org/abs/2001.08210
.. _MegatronBert: https://arxiv.org/abs/1909.08053
.. _MobileBert: https://arxiv.org/abs/2004.02984
.. _MPNet: https://arxiv.org/abs/2004.09297
.. _NeZha: https://arxiv.org/abs/1909.00204
.. _PPMiniLM: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm
.. _ProphetNet: https://arxiv.org/abs/2001.04063
.. _Reformer: https://arxiv.org/abs/2001.04451
.. _RemBert: https://arxiv.org/abs/2010.12821
.. _RoBERTa: https://arxiv.org/abs/1907.11692
.. _RoFormer: https://arxiv.org/abs/2104.09864
.. _SKEP: https://arxiv.org/abs/2005.05635
.. _SqueezeBert: https://arxiv.org/abs/2006.11316
.. _T5: https://arxiv.org/abs/1910.10683
.. _TinyBert: https://arxiv.org/abs/1909.10351
.. _UnifiedTransformer: https://arxiv.org/abs/2006.16779
.. _UNIMO: https://arxiv.org/abs/2012.15409
.. _XLNet: https://arxiv.org/abs/1906.08237

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
  `huawei-noah/Pretrained-Language-Model/NEZHA-PyTorch/ <https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch>`_,
  `ZhuiyiTechnology/simbert <https://github.com/ZhuiyiTechnology/simbert>`_
- Lan, Zhenzhong, et al. "Albert: A lite bert for self-supervised learning of language representations." arXiv preprint arXiv:1909.11942 (2019).
- Lewis, Mike, et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." arXiv preprint arXiv:1910.13461 (2019).
- Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
- Zaheer, Manzil, et al. "Big bird: Transformers for longer sequences." arXiv preprint arXiv:2007.14062 (2020).
- Stephon, Emily, et al. "Blenderbot: Recipes for building an open-domain chatbot." arXiv preprint arXiv:2004.13637 (2020).
- Stephon, Emily, et al. "Blenderbot-Small: Recipes for building an open-domain chatbot." arXiv preprint arXiv:2004.13637 (2020).
- Sun, Zijun, et al. "Chinesebert: Chinese pretraining enhanced by glyph and pinyin information." arXiv preprint arXiv:2106.16038 (2021).
- Zhang, zhengyan, et al. "CPM: A Large-scale Generative Chinese Pre-trained Language Model." arXiv preprint arXiv:2012.00413 (2020).
- Jiang, Zihang, et al. "ConvBERT: Improving BERT with Span-based Dynamic Convolution." arXiv preprint arXiv:2008.02496 (2020).
- Nitish, Bryan, et al. "CTRL: A Conditional Transformer Language Model for Controllable Generation." arXiv preprint arXiv:1909.05858 (2019).
- Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).
- Clark, Kevin, et al. "Electra: Pre-training text encoders as discriminators rather than generators." arXiv preprint arXiv:2003.10555 (2020).
- Sun, Yu, et al. "Ernie: Enhanced representation through knowledge integration." arXiv preprint arXiv:1904.09223 (2019).
- Ding, Siyu, et al. "ERNIE-Doc: A retrospective long-document modeling transformer." arXiv preprint arXiv:2012.15688 (2020).
- Xiao, Dongling, et al. "Ernie-gen: An enhanced multi-flow pre-training and fine-tuning framework for natural language generation." arXiv preprint arXiv:2001.11314 (2020).
- Xiao, Dongling, et al. "ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding." arXiv preprint arXiv:2010.12148 (2020).
- Ouyang, Xuan, et al. "ERNIE-M: enhanced multilingual representation by aligning cross-lingual semantics with monolingual corpora." arXiv preprint arXiv:2012.15674 (2020).
- Lee-Thorp, James, et al. "Fnet: Mixing tokens with fourier transforms." arXiv preprint arXiv:2105.03824 (2021).
- Dai, Zihang, et al. "Funnel-transformer: Filtering out sequential redundancy for efficient language processing." Advances in neural information processing systems 33 (2020): 4271-4282.
- Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.
- Xu, Yiheng, et al. "LayoutLM: Pre-training of Text and Layout for Document Image Understanding." arXiv preprint arXiv:1912.13318 (2019).
- Xu, Yang, et al. "LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding" arXiv preprint arXiv:2012.14740 (2020).
- Xu, Yiheng, et al. "LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding" arXiv preprint arXiv:2104.08836 (2021).
- Yamada, Ikuya, et al. "Luke: deep contextualized entity representations with entity-aware self-attention." arXiv preprint arXiv:2010.01057 (2020).
- Liu, Yinhan, et al. "MBart: Multilingual Denoising Pre-training for Neural Machine Translation" arXiv preprint arXiv:2001.08210 (2020).
- Shoeybi, Mohammad, et al. "Megatron-lm: Training multi-billion parameter language models using model parallelism." arXiv preprint arXiv:1909.08053 (2019).
- Sun, Zhiqing, et al. "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices" arXiv preprint arXiv:2004.02984 (2020).
- Song, Kaitao, et al. "MPNet: Masked and Permuted Pre-training for Language Understanding." arXiv preprint arXiv:2004.09297 (2020).
- Wei, Junqiu, et al. "NEZHA: Neural contextualized representation for chinese language understanding." arXiv preprint arXiv:1909.00204 (2019).
- Qi, Weizhen, et al. "Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training." arXiv preprint arXiv:2001.04063 (2020).
- Kitaev, Nikita, et al. "Reformer: The efficient Transformer." arXiv preprint arXiv:2001.04451 (2020).
- Chung, Hyung Won, et al. "Rethinking embedding coupling in pre-trained language models." arXiv preprint arXiv:2010.12821 (2020).
- Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).
- Su Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv preprint arXiv:2104.09864 (2021).
- Tian, Hao, et al. "SKEP: Sentiment knowledge enhanced pre-training for sentiment analysis." arXiv preprint arXiv:2005.05635 (2020).
- Forrest, ALbert, et al. "SqueezeBERT: What can computer vision teach NLP about efficient neural networks?" arXiv preprint arXiv:2006.11316 (2020).
- Raffel, Colin, et al. "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." arXiv preprint arXiv:1910.10683 (2019).
- Vaswani, Ashish, et al. "Attention is all you need." arXiv preprint arXiv:1706.03762 (2017).
- Jiao, Xiaoqi, et al. "Tinybert: Distilling bert for natural language understanding." arXiv preprint arXiv:1909.10351 (2019).
- Bao, Siqi, et al. "Plato-2: Towards building an open-domain chatbot via curriculum learning." arXiv preprint arXiv:2006.16779 (2020).
- Yang, Zhilin, et al. "Xlnet: Generalized autoregressive pretraining for language understanding." arXiv preprint arXiv:1906.08237 (2019).
- Cui, Yiming, et al. "Pre-training with whole word masking for chinese bert." arXiv preprint arXiv:1906.08101 (2019).
- Wang, Quan, et al. “Building Chinese Biomedical Language Models via Multi-Level Text Discrimination.” arXiv preprint arXiv:2110.07244 (2021).
