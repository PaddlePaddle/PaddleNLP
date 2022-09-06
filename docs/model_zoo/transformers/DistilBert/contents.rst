

------------------------------------
DistilBERT模型汇总
------------------------------------



下表汇总介绍了目前PaddleNLP支持的DistilBERT模型对应预训练权重。
关于模型的具体细节可以参考对应链接。

+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| Pretrained Weight                                                                | Language     | Details of the model                                                             |
+==================================================================================+==============+==================================================================================+
|``distilbert-base-uncased``                                                       | English      | 6-layer, 768-hidden,                                                             |
|                                                                                  |              | 12-heads, 66M parameters.                                                        |
|                                                                                  |              | The DistilBERT model distilled from                                              |
|                                                                                  |              | the BERT model ``bert-base-uncased``.                                            |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``distilbert-base-cased``                                                         | English      | 6-layer, 768-hidden,                                                             |
|                                                                                  |              | 12-heads, 66M parameters.                                                        |
|                                                                                  |              | The DistilBERT model distilled from                                              |
|                                                                                  |              | the BERT model ``bert-base-cased``.                                              |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``distilbert-base-multilingual-cased``                                            | English      | 6-layer, 768-hidden, 12-heads,                                                   |
|                                                                                  |              | 200M parameters. The DistilBERT model                                            |
|                                                                                  |              | distilled from the BERT model                                                    |
|                                                                                  |              | ``bert-base-multilingual-cased``.                                                |
|                                                                                  |              |                                                                                  |
|                                                                                  |              | Please refer to:                                                                 |
|                                                                                  |              | `distilbert-base-multilingual-cased`_                                            |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english``                | English      | 2-layer, 2-hidden,                                                               |
|                                                                                  |              | 2-heads, 50K parameters.                                                         |
|                                                                                  |              | The DistilBERT model.                                                            |
|                                                                                  |              |                                                                                  |
|                                                                                  |              | Please refer to:                                                                 |
|                                                                                  |              | `sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english`_                |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+

.. _distilbert-base-multilingual-cased: https://huggingface.co/distilbert-base-multilingual-cased
.. _sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english: https://huggingface.co/sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english
