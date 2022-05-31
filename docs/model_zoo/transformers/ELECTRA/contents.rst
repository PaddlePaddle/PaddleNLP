

------------------------------------
ELECTRA模型汇总
------------------------------------



下表汇总介绍了目前PaddleNLP支持的ELECTRA模型对应预训练权重。
关于模型的具体细节可以参考对应链接。

+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| Pretrained Weight                                                                | Language     | Details of the model                                                             |
+==================================================================================+==============+==================================================================================+
|``electra-small``                                                                 | English      | 12-layer, 768-hidden,                                                            |
|                                                                                  |              | 4-heads, 14M parameters.                                                         |
|                                                                                  |              | Trained on lower-cased English text.                                             |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``electra-base``                                                                  | English      | 12-layer, 768-hidden,                                                            |
|                                                                                  |              | 12-heads, 109M parameters.                                                       |
|                                                                                  |              | Trained on lower-cased English text.                                             |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``electra-large``                                                                 | English      | 24-layer, 1024-hidden,                                                           |
|                                                                                  |              | 16-heads, 334M parameters.                                                       |
|                                                                                  |              | Trained on lower-cased English text.                                             |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``chinese-electra-small``                                                         | Chinese      | 12-layer, 768-hidden,                                                            |
|                                                                                  |              | 4-heads, 12M parameters.                                                         |
|                                                                                  |              | Trained on Chinese text.                                                         |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``chinese-electra-base``                                                          | Chinese      | 12-layer, 768-hidden,                                                            |
|                                                                                  |              | 12-heads, 102M parameters.                                                       |
|                                                                                  |              | Trained on Chinese text.                                                         |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``ernie-health-chinese``                                                          | Chinese      | 12-layer, 768-hidden,                                                            |
|                                                                                  |              | 12-heads, 102M parameters.                                                       |
|                                                                                  |              | Trained on Chinese medical corpus.                                               |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``hfl/chinese-electra-180g-base-discriminator``                                   | Chinese      | Discriminator, 12-layer, 768-hidden,                                             |
|                                                                                  |              | 12-heads, 102M parameters.                                                       |
|                                                                                  |              | Trained on 180g Chinese text.                                                    |
|                                                                                  |              |                                                                                  |
|                                                                                  |              | Please refer to:                                                                 |
|                                                                                  |              | `hfl/chinese-electra-180g-base-discriminator`_                                   |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``hfl/chinese-electra-180g-small-ex-discriminator``                               | Chinese      | Discriminator, 24-layer, 256-hidden,                                             |
|                                                                                  |              | 4-heads, 24M parameters.                                                         |
|                                                                                  |              | Trained on 180g Chinese text.                                                    |
|                                                                                  |              |                                                                                  |
|                                                                                  |              | Please refer to:                                                                 |
|                                                                                  |              | `hfl/chinese-electra-180g-small-ex-discriminator`_                               |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``hfl/chinese-legal-electra-small-generator``                                     | Chinese      | Generator, 12-layer, 64-hidden,                                                  |
|                                                                                  |              | 1-heads, 3M parameters.                                                          |
|                                                                                  |              | Trained on Chinese legal corpus.                                                 |
|                                                                                  |              |                                                                                  |
|                                                                                  |              | Please refer to:                                                                 |
|                                                                                  |              | `hfl/chinese-legal-electra-small-generator`_                                     |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+

.. _hfl/chinese-electra-180g-base-discriminator: https://huggingface.co/hfl/chinese-electra-180g-base-discriminator
.. _hfl/chinese-electra-180g-small-ex-discriminator: https://huggingface.co/hfl/chinese-electra-180g-small-ex-discriminator
.. _hfl/chinese-legal-electra-small-generator: https://huggingface.co/hfl/chinese-legal-electra-small-generator
