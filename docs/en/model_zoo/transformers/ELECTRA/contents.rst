ELECTRA Model Summary

The following table summarizes the currently supported ELECTRA models and their corresponding pretrained weights in PaddleNLP.
For model details, please refer to the corresponding links.

+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| Pretrained Weight                                                                | Language     | Details of the model                                                             |
+==================================================================================+==============+==================================================================================+
| ``electra-small``                                                                | English      | 12-layer, 768-hidden,                                                            |
|                                                                                  |              | 4-heads, 14M parameters.                                                         |
|                                                                                  |              | Trained on lower-cased English text.                                             |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| ``electra-base``
| Model Name                                                                       | Language     | Description                                                                     |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| ``electra-small``                                                                | English      | 12-layer, 768-hidden,                                                           |
|                                                                                  |              | 12-heads, 109M parameters.                                                       |
|                                                                                  |              | Trained on lower-cased English text.                                             |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| ``electra-large``                                                                | English      | 24-layer, 1024-hidden,                                                           |
|                                                                                  |              | 16-heads, 334M parameters.                                                       |
|                                                                                  |              | Trained on lower-cased English text.                                             |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| ``chinese-electra-small``                                                        | Chinese      | 12-layer, 256-hidden,                                                            |
|                                                                                  |              | 4-heads, 12M parameters.                                                         |
|                                                                                  |              | Trained on Chinese text.                                                         |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| Model Name                                                                       | Language     | Description                                                                      |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| ``chinese-bert-wwm-ext``                                                         | Chinese      | 12-layer, 768-hidden,                                                            |
|                                                                                  |              | 4-heads, 12M parameters.                                                         |
|                                                                                  |              | Trained on Chinese text.                                                         |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| ``chinese-electra-base``                                                         | Chinese      | 12-layer, 768-hidden,                                                            |
|                                                                                  |              | 12-heads, 102M parameters.                                                       |
|                                                                                  |              | Trained on Chinese text.                                                         |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| ``ernie-health-chinese``
| Model Name                                                                       | Configuration                                                                 |
|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| ``hfl/chinese-electra-180g-base-discriminator``                                  | 12-layer, 768-hidden, 12-heads, 102M parameters. Trained on Chinese medical corpus. |
| ``hfl/chinese-electra-180g-base-discriminator`` | Discriminator, 12-layer, 768-hidden, |
|                                                 | 12-heads, 102M parameters.           |
|                                                 | Trained on 180g Chinese text.        |
|                                                 |                                      |
|                                                 | Please refer to:                     |
|                                                 | `hfl/chinese-electra-180g-base-discriminator`_ |

+-------------------------------------------------+--------------------------------------+
| ``hfl/chinese-electra-180g-small-ex-discriminator`` |
| ``hfl/chinese-electra-180g-small-ex-discriminator`` | Discriminator | 24-layer, 256-hidden,                                                           |
|-----------------------------------------------------|----------------|----------------------------------------------------------------------------------|
|                                                     |                | 4-heads, 24M parameters.                                                        |
|                                                     |                | Trained on 180g Chinese text.                                                   |
|                                                     |                |                                                                                  |
|                                                     |                | Please refer to:                                                                 |
|                                                     |                | `hfl/chinese-electra-180g-small-ex-discriminator`_                              |
+-----------------------------------------------------+----------------+----------------------------------------------------------------------------------+
| ``hfl/chinese-legal-electra-small-generator``       | Generator      | Same architecture as discriminator, with 24M parameters.                        |
|                                                     |                | Trained on Chinese legal documents.                                             |
|                                                     |                |                                                                                  |
|                                                     |                | Please refer to:                                                                 |
|                                                     |                | `hfl/chinese-legal-electra-small-generator`_                                    |
+-----------------------------------------------------+----------------+----------------------------------------------------------------------------------+

Note: Both models follow ELECTRA-style architecture. The discriminator is trained for sequence classification tasks, while the generator is used for token-level predictions in the ELECTRA framework.
| Chinese      | Generator, 12-layer, 64-hidden,                                                  |
|              | 1-heads, 3M parameters.                                                          |
|              | Trained on Chinese legal corpus.                                                 |
|              |                                                                                  |
|              | Please refer to:                                                                 |
|              | `hfl/chinese-legal-electra-small-generator`
.. _hfl/chinese-electra-180g-base-discriminator: https://huggingface.co/hfl/chinese-electra-180g-base-discriminator
.. _hfl/chinese-electra-180g-small-ex-discriminator: https://huggingface.co/hfl/chinese-electra-180g-small-ex-discriminator
.. _hfl/chinese-legal-electra-small-generator: https://huggingface.co/hfl/chinese-legal-electra-small-generator