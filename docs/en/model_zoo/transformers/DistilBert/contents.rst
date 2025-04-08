DistilBERT Model Summary

The following table summarizes the currently supported DistilBERT models and their corresponding pretrained weights in PaddleNLP. For detailed model specifications, please refer to the corresponding links.

+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| Pretrained Weight                                                                | Language     | Details of the model                                                             |
+==================================================================================+==============+==================================================================================+
| ``distilbert-base-uncased``                                                      | English      | 6-layer, 768-hidden,                                                             |
|                                                                                  |              | 12-heads, 66M parameters.                                                        |
|                                                                                  |              | The DistilBERT model distilled from                                              |
|                                                                                  |              | the BERT model ``bert-base-uncased``                                             |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| ``distilbert-base-cased``                                                        | English      | 6-layer, 768-hidden,                                                            |
|                                                                                  |              | 12-heads, 66M parameters.                                                       |
|                                                                                  |              | The DistilBERT model distilled from                                             |
|                                                                                  |              | the BERT model ``bert-base-cased``.                                             |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| ``distilbert-base-multilingual-cased``
| English                                                                          |              | 6-layer, 768-hidden, 12-heads,                                                   |
|----------------------------------------------------------------------------------|--------------|----------------------------------------------------------------------------------|
|                                                                                  |              | 200M parameters. The DistilBERT model                                            |
|                                                                                  |              | distilled from the BERT model                                                   |
|                                                                                  |              | ``bert-base-multilingual-cased``.                                                |
|                                                                                  |              |                                                                                  |
|                                                                                  |              | Please refer to:                                                                 |
|                                                                                  |              | `distilbert-base-multilingual-cased`_                                            |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
|``sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english``               | 2-layer, 128-hidden, 2-heads, 28M parameters. The tiny version of              |
|                                                                                  | the above model, with fewer layers and parameters. Also in English.             |
|                                                                                  | Licensed under Apache 2.0.                                                     |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| English                                                                          |              | 2-layer, 2-hidden,                                                               |
|                                                                                  |              | 2-heads, 50K parameters.                                                         |
|                                                                                  |              | The DistilBERT model.                                                            |
|                                                                                  |              |                                                                                  |
|                                                                                  |              | Please refer to:                                                                 |
|                                                                                  |              | `sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english`_                |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+

.. _distilbert-base-multilingual-cased: https://huggingface.co/distilbert-base-multilingual-cased
.. _sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english: https://huggingface.co/sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english