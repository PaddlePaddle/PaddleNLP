

------------------------------------
GPT模型汇总
------------------------------------



下表汇总介绍了目前PaddleNLP支持的GPT模型对应预训练权重。
关于模型的具体细节可以参考对应链接。

+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
| Pretrained Weight                                                                | Language     | Details of the model                    |
+==================================================================================+==============+=========================================+
|``gpt-cpm-large-cn``                                                              | Chinese      | 32-layer, 2560-hidden,                  |
|                                                                                  |              | 32-heads, 2.6B parameters.              |
|                                                                                  |              | Trained on Chinese text.                |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``gpt-cpm-small-cn-distill``                                                      | Chinese      | 12-layer, 768-hidden,                   |
|                                                                                  |              | 12-heads, 109M parameters.              |
|                                                                                  |              | The model distilled from                |
|                                                                                  |              | the GPT model ``gpt-cpm-large-cn``      |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``gpt2-en``                                                                       | English      | 12-layer, 768-hidden,                   |
|                                                                                  |              | 12-heads, 117M parameters.              |
|                                                                                  |              | Trained on English text.                |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``gpt2-medium-en``                                                                | English      | 24-layer, 1024-hidden,                  |
|                                                                                  |              | 16-heads, 345M parameters.              |
|                                                                                  |              | Trained on English text.                |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``gpt2-large-en``                                                                 | English      | 36-layer, 1280-hidden,                  |
|                                                                                  |              | 20-heads, 774M parameters.              |
|                                                                                  |              | Trained on English text.                |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``gpt2-xl-en``                                                                    | English      | 48-layer, 1600-hidden,                  |
|                                                                                  |              | 25-heads, 1558M parameters.             |
|                                                                                  |              | Trained on English text.                |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``junnyu/distilgpt2``                                                             | English      | 6-layer, 768-hidden,                    |
|                                                                                  |              | 12-heads, 81M parameters.               |
|                                                                                  |              | Trained on English text.                |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``junnyu/microsoft-DialoGPT-small``                                               | English      | 12-layer, 768-hidden,                   |
|                                                                                  |              | 12-heads, 124M parameters.              |
|                                                                                  |              | Trained on English text.                |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``junnyu/microsoft-DialoGPT-medium``                                              | English      | 24-layer, 1024-hidden,                  |
|                                                                                  |              | 16-heads, 354M parameters.              |
|                                                                                  |              | Trained on English text.                |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``junnyu/microsoft-DialoGPT-large``                                               | English      | 36-layer, 1280-hidden,                  |
|                                                                                  |              | 20-heads, 774M parameters.              |
|                                                                                  |              | Trained on English text.                |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``junnyu/uer-gpt2-chinese-poem``                                                  | Chinese      | 12-layer, 768-hidden,                   |
|                                                                                  |              | 12-heads, 103M parameters.              |
|                                                                                  |              | Trained on Chinese poetry corpus.       |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+

.. _microsoft-DialoGPT-small: https://huggingface.co/microsoft/DialoGPT-small
.. _microsoft-DialoGPT-medium: https://huggingface.co/microsoft/DialoGPT-medium
.. _microsoft-DialoGPT-large: https://huggingface.co/microsoft/DialoGPT-large