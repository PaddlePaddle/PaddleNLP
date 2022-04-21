

------------------------------------
BERT-Japanese模型汇总
------------------------------------



下表汇总介绍了目前PaddleNLP支持的BERT-Japanese模型预训练权重。

+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
| Pretrained Weight                                                                | Language     | Details of the model                    |
+==================================================================================+==============+=========================================+
|``iverxin/bert-base-japanese``                                                    | Japanese     | 12-layer, 768-hidden,                   |
|                                                                                  |              | 12-heads, 110M parameters.              |
|                                                                                  |              | Trained on Japanese text.               |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``iverxin/bert-base-japanese-whole-word-masking``                                 | Japanese     | 12-layer, 768-hidden,                   |
|                                                                                  |              | 12-heads, 109M parameters.              |
|                                                                                  |              | Trained on Japanese text using          |
|                                                                                  |              | Whole-Word-Masking.                     |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``iverxin/bert-base-japanese-char``                                               | Japanese     | 12-layer, 768-hidden,                   |
|                                                                                  |              | 12-heads, 89M parameters.               |
|                                                                                  |              | Trained on Japanese char text.          |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|``iverxin/bert-base-japanese-char-whole-word-masking``                            | Japanese     | 12-layer, 768-hidden,                   |
|                                                                                  |              | 12-heads, 89M parameters.               |
|                                                                                  |              | Trained on Japanese char text using     |
|                                                                                  |              | Whole-Word-Masking.                     |
+----------------------------------------------------------------------------------+--------------+-----------------------------------------+