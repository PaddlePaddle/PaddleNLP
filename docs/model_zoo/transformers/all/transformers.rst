PaddleNLP Transformer API
====================================

随着深度学习的发展，NLP领域涌现了一大批高质量的Transformer类预训练模型，多次刷新各种NLP任务SOTA（State of the Art）。
PaddleNLP为用户提供了常用的 ``BERT``、``ERNIE``、``ALBERT``、``RoBERTa``、``XLNet`` 等经典结构预训练模型，
让开发者能够方便快捷应用各类Transformer预训练模型及其下游任务。

------------------------------------
Transformer预训练模型汇总
------------------------------------



下表汇总了介绍了目前PaddleNLP支持的各类预训练模型以及对应预训练权重。我们目前提供了 **32** 种网络结构， **136** 种预训练的参数权重供用户使用，
其中包含了 **59** 种中文语言模型的预训练权重。

+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
| Model              | Pretrained Weight                                                                | Language     | Details of the model                    |
+====================+==================================================================================+==============+=========================================+
|ALBERT_             |``albert-base-v1``                                                                | English      | 12 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 768-hidden, 12-heads, 11M parameters.   |
|                    |                                                                                  |              | ALBERT base model                       |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-large-v1``                                                               | English      | 24 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 1024-hidden, 16-heads, 17M parameters.  |
|                    |                                                                                  |              | ALBERT large model                      |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-xlarge-v1``                                                              | English      | 24 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 2048-hidden, 16-heads, 58M parameters.  |
|                    |                                                                                  |              | ALBERT xlarge model                     |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-xxlarge-v1``                                                             | English      | 12 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 4096-hidden, 64-heads, 223M parameters. |
|                    |                                                                                  |              | ALBERT xxlarge model                    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-base-v2``                                                                | English      | 12 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 768-hidden, 12-heads, 11M parameters.   |
|                    |                                                                                  |              | ALBERT base model (version2)            |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-large-v2``                                                               | English      | 24 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 1024-hidden, 16-heads, 17M parameters.  |
|                    |                                                                                  |              | ALBERT large model (version2)           |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-xlarge-v2``                                                              | English      | 24 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 2048-hidden, 16-heads, 58M parameters.  |
|                    |                                                                                  |              | ALBERT xlarge model (version2)          |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-xxlarge-v2``                                                             | English      | 12 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 4096-hidden, 64-heads, 223M parameters. |
|                    |                                                                                  |              | ALBERT xxlarge model (version2)         |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-chinese-tiny``                                                           | Chinese      | 4 repeating layers, 128 embedding,      |
|                    |                                                                                  |              | 312-hidden, 12-heads, 4M parameters.    |
|                    |                                                                                  |              | ALBERT tiny model (Chinese)             |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-chinese-small``                                                          | Chinese      | 6 repeating layers, 128 embedding,      |
|                    |                                                                                  |              | 384-hidden, 12-heads, _M parameters.    |
|                    |                                                                                  |              | ALBERT small model (Chinese)            |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-chinese-base``                                                           | Chinese      | 12 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 768-hidden, 12-heads, 12M parameters.   |
|                    |                                                                                  |              | ALBERT base model (Chinese)             |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-chinese-large``                                                          | Chinese      | 24 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 1024-hidden, 16-heads, 18M parameters.  |
|                    |                                                                                  |              | ALBERT large model (Chinese)            |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-chinese-xlarge``                                                         | Chinese      | 24 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 2048-hidden, 16-heads, 60M parameters.  |
|                    |                                                                                  |              | ALBERT xlarge model (Chinese)           |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``albert-chinese-xxlarge``                                                        | Chinese      | 12 repeating layers, 128 embedding,     |
|                    |                                                                                  |              | 4096-hidden, 16-heads, 235M parameters. |
|                    |                                                                                  |              | ALBERT xxlarge model (Chinese)          |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|BART_               |``bart-base``                                                                     | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 217M parameters.              |
|                    |                                                                                  |              | BART base model (English)               |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``bart-large``                                                                    | English      | 24-layer, 768-hidden,                   |
|                    |                                                                                  |              | 16-heads, 509M parameters.              |
|                    |                                                                                  |              | BART large model (English).             |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|BERT_               |``bert-base-uncased``                                                             | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 110M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``bert-large-uncased``                                                            | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 336M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``bert-base-cased``                                                               | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 109M parameters.              |
|                    |                                                                                  |              | Trained on cased English text.          |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``bert-large-cased``                                                              | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 335M parameters.              |
|                    |                                                                                  |              | Trained on cased English text.          |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``bert-base-multilingual-uncased``                                                | Multilingual | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 168M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased text             |
|                    |                                                                                  |              | in the top 102 languages                |
|                    |                                                                                  |              | with the largest Wikipedias.            |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``bert-base-multilingual-cased``                                                  | Multilingual | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 179M parameters.              |
|                    |                                                                                  |              | Trained on cased text                   |
|                    |                                                                                  |              | in the top 104 languages                |
|                    |                                                                                  |              | with the largest Wikipedias.            |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``bert-base-chinese``                                                             | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on cased Chinese Simplified     |
|                    |                                                                                  |              | and Traditional text.                   |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``bert-wwm-chinese``                                                              | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on cased Chinese Simplified     |
|                    |                                                                                  |              | and Traditional text using              |
|                    |                                                                                  |              | Whole-Word-Masking.                     |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``bert-wwm-ext-chinese``                                                          | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on cased Chinese Simplified     |
|                    |                                                                                  |              | and Traditional text using              |
|                    |                                                                                  |              | Whole-Word-Masking with extented data.  |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/ckiplab-bert-base-chinese-ner``                                          | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Finetuned on NER task.                  |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/ckiplab-bert-base-chinese-pos``                                          | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Finetuned on POS task.                  |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/ckiplab-bert-base-chinese-ws``                                           | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Finetuned on WS task.                   |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/nlptown-bert-base-multilingual-uncased-sentiment``                       | Multilingual | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 167M parameters.              |
|                    |                                                                                  |              | Finetuned for sentiment analysis on     |
|                    |                                                                                  |              | product reviews in six languages:       |
|                    |                                                                                  |              | English, Dutch, German, French,         |
|                    |                                                                                  |              | Spanish and Italian.                    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/tbs17-MathBERT``                                                         | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 110M parameters.              |
|                    |                                                                                  |              | Trained on pre-k to graduate math       |
|                    |                                                                                  |              | language (English) using a masked       |
|                    |                                                                                  |              | language modeling (MLM) objective.      |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``macbert-base-chinese``                                                          | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Trained with novel MLM as correction    |
|                    |                                                                                  |              | pre-training task.                      |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``macbert-large-chinese``                                                         | Chinese      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 326M parameters.              |
|                    |                                                                                  |              | Trained with novel MLM as correction    |
|                    |                                                                                  |              | pre-training task.                      |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``simbert-base-chinese``                                                          | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on 22 million pairs of similar  |
|                    |                                                                                  |              | sentences crawed from Baidu Know.       |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``Langboat/mengzi-bert-base``                                                     | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Trained on 300G Chinese Corpus Datasets.|
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``Langboat/mengzi-bert-base-fin``                                                 | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Trained on 20G Finacial Corpus,         |
|                    |                                                                                  |              | based on ``Langboat/mengzi-bert-base``. |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|BERT-Japanese_      |``iverxin/bert-base-japanese``                                                    | Japanese     | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 110M parameters.              |
|                    |                                                                                  |              | Trained on Japanese text.               |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``iverxin/bert-base-japanese-whole-word-masking``                                 | Japanese     | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 109M parameters.              |
|                    |                                                                                  |              | Trained on Japanese text using          |
|                    |                                                                                  |              | Whole-Word-Masking.                     |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``iverxin/bert-base-japanese-char``                                               | Japanese     | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 89M parameters.               |
|                    |                                                                                  |              | Trained on Japanese char text.          |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``iverxin/bert-base-japanese-char-whole-word-masking``                            | Japanese     | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 89M parameters.               |
|                    |                                                                                  |              | Trained on Japanese char text using     |
|                    |                                                                                  |              | Whole-Word-Masking.                     |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|BigBird_            |``bigbird-base-uncased``                                                          | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 127M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|Blenderbot_         |``blenderbot-3B``                                                                 | English      | 26-layer,                               |
|                    |                                                                                  |              | 32-heads, 3B parameters.                |
|                    |                                                                                  |              | The Blenderbot base model.              |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``blenderbot-400M-distill``                                                       | English      | 14-layer, 384-hidden,                   |
|                    |                                                                                  |              | 32-heads, 400M parameters.              |
|                    |                                                                                  |              | The Blenderbot distil model.            |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``blenderbot-1B-distill``                                                         | English      | 14-layer,                               |
|                    |                                                                                  |              | 32-heads, 1478M parameters.             |
|                    |                                                                                  |              | The Blenderbot Distil 1B model.         |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|Blenderbot-Small_   |``blenderbot_small-90M``                                                          | English      | 16-layer,                               |
|                    |                                                                                  |              | 16-heads, 90M parameters.               |
|                    |                                                                                  |              | The Blenderbot small model.             |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|ConvBert_           |``convbert-base``                                                                 | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 106M parameters.              |
|                    |                                                                                  |              | The ConvBERT base model.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``convbert-medium-small``                                                         | English      | 12-layer, 384-hidden,                   |
|                    |                                                                                  |              | 8-heads, 17M parameters.                |
|                    |                                                                                  |              | The ConvBERT medium small model.        |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``convbert-small``                                                                | English      | 12-layer, 128-hidden,                   |
|                    |                                                                                  |              | 4-heads, 13M parameters.                |
|                    |                                                                                  |              | The ConvBERT small model.               |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|CTRL_               |``ctrl``                                                                          | English      | 48-layer, 1280-hidden,                  |
|                    |                                                                                  |              | 16-heads, 1701M parameters.             |
|                    |                                                                                  |              | The CTRL base model.                    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``sshleifer-tiny-ctrl``                                                           | English      | 2-layer, 16-hidden,                     |
|                    |                                                                                  |              | 2-heads, 5M parameters.                 |
|                    |                                                                                  |              | The Tiny CTRL model.                    |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|DistilBert_         |``distilbert-base-uncased``                                                       | English      | 6-layer, 768-hidden,                    |
|                    |                                                                                  |              | 12-heads, 66M parameters.               |
|                    |                                                                                  |              | The DistilBERT model distilled from     |
|                    |                                                                                  |              | the BERT model ``bert-base-uncased``    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``distilbert-base-cased``                                                         | English      | 6-layer, 768-hidden,                    |
|                    |                                                                                  |              | 12-heads, 66M parameters.               |
|                    |                                                                                  |              | The DistilBERT model distilled from     |
|                    |                                                                                  |              | the BERT model ``bert-base-cased``      |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``distilbert-base-multilingual-cased``                                            | English      | 6-layer, 768-hidden,                    |
|                    |                                                                                  |              | 12-heads, 200M parameters.              |
|                    |                                                                                  |              | The DistilBERT model distilled from     |
|                    |                                                                                  |              | the BERT model                          |
|                    |                                                                                  |              | ``bert-base-multilingual-cased``        |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``sshleifer-tiny-distilbert-base-uncase-finetuned-sst-2-english``                 | English      | 2-layer, 2-hidden,                      |
|                    |                                                                                  |              | 2-heads, 50K parameters.                |
|                    |                                                                                  |              | The DistilBERT model                    |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|ELECTRA_            |``electra-small``                                                                 | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 4-heads, 14M parameters.                |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``electra-base``                                                                  | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 109M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``electra-large``                                                                 | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 334M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``chinese-electra-small``                                                         | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 4-heads, 12M parameters.                |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``chinese-electra-base``                                                          | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``ernie-health-chinese``                                                          | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Trained on Chinese medical corpus.      |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/hfl-chinese-electra-180g-base-discriminator``                            | Chinese      | Discriminator, 12-layer, 768-hidden,    |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Trained on 180g Chinese text.           |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/hfl-chinese-electra-180g-small-ex-discriminator``                        | Chinese      | Discriminator, 24-layer, 256-hidden,    |
|                    |                                                                                  |              | 4-heads, 24M parameters.                |
|                    |                                                                                  |              | Trained on 180g Chinese text.           |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/hfl-chinese-legal-electra-small-generator``                              | Chinese      | Generator, 12-layer, 64-hidden,         |
|                    |                                                                                  |              | 1-heads, 3M parameters.                 |
|                    |                                                                                  |              | Trained on Chinese legal corpus.        |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|ERNIE_              |``ernie-1.0``                                                                     | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``ernie-tiny``                                                                    | Chinese      | 3-layer, 1024-hidden,                   |
|                    |                                                                                  |              | 16-heads, _M parameters.                |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``ernie-2.0-en``                                                                  | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 103M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``ernie-2.0-en-finetuned-squad``                                                  | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 110M parameters.              |
|                    |                                                                                  |              | Trained on finetuned squad text.        |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``ernie-2.0-large-en``                                                            | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 336M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|ERNIE-DOC_          |``ernie-doc-base-zh``                                                             | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``ernie-doc-base-en``                                                             | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 103M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|ERNIE-GEN_          |``ernie-gen-base-en``                                                             | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``ernie-gen-large-en``                                                            | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 336M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``ernie-gen-large-en-430g``                                                       | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 336M parameters.              |
|                    |                                                                                  |              | Trained on lower-cased English text.    |
|                    |                                                                                  |              | with extended data (430 GB).            |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|ERNIE-GRAM_         |``ernie-gram-zh``                                                                 | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
+                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``ernie-gram-zh-finetuned-dureader-robust``                                       | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    |                                                                                  |              | Then finetuned on dreader-robust        |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|GPT_                |``gpt-cpm-large-cn``                                                              | Chinese      | 32-layer, 2560-hidden,                  |
|                    |                                                                                  |              | 32-heads, 2.6B parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``gpt-cpm-small-cn-distill``                                                      | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 109M parameters.              |
|                    |                                                                                  |              | The model distilled from                |
|                    |                                                                                  |              | the GPT model ``gpt-cpm-large-cn``      |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``gpt2-en``                                                                       | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 117M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``gpt2-medium-en``                                                                | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 345M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``gpt2-large-en``                                                                 | English      | 36-layer, 1280-hidden,                  |
|                    |                                                                                  |              | 20-heads, 774M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``gpt2-xl-en``                                                                    | English      | 48-layer, 1600-hidden,                  |
|                    |                                                                                  |              | 25-heads, 1558M parameters.             |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/distilgpt2``                                                             | English      | 6-layer, 768-hidden,                    |
|                    |                                                                                  |              | 12-heads, 81M parameters.               |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/microsoft-DialoGPT-small``                                               | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 124M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/microsoft-DialoGPT-medium``                                              | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 354M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/microsoft-DialoGPT-large``                                               | English      | 36-layer, 1280-hidden,                  |
|                    |                                                                                  |              | 20-heads, 774M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``junnyu/uer-gpt2-chinese-poem``                                                  | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 103M parameters.              |
|                    |                                                                                  |              | Trained on Chinese poetry corpus.       |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|LayoutLM_           |``layoutlm-base-uncased``                                                         | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 339M parameters.              |
|                    |                                                                                  |              | LayoutLm base uncased model.            |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``layoutlm-large-uncased``                                                        | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 51M parameters.               |
|                    |                                                                                  |              | LayoutLm large Uncased model.           |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|LayoutLMV2_         |``layoutlmv2-base-uncased``                                                       | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 200M parameters.              |
|                    |                                                                                  |              | LayoutLmv2 base uncased model.          |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``layoutlmv2-large-uncased``                                                      | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, _M parameters.                |
|                    |                                                                                  |              | LayoutLmv2 large uncased model.         |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|LayoutXLM_          |``layoutxlm-base-uncased``                                                        | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 369M parameters.              |
|                    |                                                                                  |              | Layoutxlm base uncased model.           |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|MBart_              |``mbart-large-cc25``                                                              | English      | 12-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 12-heads, 1123M parameters.             |
|                    |                                                                                  |              | The ``mbart-large-cc25`` model.         |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``mbart-large-en-ro``                                                             | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 16-heads, 1123M parameters.             |
|                    |                                                                                  |              | The ``mbart-large rn-ro`` model .       |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``mbart-large-50-one-to-many-mmt``                                                | English      | 12-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 1123M parameters.             |
|                    |                                                                                  |              | ``mbart-large-50-one-to-many-mmt``      |
|                    |                                                                                  |              | model.                                  |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``mbart-large-50-many-to-one-mmt``                                                | English      | 12-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 1123M parameters.             |
|                    |                                                                                  |              | ``mbart-large-50-many-to-one-mmt``      |
|                    |                                                                                  |              | model.                                  |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``mbart-large-50-many-to-many-mmt``                                               | English      | 12-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 1123M parameters.             |
|                    |                                                                                  |              | ``mbart-large-50-many-to-many-mmt``     |
|                    |                                                                                  |              | model.                                  |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|Mobilebert_         |``mobilebert-uncased``                                                            | English      | 24-layer, 512-hidden,                   |
|                    |                                                                                  |              | 4-heads, 24M parameters.                |
|                    |                                                                                  |              | Mobilebert uncased Model.               |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|MPNet_              |``mpnet-base``                                                                    | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 109M parameters.              |
|                    |                                                                                  |              | MPNet Base Model.                       |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|NeZha_              |``nezha-base-chinese``                                                            | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nezha-large-chinese``                                                           | Chinese      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 336M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nezha-base-wwm-chinese``                                                        | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 16-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nezha-large-wwm-chinese``                                                       | Chinese      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 336M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|Reformer_           |``reformer-enwik8``                                                               | English      | 12-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 8-heads, 148M parameters.               |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``reformer-crime-and-punishment``                                                 | English      | 6-layer, 256-hidden,                    |
|                    |                                                                                  |              | 2-heads, 3M parameters.                 |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|RoBERTa_            |``roberta-wwm-ext``                                                               | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Trained on English Text using           |
|                    |                                                                                  |              | Whole-Word-Masking with extended data.  |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roberta-wwm-ext-large``                                                         | Chinese      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 325M parameters.              |
|                    |                                                                                  |              | Trained on English Text using           |
|                    |                                                                                  |              | Whole-Word-Masking with extended data.  |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``rbt3``                                                                          | Chinese      | 3-layer, 768-hidden,                    |
|                    |                                                                                  |              | 12-heads, 38M parameters.               |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``rbtl3``                                                                         | Chinese      | 3-layer, 1024-hidden,                   |
|                    |                                                                                  |              | 16-heads, 61M parameters.               |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nosaydomore/deepset-roberta-base-squad2``                                       | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 124M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nosaydomore/roberta-en-base``                                                   | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 163M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nosaydomore/roberta-en-large``                                                  | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 408M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nosaydomore/sshleifei-tiny-distilroberta-base``                                 | English      | 2-layer, 2-hidden,                      |
|                    |                                                                                  |              | 2-heads, 0.25M parameters.              |
|                    |                                                                                  |              | Trained on English text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nosaydomore/uer-roberta-base-chn-extractive-qa``                                | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 101M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nosaydomore/uer-roberta-base-ft-chinanews-chn``                                 | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 102M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``nosaydomore/uer-roberta-base-ft-cluener2020-chn``                               | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 101M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|RoFormer_           |``roformer-chinese-small``                                                        | Chinese      | 6-layer, 384-hidden,                    |
|                    |                                                                                  |              | 6-heads, 30M parameters.                |
|                    |                                                                                  |              | Roformer Small Chinese model.           |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roformer-chinese-base``                	                                        | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 124M parameters.              |
|                    |                                                                                  |              | Roformer Base Chinese model.            |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roformer-chinese-char-small``                                                   | Chinese      | 6-layer, 384-hidden,                    |
|                    |                                                                                  |              | 6-heads, 15M parameters.                |
|                    |                                                                                  |              | Roformer Chinese Char Small model.      |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roformer-chinese-char-base``                                                    | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 95M parameters.               |
|                    |                                                                                  |              | Roformer Chinese Char Base model.       |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roformer-chinese-sim-char-ft-small``                                            | Chinese      | 6-layer, 384-hidden,                    |
|                    |                                                                                  |              | 6-heads, 15M parameters.                |
|                    |                                                                                  |              | Roformer Chinese Char Ft Small model.   |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roformer-chinese-sim-char-ft-base``                                             | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 95M parameters.               |
|                    |                                                                                  |              | Roformer Chinese Char Ft Base model.    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roformer-chinese-sim-char-small``                                               | Chinese      | 6-layer, 384-hidden,                    |
|                    |                                                                                  |              | 6-heads, 15M parameters.                |
|                    |                                                                                  |              | Roformer Chinese Sim Char Small model.  |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roformer-chinese-sim-char-base``                                                | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 95M parameters.               |
|                    |                                                                                  |              | Roformer Chinese Sim Char Base model.   |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roformer-english-small-discriminator``                                          | English      | 12-layer, 256-hidden,                   |
|                    |                                                                                  |              | 4-heads, 13M parameters.                |
|                    |                                                                                  |              | Roformer English Small Discriminator.   |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``roformer-english-small-generator``                                              | English      | 12-layer, 64-hidden,                    |
|                    |                                                                                  |              | 1-heads, 5M parameters.                 |
|                    |                                                                                  |              | Roformer English Small Generator.       |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|SKEP_               |``skep_ernie_1.0_large_ch``                                                       | Chinese      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 336M parameters.              |
|                    |                                                                                  |              | Trained using the Erine model           |
|                    |                                                                                  |              | ``ernie_1.0``                           |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``skep_ernie_2.0_large_en``                                                       | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 336M parameters.              |
|                    |                                                                                  |              | Trained using the Erine model           |
|                    |                                                                                  |              | ``ernie_2.0_large_en``                  |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``skep_roberta_large_en``                                                         | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 355M parameters.              |
|                    |                                                                                  |              | Trained using the RoBERTa model         |
|                    |                                                                                  |              | ``roberta_large_en``                    |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|SqueezeBert_        |``squeezebert-uncased``                                                           | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 51M parameters.               |
|                    |                                                                                  |              | SqueezeBert Uncased model.              |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``squeezebert-mnli``                                                              | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 51M parameters.               |
|                    |                                                                                  |              | SqueezeBert Mnli model.                 |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``squeezebert-mnli-headless``                                                     | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 51M parameters.               |
|                    |                                                                                  |              | SqueezeBert Mnli Headless model.        |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|T5_                 |``t5-small``                                                                      | English      | 6-layer, 512-hidden,                    |
|                    |                                                                                  |              | 8-heads, 93M parameters.                |
|                    |                                                                                  |              | T5 small model.                         |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``t5-base``                                                                       | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 272M parameters.              |
|                    |                                                                                  |              | T5 base model.                          |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``t5-large``                                                                      | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 803M parameters.              |
|                    |                                                                                  |              | T5 large model.                         |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|TinyBert_           |``tinybert-4l-312d``                                                              | English      | 4-layer, 312-hidden,                    |
|                    |                                                                                  |              | 12-heads, 14.5M parameters.             |
|                    |                                                                                  |              | The TinyBert model distilled from       |
|                    |                                                                                  |              | the BERT model ``bert-base-uncased``    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``tinybert-6l-768d``                                                              | English      | 6-layer, 768-hidden,                    |
|                    |                                                                                  |              | 12-heads, 67M parameters.               |
|                    |                                                                                  |              | The TinyBert model distilled from       |
|                    |                                                                                  |              | the BERT model ``bert-base-uncased``    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``tinybert-4l-312d-v2``                                                           | English      | 4-layer, 312-hidden,                    |
|                    |                                                                                  |              | 12-heads, 14.5M parameters.             |
|                    |                                                                                  |              | The TinyBert model distilled from       |
|                    |                                                                                  |              | the BERT model ``bert-base-uncased``    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``tinybert-6l-768d-v2``                                                           | English      | 6-layer, 768-hidden,                    |
|                    |                                                                                  |              | 12-heads, 67M parameters.               |
|                    |                                                                                  |              | The TinyBert model distilled from       |
|                    |                                                                                  |              | the BERT model ``bert-base-uncased``    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``tinybert-4l-312d-zh``                                                           | Chinese      | 4-layer, 312-hidden,                    |
|                    |                                                                                  |              | 12-heads, 14.5M parameters.             |
|                    |                                                                                  |              | The TinyBert model distilled from       |
|                    |                                                                                  |              | the BERT model ``bert-base-uncased``    |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``tinybert-6l-768d-zh``                                                           | Chinese      | 6-layer, 768-hidden,                    |
|                    |                                                                                  |              | 12-heads, 67M parameters.               |
|                    |                                                                                  |              | The TinyBert model distilled from       |
|                    |                                                                                  |              | the BERT model ``bert-base-uncased``    |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|UnifiedTransformer_ |``unified_transformer-12L-cn``                                                    | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text.                |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``unified_transformer-12L-cn-luge``                                               | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 108M parameters.              |
|                    |                                                                                  |              | Trained on Chinese text (LUGE.ai).      |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``plato-mini``                                                                    | Chinese      | 6-layer, 768-hidden,                    |
|                    |                                                                                  |              | 12-heads, 66M parameters.               |
|                    |                                                                                  |              | Trained on Chinese text.                |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|UNIMO_              |``unimo-text-1.0``                                                                | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 99M parameters.               |
|                    |                                                                                  |              | UNIMO-text-1.0 model.                   |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``unimo-text-1.0-lcsts-new``                                                      | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 99M parameters.               |
|                    |                                                                                  |              | Finetuned on lcsts_new dataset.         |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``unimo-text-1.0-large``                                                          | Chinese      | 24-layer, 768-hidden,                   |
|                    |                                                                                  |              | 16-heads, 316M parameters.              |
|                    |                                                                                  |              | UNIMO-text-1.0 large model.             |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|XLNet_              |``xlnet-base-cased``                                                              | English      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 110M parameters.              |
|                    |                                                                                  |              | XLNet English model                     |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``xlnet-large-cased``                                                             | English      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, 340M parameters.              |
|                    |                                                                                  |              | XLNet Large English model               |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``chinese-xlnet-base``                                                            | Chinese      | 12-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 117M parameters.              |
|                    |                                                                                  |              | XLNet Chinese model                     |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``chinese-xlnet-mid``                                                             | Chinese      | 24-layer, 768-hidden,                   |
|                    |                                                                                  |              | 12-heads, 209M parameters.              |
|                    |                                                                                  |              | XLNet Medium Chinese model              |
|                    +----------------------------------------------------------------------------------+--------------+-----------------------------------------+
|                    |``chinese-xlnet-large``                                                           | Chinese      | 24-layer, 1024-hidden,                  |
|                    |                                                                                  |              | 16-heads, _M parameters.                |
|                    |                                                                                  |              | XLNet Large Chinese model               |
+--------------------+----------------------------------------------------------------------------------+--------------+-----------------------------------------+


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
|ConvBert_           | ✅                      | ✅                   | ✅                 | ✅              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|CTRL_               | ✅                      | ❌                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|DistilBert_         | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ELECTRA_            | ✅                      | ✅                   | ❌                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE_              | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE-DOC_          | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE-GEN_          | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ERNIE-GRAM_         | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|GPT_                | ✅                      | ✅                   | ❌                 | ✅              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|LayoutLM_           | ✅                      | ✅                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|LayoutLMV2_         | ❌                      | ✅                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|LayoutXLM_          | ❌                      | ✅                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|Mbart_              | ✅                      | ❌                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|MobileBert_         | ✅                      | ❌                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|MPNet_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|NeZha_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|ReFormer_           | ✅                      | ❌                   | ✅                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+
|RoBERTa_            | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
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
|XLNet_              | ✅                      | ✅                   | ❌                 | ❌              | ❌              |
+--------------------+-------------------------+----------------------+--------------------+-----------------+-----------------+

.. _ALBERT: https://arxiv.org/abs/1909.11942
.. _BART: https://arxiv.org/abs/1910.13461
.. _BERT: https://arxiv.org/abs/1810.04805
.. _BERT-Japanese: https://arxiv.org/abs/1810.04805
.. _BigBird: https://arxiv.org/abs/2007.14062
.. _Blenderbot: https://arxiv.org/pdf/2004.13637.pdf
.. _Blenderbot-Small: https://arxiv.org/pdf/2004.13637.pdf
.. _ConvBert: https://arxiv.org/abs/2008.02496
.. _CTRL: https://arxiv.org/abs/1909.05858
.. _DistilBert: https://arxiv.org/abs/1910.01108
.. _ELECTRA: https://arxiv.org/abs/2003.10555
.. _ERNIE: https://arxiv.org/abs/1904.09223
.. _ERNIE-DOC: https://arxiv.org/abs/2012.15688
.. _ERNIE-GEN: https://arxiv.org/abs/2001.11314
.. _ERNIE-GRAM: https://arxiv.org/abs/2010.12148
.. _GPT: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
.. _LayoutLM: https://arxiv.org/abs/1912.13318
.. _LayoutLMV2: https://arxiv.org/abs/2012.14740
.. _LayoutXLM: https://arxiv.org/abs/2104.08836
.. _MBart: https://arxiv.org/abs/2001.08210
.. _MobileBert: https://arxiv.org/abs/2004.02984
.. _MPNet: https://arxiv.org/abs/2004.09297
.. _NeZha: https://arxiv.org/abs/1909.00204
.. _ReFormer: https://arxiv.org/abs/2001.04451
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
预训练模型使用方法
------------------------------------

PaddleNLP Transformer API在提丰富预训练模型的同时，也降低了用户的使用门槛。
使用Auto模块，可以加载不同网络结构的预训练模型，无需查找
模型对应的类别。只需十几行代码，用户即可完成模型加载和下游任务Fine-tuning。

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
- Zhang, zhengyan, et al. "CPM: A Large-scale Generative Chinese Pre-trained Language Model." arXiv preprint arXiv:2012.00413 (2020).
- Jiang, Zihang, et al. "ConvBERT: Improving BERT with Span-based Dynamic Convolution." arXiv preprint arXiv:2008.02496 (2020).
- Nitish, Bryan, et al. "CTRL: A Conditional Transformer Language Model for Controllable Generation." arXiv preprint arXiv:1909.05858 (2019).
- Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).
- Clark, Kevin, et al. "Electra: Pre-training text encoders as discriminators rather than generators." arXiv preprint arXiv:2003.10555 (2020).
- Sun, Yu, et al. "Ernie: Enhanced representation through knowledge integration." arXiv preprint arXiv:1904.09223 (2019).
- Xiao, Dongling, et al. "Ernie-gen: An enhanced multi-flow pre-training and fine-tuning framework for natural language generation." arXiv preprint arXiv:2001.11314 (2020).
- Xiao, Dongling, et al. "ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding." arXiv preprint arXiv:2010.12148 (2020).
- Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.
- Xu, Yiheng, et al. "LayoutLM: Pre-training of Text and Layout for Document Image Understanding." arXiv preprint arXiv:1912.13318 (2019).
- Xu, Yang, et al. "LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding" arXiv preprint arXiv:2012.14740 (2020).
- Xu, Yiheng, et al. "LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding" arXiv preprint arXiv:2104.08836 (2021).
- Liu, Yinhan, et al. "MBart: Multilingual Denoising Pre-training for Neural Machine Translation" arXiv preprint arXiv:2001.08210 (2020).
- Sun, Zhiqing, et al. "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices" arXiv preprint arXiv:2004.02984 (2020).
- Song, Kaitao, et al. "MPNet: Masked and Permuted Pre-training for Language Understanding." arXiv preprint arXiv:2004.09297 (2020).
- Wei, Junqiu, et al. "NEZHA: Neural contextualized representation for chinese language understanding." arXiv preprint arXiv:1909.00204 (2019).
- Kitaev, Nikita, et al. "Reformer: The efficient Transformer." arXiv preprint arXiv:2001.04451 (2020).
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
