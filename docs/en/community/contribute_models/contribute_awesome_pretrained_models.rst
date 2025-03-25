====================================================================================
Contribute Pre-trained Model Weights
====================================================================================

1. Model Network Architecture Types
------------------------------------------------------------------------------------
PaddleNLP currently supports most mainstream pre-trained model architectures, including both Baidu's self-developed models (e.g., ERNIE series) and widely-used models in the industry (e.g., BERT, ALBERT, GPT, RoBERTa, XLNet, etc.).

For a comprehensive list of supported pre-trained model architectures in PaddleNLP, please refer to
`Transformer Pre-trained Models Collection <https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html>`_
(continuously updated, and we warmly welcome contributions of new models: `How to Contribute New Models <https://paddlenlp.readthedocs.io/zh/latest/community/contribute_models/contribute_new_models.html>`_).

2. Model Parameter Weight Types
------------------------------------------------------------------------------------
We sincerely welcome contributions of high-quality model parameter weights.
Supported parameter weight types include but are not limited to (taking BERT model as example):

- BERT pre-trained model parameter weights not yet included in PaddleNLP
  (e.g., `bert-base-japanese-char <https://huggingface.co/cl-tohoku/bert-base-japanese-char>`, `danish-bert-botxo <https://huggingface.co/Maltehb/danish-bert-botxo>`_);
- Pre-trained BERT model weights in vertical domains (e.g., mathematics, finance, legal, medical, etc.)
  (e.g., `MathBERT <https://huggingface.co/tbs17/MathBERT>`, `finbert <https://huggingface.co/ProsusAI/finbert>`_);
- Fine-tuned BERT model weights on specific downstream tasks
  (e.g., `bert-base-multilingual-uncased-sentiment <https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment>`,
  `bert-base-NER <https://huggingface.co/dslim/bert-base-NER>`_).
You are a professional NLP technical translator. Translate Chinese to English while:
1. Preserving EXACT formatting (markdown/rst/code)
2. Keeping technical terms in English
3. Maintaining code/math blocks unchanged
4. Using proper academic grammar
5. Keep code block in documents original
6. Keep the link in markdown/rst the same. 如[链接](#这里), 翻译为 [link](#这里) 而不是 [link](#here)
7. Keep the html tag in markdown/rst the same.
6. Just return the result of Translate. no additional messages.
Maintain RST syntax exactly, translate section headers but keep anchors

3. Parameter Weight Format Conversion
------------------------------------------------------------------------------------
When we want to contribute model weights from an open-source project on GitHub, but find the weights saved in other deep learning framework formats (PyTorch, TensorFlow, etc.),
we need to perform model format conversion between different deep learning frameworks. The following link provides a detailed tutorial on PyTorch to Paddle model format conversion:
`PyTorch to Paddle Model Format Conversion Documentation <./convert_pytorch_to_paddle.rst>`_.

4. Contribution Process
------------------------------------------------------------------------------------
4.1 Preparing Weight-related Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generally, we need to prepare four files for parameter weight contributions: **model_state.pdparams**, **vocab.txt**, **tokenizer_config.json**,
and **model_config.json**.

- The model_state.pdparams file can be obtained through the aforementioned parameter weight format conversion process;
- The vocab.txt file can directly use the original model's corresponding vocab file (depending on the model's tokenizer type, this filename might be spiece.model, etc.);
- The model_config.json file can refer to the model_config.json file saved via the corresponding model.save_pretrained() interface;
- The tokenizer_config.json file can refer to the tokenizer_config.json file saved via the corresponding tokenizer.save_pretrained() interface;

4.2 Creating Personal Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If this is your first time contributing weights, you need to create a new directory under ``PaddleNLP/community/``.
The directory name should use your GitHub username, e.g., creating a directory ``PaddleNLP/community/yingyibiao/``.
If a personal directory already exists, you can skip this step.

4.3 Creating Weight Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a new weight directory under your personal directory from step 4.2. The weight directory name should be the name of the model weights being contributed.
For example, to contribute a model called ``bert-base-uncased-sst-2-finetuned``,
create a weight directory ``PaddleNLP/community/yingyibiao/bert-base-uncased-sst-2-finetuned/``
4.4 Add PR-related files in the weights directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Add two files under the directory from step 4.3: ``README.md`` and ``files.json``.

- ``README.md`` contains detailed introduction, usage examples, weight sources, etc. about your contributed weights.
- ``files.json`` contains the weight-related files and corresponding addresses obtained in step 4.1. An example of files.json content is shown below. Just replace *yingyibiao* and *bert-base-uncased-sst-2-finetuned* in the URLs with your GitHub username and weight name respectively.

.. code:: python

  {
    "model_config_file": "https://bj.bcebos.com/paddlenlp/models/transformers/community/yingyibiao/bert-base-uncased-sst-2-finetuned/model_config.json",
    "model_state": "https://bj.bcebos.com/paddlenlp/models/transformers/community/yingyibiao/bert-base-uncased-sst-2-finetuned/model_state.pdparams",
    "tokenizer_config_file": "https://bj.bcebos.com/paddlenlp/models/transformers/community/yingyibiao/bert-base-uncased-sst-2-finetuned/tokenizer_config.json",
    "vocab_file": "https://bj.bcebos.com/paddlenlp/models/transformers/community/yingyibiao/bert-base-uncased-sst-2-finetuned/vocab.txt"
  }

4.5 Submit PR on GitHub for contribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- For first-time open source contributors, please refer to `first-contributions <https://github.com/firstcontributions/first-contributions>`_.
- Please refer to the `bert-base-uncased-sst-2-finetuned PR <.>`_ for an example of model weight contribution PR.