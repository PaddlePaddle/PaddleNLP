Welcome to PaddleNLP
=====================

`PaddleNLP <https://github.com/PaddlePaddle/PaddleNLP>`_ is the natural language processing development library of PaddlePaddle, featuring **user-friendly text domain APIs**, **multi-scenario application examples**, and **high-performance distributed training**. It aims to enhance the modeling efficiency in the text domain for PaddlePaddle developers and provides a wealth of NLP application examples.

- **User-friendly Text Domain APIs**

  - Offers a rich set of industry-grade preset task capabilities **Taskflow** and comprehensive text domain APIs: supports a wide range of Chinese dataset loading with **Dataset API**, enables flexible and efficient data preprocessing with **Data API**, provides over 60+ pretrained word vectors with **Embedding API**, and offers over 100+ pretrained models with **Transformer API**, significantly boosting the efficiency of NLP task modeling.

- **Multi-scenario Application Examples**

  - Covers NLP application examples from academic to industry levels, including fundamental NLP technologies, NLP system applications, and related extended applications. Developed entirely based on the new API system of the PaddlePaddle core framework 2.0, it provides developers with best practices in the text domain of PaddlePaddle.

- **High-performance Distributed Training**

  - Based on the leading automatic mixed precision optimization strategy of the PaddlePaddle core framework, combined with the distributed Fleet API, it supports 4D hybrid parallel strategies, efficiently completing large-scale pretrained model training.

* Project GitHub: https://github.com/PaddlePaddle/PaddleNLP
* Project Gitee: https://gitee.com/paddlepaddle/PaddleNLP
* GitHub Issue Feedback: https://github.com/PaddlePaddle/PaddleNLP/issues
* WeChat Group: Scan the QR code on WeChat and fill out the questionnaire to join the group for in-depth communication with numerous community developers and the official team.

.. image:: https://github.com/user-attachments/assets/3a58cc9f-69c7-4ccb-b6f5-73e966b8051a
   :width: 200px
   :align: center
   :alt: paddlenlp WeChat group QR code


.. toctree::
   :maxdepth: 1
   :caption: Model Library List
   
   Introduction to Popular Models <hot_model.md>
   Model Library List <model_list.rst>
   

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   Installation <get_started/installation>
   Text Generation <get_started/generate>
   Quick Training <get_started/training>
   Quick Inference <get_started/inference>

.. toctree::
   :maxdepth: 1
   :caption: Paddle LLM Training
   
   Paddle LLM Main Documentation <llm/README.md>
   LLM - Pre-training Documentation <llm/docs/pretrain.rst>
   LLM - Fine-tuning Documentation <llm/docs/finetune.md>
   LLM - DPO Documentation <llm/docs/dpo.md>
   LLM - RLHF Documentation <llm/docs/rlhf.md>
   Model Merging Documentation <llm/docs/mergekit.md>


.. toctree::
   :maxdepth: 1
   :caption: Paddle LLM Inference

   Docker Deployment - Quick Start Guide <llm/server/docs/general_model_inference.md>
   LLM Inference Tutorial <llm/docs/predict/inference_index.rst>
   Practical Optimization <llm/docs/predict/infer_optimize.rst>
   Static Graph Model List <llm/server/docs/static_models.md>
   Inference Quantization Tutorial for Various Models <llm/docs/predict/models.rst>
   Heterogeneous Device Inference <llm/docs/predict/devices.rst>
   LLM - Quantization Tutorial <llm/docs/quantization.md>


.. toctree::
   :maxdepth: 1
   :caption: Paddle LLM Featured Technologies

   <llm/docs/unified_checkpoint.md>
   <llm/docs/flashmask.md>
   <llm/docs/llm_trainer.rst>



.. toctree::
   :maxdepth: 1
   :caption: PaddleNLP Toolkit

   One-click Prediction Function <model_zoo/taskflow>
   Pre-trained Word Embeddings <model_zoo/embeddings>

.. toctree::
   :maxdepth: 1
   :caption: PaddleNLP Tutorials

   Transformer Pre-trained Model <model_zoo/index>
   Trainer API Training Tutorial <trainer.md>
   Dialogue Template Tutorial <get_started/chat_template>
   Multi-turn Dialogue Fine-tuning Tutorial <llm/docs/chat_template.md>
   Chinese Sentiment Analysis Tutorial <get_started/quick_start>
   Model Compression Tutorial <compression.md>
   Data Distillation Tutorial <llm/application/distill/README.md>
   Torch2Paddle Weight Conversion Tutorial <llm/docs/torch2paddle.md>



.. toctree::
   :maxdepth: 1
   :caption: Evaluation Metrics

   Evaluation Metrics <metrics/metrics.md>



..    :maxdepth: 1
..    :caption: Data Preparation

..    Overview <data_prepare/overview>
..    Dataset List <data_prepare/dataset_list>
..    Load Dataset <data_prepare/dataset_load>
..    Custom Dataset <data_prepare/dataset_self_defined>
..    Data Preprocessing <data_prepare/data_preprocess>


.. toctree::
   :maxdepth: 1
   :caption: Practical Tutorials

   AI Studio Notebook <tutorials/overview>

.. toctree::
   :maxdepth: 1
   :caption: Advanced Guide

   Model Compression <advanced_guide/model_compression/index>
   High-Performance Acceleration for Text Generation <advanced_guide/fastgeneration/index>
   Large-Scale Distributed Training <advanced_guide/distributed_training>

.. toctree::
   :maxdepth: 1
   :caption: Community Collaboration

   How to Contribute Models <community/contribute_models/index>
   How to Contribute Datasets <community/contribute_datasets/index>
   How to Contribute Documentation Examples <community/contribute_docs>
   How to Join Interest Groups <community/join_in_PaddleNLP-SIG>

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   FAQ <FAQ.md>

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   
   paddlenlp.data <source/paddlenlp.data>
   paddlenlp.datasets <source/paddlenlp.datasets>
   paddlenlp.embeddings <source/paddlenlp.embeddings>
   paddlenlp.layers <source/paddlenlp.layers>
   paddlenlp.losses <source/paddlenlp.losses>
   paddlenlp.metrics <source/paddlenlp.metrics>
   paddlenlp.ops <source/paddlenlp.ops>
   paddlenlp.seq2vec <source/paddlenlp.seq2vec>
   paddlenlp.taskflow <source/paddlenlp.taskflow>
   paddlenlp.trainer <source/paddlenlp.trainer>
   paddlenlp.transformers <source/paddlenlp.transformers>
   paddlenlp.utils <source/paddlenlp.utils>

Indices and tables
====================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
