欢迎使用PaddleNLP
==================

`PaddleNLP 2.0 <https://github.com/PaddlePaddle/PaddleNLP>`_ 是基于飞桨的文本领域核心库，具备 **易用的文本领域API**，**多场景的应用示例**、和 **高性能分布式训练** 三大特点，旨在提升飞桨开发者文本领域建模效率，并提供基于飞桨核心框架2.0的NLP任务最佳实践。


- **易用的文本领域API**

  - 提供从数据集加载、文本预处理、模型组网、模型评估、到推理加速的领域API：如一键加载中文数据集的 **Dataset API**，可灵活高效地完成数据预处理的Data API，预置60+预训练词向量的 **Embedding API**; 提供50+预训练模型的生态基础能力的 **Transformer API**，可大幅提升NLP任务建模和迭代的效率。

- **多场景的应用示例**

  - PaddleNLP 2.0提供多粒度多场景的应用示例，涵盖从NLP基础技术、NLP核心技术、NLP系统应用以及文本相关的拓展应用等。全面基于飞桨2.0全新API体系开发，为开发提供飞桨2.0框架在文本领域的最佳实践。

- **高性能分布式训练**

  - 基于飞桨核心框架『**动静统一**』的特性与领先的自动混合精度优化策略，通过分布式Fleet API可支持超大规模参数的4D混合并行策略，并且可根据硬件情况灵活可配，高效地完成超大规模参数的模型训练。


* 项目GitHub: https://github.com/PaddlePaddle/PaddleNLP
* 项目Gitee: https://gitee.com/paddlepaddle/PaddleNLP
* GitHub Issue反馈: https://github.com/PaddlePaddle/PaddleNLP/issues
* 官方QQ技术交流群: 973379845


.. toctree::
   :maxdepth: 2
   :caption: 快速开始

   安装 <get_started/installation>
   10分钟完成高精度中文情感分析 <get_started/quick_start>

.. toctree::
   :maxdepth: 2
   :caption: 数据准备

   整体介绍 <data_prepare/overview>
   数据集列表 <data_prepare/dataset_list>
   加载数据集 <data_prepare/dataset_load>
   自定义数据集 <data_prepare/dataset_self_defined>
   数据处理 <data_prepare/data_preprocess>

.. toctree::
   :maxdepth: 2
   :caption: 模型库

   Transformer预训练模型 <model_zoo/transformers>
   预训练词向量 <model_zoo/embeddings>

.. toctree::
   :maxdepth: 2
   :caption: 评价指标

   评价指标 <metrics/metrics.md>

.. toctree::
   :maxdepth: 2
   :caption: 实践教程

   AI Studio Notebook <tutorials/overview>

.. toctree::
   :maxdepth: 2
   :caption: 进阶指南

   模型压缩 <advanced_guide/model_compression/index>
   高性能预测部署 <advanced_guide/deployment>
   大规模分布式训练 <advanced_guide/distributed_training>

.. toctree::
   :maxdepth: 2
   :caption: 社区交流共建

   如何贡献模型 <community/contribute_models/index>
   如何贡献数据集 <community/contribute_datasets/index>
   如何贡献文档案例 <community/contribute_docs>
   如何加入兴趣小组 <community/join_in_PaddleNLP-SIG>

.. toctree::
   :maxdepth: 2
   :caption: FAQ

   FAQ <FAQ.md>

.. toctree::
   :maxdepth: 2
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
   paddlenlp.transformers <source/paddlenlp.transformers>
   paddlenlp.utils <source/paddlenlp.utils>

Indices and tables
====================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
