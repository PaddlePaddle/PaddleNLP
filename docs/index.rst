欢迎使用PaddleNLP
==================

`PaddleNLP <https://github.com/PaddlePaddle/PaddleNLP>`_ 是飞桨自然语言处理开发库，具备 **易用的文本领域API**，**多场景的应用示例**、和 **高性能分布式训练** 三大特点，旨在提升飞桨开发者文本领域建模效率，旨在提升开发者在文本领域的开发效率，并提供丰富的NLP应用示例。


- **易用的文本领域API**

  - 提供丰富的产业级预置任务能力 **Taskflow** 和全流程的文本领域API：支持丰富中文数据集加载的 **Dataset API**，可灵活高效地完成数据预处理的 **Data API** ，预置60+预训练词向量的 **Embedding API** ，提供100+预训练模型的 **Transformer API** 等，可大幅提升NLP任务建模的效率。

- **多场景的应用示例**

  - 覆盖从学术到产业级的NLP应用示例，涵盖NLP基础技术、NLP系统应用以及相关拓展应用。全面基于飞桨核心框架2.0全新API体系开发，为开发者提供飞桨文本领域的最佳实践。

- **高性能分布式训练**

  - 基于飞桨核心框架领先的自动混合精度优化策略，结合分布式Fleet API，支持4D混合并行策略，可高效地完成大规模预训练模型训练。


* 项目GitHub: https://github.com/PaddlePaddle/PaddleNLP
* 项目Gitee: https://gitee.com/paddlepaddle/PaddleNLP
* GitHub Issue反馈: https://github.com/PaddlePaddle/PaddleNLP/issues
* 官方QQ技术交流群: 973379845


.. toctree::
   :maxdepth: 1
   :caption: 快速开始

   安装 <get_started/installation>
   10分钟完成高精度中文情感分析 <get_started/quick_start>

.. toctree::
   :maxdepth: 1
   :caption: 数据准备

   整体介绍 <data_prepare/overview>
   数据集列表 <data_prepare/dataset_list>
   加载数据集 <data_prepare/dataset_load>
   自定义数据集 <data_prepare/dataset_self_defined>
   数据处理 <data_prepare/data_preprocess>

.. toctree::
   :maxdepth: 1
   :caption: 模型库

   Transformer预训练模型 <model_zoo/index>
   使用Trainer API训练 <trainer.md>
   一键预测功能 <model_zoo/taskflow>
   预训练词向量 <model_zoo/embeddings>
   

.. toctree::
   :maxdepth: 1
   :caption: 评价指标

   评价指标 <metrics/metrics.md>

.. toctree::
   :maxdepth: 1
   :caption: 实践教程

   AI Studio Notebook <tutorials/overview>

.. toctree::
   :maxdepth: 1
   :caption: 进阶指南

   模型压缩 <advanced_guide/model_compression/index>
   文本生成高性能加速 <advanced_guide/fastergeneration/index>
   大规模分布式训练 <advanced_guide/distributed_training>

.. toctree::
   :maxdepth: 1
   :caption: 社区交流共建

   如何贡献模型 <community/contribute_models/index>
   如何贡献数据集 <community/contribute_datasets/index>
   如何贡献文档案例 <community/contribute_docs>
   如何加入兴趣小组 <community/join_in_PaddleNLP-SIG>

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
