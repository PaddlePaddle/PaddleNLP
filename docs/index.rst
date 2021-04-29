欢迎使用PaddleNLP
==================

`PaddleNLP 2.0 <https://github.com/PaddlePaddle/PaddleNLP>`_ 拥有覆盖多场景的模型库、简洁易用的全流程API与动静统一的高性能分布式训练能力，旨在为飞桨开发者提升文本领域建模效率，并提供基于PaddlePaddle 2.0的NLP领域最佳实践。具备如下特性：


- **覆盖多场景的模型库**

  - PaddleNLP集成了RNN与Transformer等多种主流模型结构，涵盖从词向量、词法分析、命名实体识别、语义表示等NLP基础技术，到文本分类、文本匹配、文本生成、文本图学习、信息抽取等NLP核心技术。同时针对机器翻译、通用对话、阅读理解等系统应用提供相应核心组件与预训练模型。



- **简洁易用的全流程API**

  - 深度兼容飞桨2.0的高层API体系，内置可复用的文本建模模块(Embedding、CRF、 Seq2Vec、Transformer)，可大幅度减少在数据处理、模型组网、训练与评估、推理部署环节的开发量，提升NLP任务迭代与落地的效率。



- **动静统一的高性能分布式训练**

  - 基于飞桨2.0核心框架『动静统一』的特性与领先的混合精度优化策略，结合Fleet分布式训练API，可充分利用GPU集群资源，高效完成大规模预训练模型的分布式训练。



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

   预训练模型 <model_zoo/transformers.md>
   基本组网单元 <model_zoo/others>

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

   如何向PaddleNLP贡献数据集 <community/contribute_datasets>
   如何创建 :class:`DatasetBuilder` <community/how_to_write_a_DatasetBuilder>
   如何贡献模型 <community/contribute_models>
   如何贡献文档案例 <community/contribute_docs>
   如何加入兴趣小组 <community/join_in_PaddleNLP-SIG>

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   paddlenlp.data <source/paddlenlp.data>
   paddlenlp.datasets <source/paddlenlp.datasets>
   paddlenlp.embeddings <source/paddlenlp.embeddings>
   paddlenlp.layers <source/paddlenlp.layers>
   paddlenlp.metrics <source/paddlenlp.metrics>
   paddlenlp.ops <source/paddlenlp.ops>
   paddlenlp.seq2vec <source/paddlenlp.seq2vec>
   paddlenlp.transformers <source/paddlenlp.transformers>
   paddlenlp.utils <source/paddlenlp.utils>

Indices and tables
====================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
