欢迎使用PaddleNLP
==================

`PaddleNLP <https://github.com/PaddlePaddle/PaddleNLP>`_ PaddleNLP是一款基于飞桨深度学习框架的大语言模型(LLM)开发套件，支持在多种硬件上进行高效的大模型训练、无损压缩以及高性能推理。PaddleNLP 具备简单易用和性能极致的特点，致力于助力开发者实现高效的大模型产业级应用。


- **🔧 多硬件训推一体**

  - 支持英伟达 GPU、昆仑 XPU、昇腾 NPU、燧原 GCU 和海光 DCU 等多个硬件的大模型和自然语言理解模型训练和推理，套件接口支持硬件快速切换，大幅降低硬件切换研发成本。

- **🚀 高效易用的预训练**

  - 支持纯数据并行策略、分组参数切片的数据并行策略、张量模型并行策略和流水线模型并行策略的4D 高性能训练，Trainer 支持分布式策略配置化，降低复杂分布式组合带来的使用成本；可以使得训练断点支持机器资源动态扩缩容恢复。此外，异步保存，模型存储可加速95%，Checkpoint 压缩，可节省78.5%存储空间。

- **🤗 高效精调**

  - 精调算法深度结合零填充数据流和 FlashMask 高性能算子，降低训练无效数据填充和计算，大幅提升精调训练吞吐。

- **🎛️ 无损压缩和高性能推理**

  - 大模型套件高性能推理模块内置动态插入和全环节算子融合策略，极大加快并行推理速度。底层实现细节封装化，实现开箱即用的高性能并行推理能力。



* 项目GitHub: https://github.com/PaddlePaddle/PaddleNLP
* 项目Gitee: https://gitee.com/paddlepaddle/PaddleNLP
* GitHub Issue反馈: https://github.com/PaddlePaddle/PaddleNLP/issues
* 微信交流群: 微信扫描二维码并填写问卷之后，即可加入交流群，与众多社区开发者以及官方团队深度交流。

.. image:: https://github.com/user-attachments/assets/3a58cc9f-69c7-4ccb-b6f5-73e966b8051a
   :width: 200px
   :align: center
   :alt: paddlenlp微信交流群二维码


.. toctree::
   :maxdepth: 1
   :caption: 模型库列表
   
   热门模型介绍 <hot_model.md>
   模型库列表 <website/index.md>
   

.. toctree::
   :maxdepth: 1
   :caption: 快速开始

   安装 <get_started/installation>
   文本生成  <get_started/generate>
   快速训练 <get_started/training>
   快速推理 <get_started/inference>

.. toctree::
   :maxdepth: 1
   :caption: 飞桨大模型训练

   飞桨大模型主文档 <llm/README.md>
   大模型-预训练文档 <llm/docs/pretrain.rst>
   大模型-精调文档 <llm/docs/finetune.md>
   大模型-DPO文档 <llm/docs/dpo.md>
   大模型-RLHF文档 <llm/docs/rlhf.md>
   模型融合文档 <llm/docs/mergekit.md>


.. toctree::
   :maxdepth: 1
   :caption: 飞桨大模型推理

   Docker快速部署教程 <llm/server/docs/general_model_inference.md>
   大模型推理教程 <llm/docs/predict/inference_index.rst>
   实践调优 <llm/docs/predict/infer_optimize.rst>
   静态图模型列表 <llm/server/docs/static_models.md>
   各个模型推理量化教程 <llm/docs/predict/models.rst>
   异构设备推理 <llm/docs/predict/devices.rst>
   大模型-量化教程 <llm/docs/quantization.md>


.. toctree::
   :maxdepth: 1
   :caption: 飞桨大模型特色技术

   飞桨大模型统一存储文档 Unified Checkpoint <llm/docs/unified_checkpoint.md>
   灵活注意力掩码 FlashMask <llm/docs/flashmask.md>
   飞桨大模型统一训练器 PaddleNLP Trainer <llm/docs/llm_trainer.rst>


.. toctree::
   :maxdepth: 1
   :caption: PaddleNLP工具库

   一键预测功能 <model_zoo/taskflow>

.. toctree::
   :maxdepth: 1
   :caption: PaddleNLP 教程

   Transformer预训练模型 <model_zoo/index>
   Trainer API训练教程 <trainer.md>
   对话模板教程 <get_started/chat_template>
   多轮对话精调教程 <llm/docs/chat_template.md>
   中文情感分析教程 <get_started/quick_start>
   模型压缩教程 <compression.md>
   数据蒸馏教程 <llm/application/distill/README.md>
   Torch2Paddle 权重转换教程 <llm/docs/torch2paddle.md>



.. toctree::
   :maxdepth: 1
   :caption: 评价指标

   评价指标 <metrics/metrics.md>



..    :maxdepth: 1
..    :caption: 数据准备

..    整体介绍 <data_prepare/overview>
..    数据集列表 <data_prepare/dataset_list>
..    加载数据集 <data_prepare/dataset_load>
..    自定义数据集 <data_prepare/dataset_self_defined>
..    数据处理 <data_prepare/data_preprocess>


.. toctree::
   :maxdepth: 1
   :caption: 实践教程

   AI Studio Notebook <tutorials/overview>

.. toctree::
   :maxdepth: 1
   :caption: 进阶指南

   模型压缩 <advanced_guide/model_compression/index>
   文本生成高性能加速 <advanced_guide/fastgeneration/index>
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
