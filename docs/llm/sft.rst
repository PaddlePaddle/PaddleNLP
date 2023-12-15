==================
大模型微调SFT
==================

PaddleNLP大模型套件支持 LLaMA 、GPT-3、BaiChuan、Qwen 等大模型的指令微调SFT支持。

git clone 代码到本地，即可开始。

.. code-block:: bash

    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP 使用develop版本
    cd PaddleNLP/llm
    # 到达运行目录


精调训练数据格式
=====================

为了方便用户测试，我们也提供示例数据集 `广告生成数据集 <https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz>`_ ，用户也可以仿照数据集的格式制作自己的数据集进行精调。我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：


.. code-block:: text

    - `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
    - `tgt` : `str, List(str)`, 模型的输出。

样例数据：

.. code-block:: python

    {"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}



SFT
========================

SFT（Supervised Fine-Tuning）依托飞桨提出的 `4D混合分布式并行 <https://ai.baidu.com/forum/topic/show/987996>`_ 能力，
支持使用Trainer API轻松切换数据并行(DP)、 `张量并行(TP,Tensor Parallelism) <https://arxiv.org/abs/1909.08053>`_ 、
`流水线并行(PP,Pipeline Parallelism) <https://arxiv.org/abs/1811.06965>`_ （目前支持Llama、GPT-3、Qwen等模型）等多种分布式训练策略。

4D 混合并行策略的最佳配置实践如图下所示，在单机内使用通信量较大，适合使用机器内卡间通信的张量并行（张量并行又称模型并行，MP）和分组参数切片（Sharding）的2D组合策略；训练千亿规模模型时，叠加流水线并行策略使用多台机器共同分担；同时叠加数据并行来增加并发数量，提升训练速度。

.. image:: https://ai.bdstatic.com/file/63F5EBB1E188457ABAFD311CFC1D8658
    :width: 400
    :height: 350

.. code-block:: bash

    # 张量并行分布式训练（常用）
    python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_argument.json

    # 目前ChatGLM2、OPT不支持张量并行，默认使用Sharding策略（Paddle 2.5.1支持Sharding Stage2，Sharding Stage3需要使用Paddle develop版本）
    python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./chatglm2/sft_argument.json

    # 张量并行&流水线并行分布式训练（目前仅支持Llama）
    python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_pp_argument.json
