=====================================================
飞桨大模型统一训练器 PaddleNLP Trainer 训练教程
=====================================================


Trainer 特色能力简介
==========================

- **全策略分布式策略支持**

随着模型的复杂度越来越大，大规模分布式训练能力对LLM至关重要。
Trainer 提供了多种策略支持，支持从单卡、多卡数据并行，到 sharding 并行（stage1、2、3），到张量并行、流水线并行。做到了分布式训练的全策略支持。
Trainer 提供了简单易用的接口，可以轻松实现不同策略的训练。支持了混合精度训练、master weight/gradient、梯度累积等特性, 方便用户使用。


- **大模型统一断点存储支持**

大模型时代，模型使用张量并行、流水线并行等策略训练，需要将模型切分为多个部分存储。
与单模型存储格式不同，无法适配分布式切分策略改变，无法直接用于下游精调、推理等场景。
Trainer的 ``unified_checkpoint`` 技术 提供了统一断点存储接口，统一了分布式各种情况下模型存储格式，与单卡一致。
同时还支持了跨分布式策略断点续训，支持多机动态扩、缩容启动、支持异步保存等特性。



Trainer进阶分布式能力使用介绍
============================

本教程将以PaddleNLP中的LLaMA模型预训练为例，详细介绍 Trainer 使用。


**使用示例 TL:DR**


参数文档参见 https://paddlenlp.readthedocs.io/zh/latest/trainer.html

*关键配置项:*

.. code-block:: text

  --sharding "stage1"  --sharding_parallel_degree 2
        sharding 参数表示开启sharding功能。
        sharding_parallel_degree 表示sharding发生在多少路数据流之间。如果不想跨机sharding的话，可以设置为8

  --tensor_parallel_degree 2 
        表示张量并行 将一层 transformer 计算划分到几张卡上去计算

  --pipeline_parallel_degree 2 
        表示将transformer模型不同层划分为几块


注：

* 总卡数=sharding_parallel_dergee * tensor_parallel_dergee * pipeline_parallel_degree * data_parallel_degree
* data_parallel_degree 不需要传入参数设置，由 总卡数/(sharding_parallel_dergee * tensor_parallel_dergee * pipeline_parallel_degree) 计算得来 

.. code-block:: bash

    # 单机单卡
    python train.py

    # 单机(多机)多卡/数据并行
    paddle.distruted.launch --devices "0,1,2,3,4,5,6,7" train.py

    # 单机(多机)多卡/Sharding并行 
    paddle.distruted.launch --devices "0,1,2,3,4,5,6,7" train.py -sharding "stage2"

    # 单机(多机)多卡/Sharding并行 + 数据并行 (sharding4 dp2)
    paddle.distruted.launch --devices "0,1,2,3,4,5,6,7" train.py --sharding "stage2" --sharding_parallel_degree 4

    # 单机(多机)多卡/ 张量并行 TP8
    paddle.distruted.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 8

    # 单机(多机)多卡/ 张量并行+数据并行 TP4 DP2
    paddle.distruted.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 4

    # 单机(多机)多卡/ 张量并行+sharding并行 TP4 Sharding2
    paddle.distruted.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 4 \
        --sharding "stage1"  --sharding_parallel_degree 2

    # 单机(多机)多卡/ 张量并行+流水线并行 TP2 PP4
    paddle.distruted.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 4

    # 单机(多机)多卡/ 张量并行+流水线并行+sharding并行  TP2 PP2 Sharding2
    paddle.distruted.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --sharding "stage1"  --sharding_parallel_degree 2

    # 4D 并行，需要两机
    # 单机(多机)多卡/ 张量并行+流水线并行+sharding并行  TP2 PP2 Sharding2 DP2
    paddle.distruted.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --sharding "stage1"  --sharding_parallel_degree 2


Trainer 分布式能力
==================

功能特色：

* TP
   
  * 简单配置即可实现参数自动切分加载 合并
  * 组网改造简便，容易对齐精度

* PP 
   
  * 同时继承 PaddleNLP PertrainedModel
  * 模型参数自动加载，参数名映射到单卡模型。
  * Layer初始化参数全部config化，精简参数传递


通用分布式能力: DP + Sharding 
------------------------------

对于通用的分布式能力, PaddleNLP适配了数据并行data_parallel, 分布式参数sharding功能的支持。

用户使用 paddle.distruted.launch --devices "0,1,2,3" train.py即可将运行的程序切换为多卡数据并行. 如果想要使用sharding功能, 减少模型显存占用, 指定参数--sharding "stage2"即可. 更多sharding功能配置见参数介绍部分.

DP 或者sharding，这类功能无需用户修改组网, 直接多卡即可运行。目前已经支持PaddleNLP所有模型。


混合并行分布式能力: TP + PP 
------------------------------

飞桨4D并行, 即: ``data parallel`` + ``sharding parallel`` + ``tensor parallel`` + ``pipeline parallel`` .
混合并行这里, 主要添加了 ``tensor parallel`` (TP) 和 ``pipeline parallel`` (PP)支持. 

目前, PaddleNLP主要对一些大模型, 如 GPT, Llama(系列)，Qwen等做了 TP PP支持, 用户可以使用这些策略.相关代码实现可以参考llama训练的例子
流水线并行的组网改造可以参见modeling_pp.py当组网适配好 张量并行(TP), 流水线并行(PP)之后, 
用户使用 ``--tensor_parallel_degree`` 和 ``--pipeline_parallel_degree`` 即可启用混合并行训练.


张量并行如何接入、使用？
===========================

Tensor Parallel接入:
------------------------------

当前大模型接入 张量并行（TP） 主要有以下步骤

* 模型config配置
   
  * 此部分只需要配置一些默认参数，比如tensor_parallel_output之类的（是否合并最后TP计算出来的logits）

* 模型组网修改
  
  * 核心工作：主要修改的点有，

    i. Attention 模块 https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L363-L381
    ii. MLP模块 https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L320-L338
    iii. 词表模块 https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L655-L659
    iv. LMHead https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L875-L887
  
  * 此时修改较多，建议用户可以先修改 MLP模块 ，简单对齐之后，再去修改其他模块。参数转换对齐见后文。

*  参数切分自动转换mappings

  * 当我们修改了网络的时候，需要与单卡模型对齐，验证正确性。
  * 如llama代码，我们自提供了自动转换的接入函数，用户只需要配置 state_dict 中一些 linear 是 行切分或者列切分即可。 is_column 
  * `参考代码 <https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L565-L602>`_

.. image:: https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/1d6be372-e9de-4ec2-a8aa-705a4bafb097

* 对齐TP与单卡精度

  * 注意建议使用上文自动转换的mappinng配置，将极大减小工作量
  * 注意使用float32进行精度对齐，需要 export NVIDIA_TF32_OVERRIDE=0 关闭TF32


Tensor Parallel 使用
------------------------------

一般而言，对于TP单独使用的情况：
1. 只需要初始化分布式环境，获得 ``tp_degree`` ，``tp_rank`` 。
2. 然后传入到模型，即可完成模型初始化

加载的模型参数，会根据实际的 ``tp_degree`` ，``tp_rank`` ，自动将参数切分好，直接 运行 ``model.forward`` 可以做到与单卡一致的体验。

.. code-block:: python

    tp_degree = paddle.distributed.get_world_size()
    tp_rank = 0
    if tp_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tp_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        tp_rank = hcg.get_model_parallel_rank()

    # Load the pretrained language model.
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        tensor_parallel_degree=tp_degree,
        tensor_parallel_rank=tp_rank,
        dtype="float16", 
    )


流水线并行 (Pipeline Parallel) 如何接入、使用？
======================================================


Pipeline Parallel 接入
---------------------------

PP接入的本质是把模型写成一个 sequential 的形式，即模型之间的层是连续的不存在一些嵌套关系。我们实现了 PipelinePretrainedModel的模型基类。用户调用 add_sequential_layer即可添加模型一层。
从结果形式上而言就是把原来的模型LlamaForCausalLM 重写为 LlamaForCausalLMPipe

当前大模型接入 流水线并行（PP） 主要有以下步骤：

* 模型基类集成

  * 注意，模型需要同时继承 PipelinePretrainedModel 和 PipelineLayer
  * 模型的 config_class _get_tensor_parallel_mappings  _init_weights与原模型相同
  * `参考此处代码 <https://github.com/PaddlePaddle/PaddleNLP/blob/b5ca5bc767eddf2593839e47665e6b4abf2de91b/examples/language_model/llama/modeling_pp.py#L192-L202>`_ 

.. image:: https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/92b99bd6-90e4-45d0-8723-cf14fc258466


* 添加模型的层。

  * 模型layer 通过 LayerDesc 包裹
  * Layer的初始化，只接受模型config一个参数
  * add_sequential_layer 最后一个str参数是这一层模型，在原来网络中的前缀名

    i. 比如 embedding 层。原来在模型中是 llama.embeding.weight 这里的前缀是 llama
    ii. 后面的Decoder层，就是 llama.layers.0  llama.layers.1 之类
    iii. 此处的名字，可以将模型的命名结构映射到单卡

.. image:: https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/a511bc41-1ab3-414b-a076-09d17f06d94b
  

* 其他。配置一些其他选项，如：

  a. 指定切分pp的层
  b. virtual_pp
  c. 初始化权重

.. image:: https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/a1085022-d3c7-4b0c-9046-73af5a39231d


Pipeline Parallel 使用
------------------------

参见 `此处单测 <https://github.com/PaddlePaddle/PaddleNLP/blob/6c6e72bab2d5282df5a36d5e283f729fa89bccc6/examples/language_model/llama/tests/test_pipeline_parallel.py#L28-L67>`_ ， 使用LlamaForCausalLMPipe.from_pretrained 即可加载好模型。

.. code-block:: python

    world_size = paddle.distributed.get_world_size()
    pp_degree = world_size
    tp_degree = 1
    if world_size > 2:
        pp_degree = 2
        assert world_size % pp_degree == 0
        tp_degree = world_size // pp_degree

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tp_degree,
        "pp_degree": pp_degree,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()

    if pp_degree > 1:
        model_class = LlamaForCausalLMPipe
    else:
        model_class = LlamaForCausalLM

    model_name_or_path = "./llama-7b"
    model = model_class.from_pretrained(
        model_name_or_path,
        tensor_parallel_degree=tp_degree,
        tensor_parallel_rank=hcg.get_model_parallel_rank(),
        lm_shift_labels=True,
        tensor_parallel_output=False,
        # use_flash_attention=True,
    )

    model.eval()


    input_ids = paddle.to_tensor([[x for x in range(100, 110)]], dtype="int64")
    labels = paddle.to_tensor([[x for x in range(101, 111)]], dtype="int64")
    attention_mask = None

    if pp_degree > 1:
        pp_model = PipelineParallel(layers=model, hcg=hcg, strategy=strategy)
        ret = pp_model.eval_batch(data=[input_ids, labels], compute_loss=True)



附录并行能力简介
==================

* `数据并行 <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/data_parallel/index_cn.html>`_
* `sharding 并行 <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/group_sharded_parallel_cn.html#fenzuqiefenbingxing>`_ 
* `张量并行 <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/model_parallel_cn.html#zhangliangmoxingbingxing>`_
* `流水线并行 <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/pipeline_parallel_cn.html>`_