=====================================================
PaddleNLP Trainer for Large-Scale Model Training
=====================================================

Introduction to Key Features of Trainer
========================================

- **Comprehensive Distributed Strategy Support**

As model complexity continues to increase, large-scale distributed training capabilities become crucial for LLM development.
Trainer provides comprehensive support for various parallel strategies, including:
  - Single-GPU and multi-GPU data parallelism
  - Sharding parallelism (stage 1, 2, 3)
  - Tensor parallelism and pipeline parallelism

Trainer offers simple yet powerful APIs to implement different training strategies. It supports mixed-precision training, master weight/gradient management, and gradient accumulation for user convenience.

- **Unified Checkpoint Support for Large Models**

In the era of large models, distributed training with tensor parallelism and pipeline parallelism requires partitioned model storage.
Traditional single-model storage formats cannot adapt to changing distributed partitioning strategies and are unsuitable for downstream fine-tuning or inference scenarios.

Trainer's ``unified_checkpoint``
# Advanced Distributed Capabilities of Trainer
# ================================================================

This tutorial will use the LLaMA model pre-training in PaddleNLP as an example to explain the advanced usage of Trainer.

**Quick Start Example TL:DR**

Parameter documentation: https://paddlenlp.readthedocs.io/zh/latest/trainer.html

*Key Configuration Items:*

.. code-block:: text

  --sharding "stage1"  --sharding_parallel_degree 2
        sharding parameter enables sharding functionality.
        sharding_parallel_degree specifies the number of data streams for sharding. To avoid cross-machine sharding, set to 8

  --tensor_parallel_degree 2 
        Enables tensor parallelism by splitting transformer layer computations across multiple GPUs

  --pipeline_parallel_degree 2 
        Divides different layers of the transformer model into multiple stages

Note:

* Total GPUs = sharding_parallel_degree * tensor_parallel_degree * pipeline_parallel_degree * data_parallel_degree
* data_parallel_degree is automatically calculated as: total_GPUs / (sharding_parallel_degree * tensor_parallel_degree * pipeline_parallel_degree)

.. code-block:: bash

    # Single GPU
    python train.py

    # Single-node/Multi-node Multi-GPU/Data Parallel
    paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" train.py

    # Single-node/Multi-node Multi-GPU/Sharding Parallel 
    paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" train.py --sharding "stage2"

    # Sharding Parallel + Data Parallel (sharding4 dp2)
    paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" train.py --sharding "stage2" --sharding_parallel_degree 4

    # Tensor Parallel TP8
    paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 8

    # Tensor Parallel + Data Parallel TP4 DP2
    paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 4

    # Tensor Parallel + Sharding Parallel TP4 Sharding2
    paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 4 \
        --sharding "stage1"  --sharding_parallel_degree 2

    # Tensor Parallel + Pipeline Parallel TP2 PP4
    paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 4

    # Tensor Parallel + Pipeline Parallel + Sharding Parallel TP2 PP2 Sharding2
    paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --sharding "stage1"  --sharding_parallel_degree 2

    # 4D Parallel (Requires two nodes)
    # Tensor Parallel + Pipeline Parallel + Sharding Parallel TP2 PP2 Sharding2 DP2
    paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" train.py --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --sharding "stage1"  --sharding_parallel_degree 2

# Distributed Capabilities of Trainer
# ================================================================

Key Features:

* Tensor Parallelism (TP)
   
  - Automatic parameter partitioning/loading with simple configuration
  - Simplified network modification and precision alignment

* Pipeline Parallelism (PP)
   
  - Inherits PaddleNLP PretrainedModel
  - Automatic model parameter loading with name mapping to single-GPU model
  - Layer initialization parameters fully configurable

## General Distributed Capabilities: DP + Sharding
# -------------------------------------------------------------

For general distributed capabilities, PaddleNLP supports data parallelism (DP) and sharding:

- Users can run multi-GPU data parallel via `paddle.distributed.launch --devices "0,1,2,3" train.py`
- Enable sharding with `--sharding "stage2"` to reduce memory usage. More sharding configurations refer to parameter documentation.

DP and sharding require no network modification and support all PaddleNLP models.

## Hybrid Parallelism: TP + PP
# -------------------------------------------------------------

Paddle 4D Parallelism combines:

- Data Parallel (DP)
- Sharding
- Tensor Parallel (TP)
- Pipeline Parallel (PP)

Key Features:

- **Unified Checkpoint Interface**: Provides consistent model storage format across distributed scenarios, maintaining compatibility with single-GPU checkpoints
- **Elastic Training**: Supports resuming from checkpoints across different parallel strategies, dynamic cluster scaling, and asynchronous saving
- **Simplified Configuration**: Achieve hybrid parallelism through simple parameter settings without modifying network code

To enable 4D Parallel:

1. Configure tensor_parallel_degree and pipeline_parallel_degree
2. Set sharding_parallel_degree (optional)
3. Paddle automatically calculates data_parallel_degree based on total GPUs

Example Configuration:

```python
# 4D Parallel (TP=2, PP=2, Sharding=2, DP=2)
trainer = Trainer(
    tensor_parallel_degree=2,
    pipeline_parallel_degree=2,
    sharding_parallel_degree=2,
    # data_parallel_degree is auto-calculated as 8/(2*2*2) = 1 (not 2 as in example)
)
```

Note: The actual data_parallel_degree depends on total available GPUs and other parallelism configurations.
`data parallel` + `sharding parallel` + `tensor parallel` + `pipeline parallel`.

In hybrid parallel training, we mainly add support for `tensor parallel` (TP) and `pipeline parallel` (PP).

Currently, PaddleNLP primarily supports TP and PP strategies for large models such as GPT, Llama (series), Qwen, etc. Users can employ these strategies. The corresponding code implementations can be referred to in the Llama training examples.

The network modifications for pipeline parallel can be found in modeling_pp.py. After adapting the network for tensor parallel (TP) and pipeline parallel (PP), users can enable hybrid parallel training using `--tensor_parallel_degree` and `--pipeline_parallel_degree`.

How to Use Tensor Parallel?
===========================

Tensor Parallel Integration:
------------------------------

The current process for integrating Tensor Parallel (TP) into large models consists of the following steps:

* Model config configuration

  * This only requires setting some default parameters, such as tensor_parallel_output (whether to merge the final TP-computed logits)

* Model network modification

  * Core modifications mainly include:

    i. Attention module: [Llama Attention Code](https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L363-L381)  
    ii. MLP module: [Llama MLP Code](https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L320-L338)  
    iii. Embedding module: [Llama Embedding Code](https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L655-L659)  
    iv. LMHead: [Llama LMHead Code](https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L875-L887)  

  * Since this involves significant modifications, it's recommended to start with the MLP module for basic alignment before modifying other modules. Parameter conversion alignment is discussed below.

* Automatic parameter sharding conversion mappings

  * When modifying the network, we need to validate consistency with the single-card model.
  * As shown in the Llama code, we provide automatic conversion interface functions. Users only need to configure whether linears in the state_dict are row-partitioned or column-partitioned (is_column).
Reference Code <https://github.com/PaddlePaddle/PaddleNLP/blob/acfd537f3c859d80bf5d1f0a2fb26f485ef015b5/paddlenlp/transformers/llama/modeling.py#L565-L602>_

.. image:: https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/1d6be372-e9de-4ec2-a8aa-705a4bafb097

* Align TP with single-card precision

  * Note: It is recommended to use the automatically converted mapping configuration mentioned above, which will significantly reduce workload
  * Note: Use float32 for precision alignment. Need to export NVIDIA_TF32_OVERRIDE=0 to disable TF32

Tensor Parallel Usage
------------------------------

In general, for standalone TP usage:
1. Just initialize the distributed environment to obtain ``tp_degree`` and ``tp_rank``
2. Then pass them to the model to complete model initialization

The loaded model parameters will be automatically partitioned according to the actual ``tp_degree`` and ``tp_rank``. Directly run ``model.forward``
It can provide an experience consistent with single-card training.

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


How to Integrate and Use Pipeline Parallel (PP)?
================================================


Integration of Pipeline Parallel
--------------------------------

The essence of PP integration is to restructure the model into a sequential form, where layers are continuous without nested relationships. We implemented the PipelinePretrainedModel base class. Users can add model layers by calling add_sequential_layer. In practical terms, this means rewriting the original model (e.g., LlamaForCausalLM) into a pipelined version (e.g., LlamaForCausalLMPipe).

The main steps for integrating Pipeline Parallel (PP) into large models are:

* Model Base Class Integration

  * Note: The model should inherit from both PipelinePretrainedModel and PipelineLayer
  * The model's config_class, _get_tensor_parallel_mappings, and _init_weights should remain consistent with the original model
  * `Refer to this code <https://github.com/PaddlePaddle/PaddleNLP/blob/b5ca5bc767eddf2593839e47665e6b4abf2de91b/examples/language_model/llama/modeling_pp.py#L192-L202>`_

.. image::
* Adding model layers.

  * Model layers are wrapped with LayerDesc
  * Layer initialization only accepts the model config as single parameter
  * The last str parameter of add_sequential_layer is the prefix name of this layer in the original network

    i. For example, the embedding layer. In the original model it's named like llama.embedding.weight, where the prefix is llama
    ii. Subsequent Decoder layers would be named like llama.layers.0, llama.layers.1, etc.
    iii. This naming allows mapping the model's naming structure to single-card configuration

.. image:: https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/a511bc41-1ab3-414b-a076-09d17f06d94b

* Other configurations. Configure additional options such as:

  a. Specify layers for pipeline parallelism
  b. Virtual pipeline parallelism
  c. Weight initialization

.. image:: https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/a1085022-d3c7-4b0c-9046-73af5a39231d

Using Pipeline Parallelism
--------------------------

See `unit test example <https://github.com/PaddlePaddle/PaddleNLP/blob/6c6e72bab2d5282df5a36d5e283f729fa89bccc6/examples/language_model/llama/tests/test_pipeline_parallel.py#L28-L67>`
, use `LlamaForCausalLMPipe.from_pretrained` to load the model directly.

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

Appendix Parallel Capability Introduction
==========================================

*
`Data Parallelism <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/data_parallel/index_cn.html>`_
* `Sharding Parallelism <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/group_sharded_parallel_cn.html#fenzuqiefenbingxing>`_
* `Tensor Parallelism <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/model_parallel_cn.html#zhangliangmoxingbingxing>`_
* `Pipeline Parallelism <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/pipeline_parallel_cn.html>`_