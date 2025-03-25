==========================================
精度对齐
==========================================

1. 总览
==========================================

1.1 背景
------------------------------------------
模型精度对齐是开展后续工作的关键，确保了相同模型在相同环境和相同参数配置下输出结果的稳定性和一致性，为后续的数据分析、决策制定以及系统优化提供了坚实的基础。

1.2 前序工作
------------------------------------------
基于精度对齐验收标准，建议准备以下内容：

* 准备好训练/验证数据集，用于模型训练与评估。
* 准备好PyTorch模型结构，作为模型精度baseline。
* 准备好验证设备，如模型参数为fp16，可准备V100、A100等计算卡，如模型参数为bf16，需准备A100等计算卡。

2. 整体流程
==========================================
整体流程包含模型结构对齐、准备小数据集、前向初次对齐、损失函数对齐、优化器对齐、学习率对齐、正则化策略对齐、反向初次对齐、训练集数据对齐和训练对齐。针对采用并行策略的大模型而言，分别增加了并行模型结构对齐、并行前向初次对齐和并行反向初次对齐。

2.1 流程概览
------------------------------------------
验证模型精度的整体流程如下图所示：

.. figure:: https://github.com/user-attachments/assets/e20aeed6-fc54-49ca-95c9-8e9863416796
  :width: 300px
  :alt: align_workflow
  :align: center


3. 模型对齐流程
==========================================

3.1 模型结构对齐
------------------------------------------

对齐模型结构时，一般有3个主要步骤：

* 网络结构代码转换
* 权重转换
* 模型组网正确性验证

3.1.1 网络结构代码转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【基本流程】

PyTorch的API和PaddlePaddle的API基本相似，
可以参考 `PyTorch最新release与Paddle develop API映射表`_ ，
部分组网代码也可手动转换。

.. _PyTorch最新release与Paddle develop API映射表 : https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html

【代码自动转换工具】

`代码自动转换工具PaConvert`_ 能自动将其它深度学习框架训练或推理的代码，转换为 PaddlePaddle 的代码，方便快速自动地 模型代码迁移。

目前仅支持自动转换 Pytorch 代码，其它深度学习框架的支持后续新增中，
转换时会尽量保持原代码的风格与结构，将其它深度学习框架的 API 接口 转换为 PaddlePaddle 的 API 接口。

.. _代码自动转换工具PaConvert : https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/convert_from_pytorch/paconvert_introduction_cn.html


【大模型网络结构示例】

* Llama: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/llama/modeling.py
* Qwen2: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/qwen2/modeling.py


3.1.2 权重转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【基本流程】

组网代码转换完成之后，需要对模型权重进行转换。

.. code-block:: python
    :linenos:

    import json
    import os
    import shutil
    import copy
    import paddle
    import torch
    from safetensors.torch import load_file
    from safetensors.numpy import save_file
    from paddlenlp.utils.log import logger
    from paddlenlp.transformers import Qwen2MoeForCausalLM, AutoConfig


    def execute_cmd(cmd, file_path):
        cmd = cmd + " " + file_path
        os.system(cmd)


    def convert_from_torch_to_paddle(torch_path=None, paddle_path=None, torch_prefix_key="model.", paddle_class=Qwen2MoeForCausalLM, delete_after_convert=False):
        assert torch_path is not None
        if paddle_path is None:
            paddle_path = torch_path + "-paddle"
        if not os.path.exists(paddle_path):
            os.mkdir(paddle_path)

        config = AutoConfig.from_pretrained(torch_path)
        name_mappings = paddle_class._get_name_mappings(config=config)

        torch_prefix_key = torch_prefix_key
        paddle_prefix_key = paddle_class.base_model_prefix + "."

        if os.path.exists(os.path.join(torch_path, "model.safetensors.index.json")):
            index = json.load(open(os.path.join(torch_path, "model.safetensors.index.json")))
            dst_index = copy.deepcopy(index)

            for key in list(dst_index["weight_map"].keys()):
                paddle_key = key.replace(torch_prefix_key, paddle_prefix_key)
                dst_index["weight_map"][paddle_key] = dst_index["weight_map"].pop(key)

            files = set(index["weight_map"].values())
            logger.info(files)

            for file_name in sorted(os.listdir(torch_path)):
                # skip hidden files
                if file_name.startswith("."):
                    continue

                logger.info(file_name)
                if file_name in files:
                    # convert safetensors to safetensors(paddle)
                    convert_safetensors_from_torch_to_paddle(file_name,
                                                            torch_path,
                                                            paddle_path,
                                                            torch_prefix_key,
                                                            paddle_prefix_key,
                                                            name_mappings,
                                                            delete_after_convert=False)
                else:
                    # copy config.json and other files
                    shutil.copy(os.path.join(torch_path, file_name), os.path.join(paddle_path, file_name))

            json.dump(dst_index, open(os.path.join(paddle_path, "model.safetensors.index.json"), "w"), indent=2)
        else:
            for file_name in sorted(os.listdir(torch_path)):
                # skip hidden files
                if file_name.startswith("."):
                    continue

                logger.info(file_name)
                if file_name == "model.safetensors":
                    convert_safetensors_from_torch_to_paddle(file_name,
                                                            torch_path,
                                                            paddle_path,
                                                            torch_prefix_key,
                                                            paddle_prefix_key,
                                                            name_mappings,
                                                            delete_after_convert=False)
                else:
                    # copy config.json and other files
                    shutil.copy(os.path.join(torch_path, file_name), os.path.join(paddle_path, file_name))

        execute_cmd(cmd="sed -i -e  's/torch_dtype/dtype/g' ",
                    file_path=os.path.join(paddle_path, "config.json"))

    def convert_safetensors_from_torch_to_paddle(file_name, torch_path, paddle_path, torch_prefix_key, paddle_prefix_key, name_mappings, delete_after_convert=False):
        tensors = load_file(os.path.join(torch_path, file_name))

        transpose_state_dict = {}
        for name_mapping in name_mappings:
            if name_mapping.action == "transpose":
                transpose_state_dict[name_mapping.target_name] = True
            else:
                transpose_state_dict[name_mapping.target_name] = False

        for key in list(tensors.keys()):
            paddle_key = key.replace(torch_prefix_key, paddle_prefix_key)
            logger.info("{} {}".format(key, tensors[key].shape))
            if transpose_state_dict[paddle_key]:
                t = tensors.pop(key).cuda().t().contiguous()
                capsule = torch.utils.dlpack.to_dlpack(t)
                t = paddle.utils.dlpack.from_dlpack(capsule)
                tensors[paddle_key] = t.numpy()
            else:
                t = tensors.pop(key).cuda()
                capsule = torch.utils.dlpack.to_dlpack(t)
                t = paddle.utils.dlpack.from_dlpack(capsule)
                tensors[paddle_key] = t.numpy()

                # tensors[dst_key] = paddle.to_tensor(tensors.pop(key).cuda().float().cpu().numpy(), dtype="bfloat16").numpy()
            logger.info("{} {}".format(paddle_key, tensors[paddle_key].shape))

        save_file(tensors, os.path.join(paddle_path, file_name), metadata={"format": "np"})
        if delete_after_convert:
            os.remove(os.path.join(torch_path, file_name))


    convert_from_paddle_to_torch(paddle_path="/root/code/PaddleNLP/ckpt/Qwen/Qwen2-0.5B"， paddle_class=Qwen2MoeForCausalLM)

其中，模型结构中需实现_get_name_mapping方法，在这个方法中会将线性层参数标识需要转置的参数，进而适配Paddle nn.Linear的参数。参考如Qwen模型结构：

https://github.com/PaddlePaddle/PaddleNLP/blob/0040a6068f56df27e0ae98e15f52d54eeb17058d/paddlenlp/transformers/qwen2/modeling.py#L732-L766

.. code-block:: python
    :linenos:

    class Qwen2PretrainedModel(PretrainedModel):
        @classmethod
        def _get_name_mappings(cls, config: Qwen2Config) -> list[StateDictNameMapping]:
            mappings: list[StateDictNameMapping] = []
            model_mappings = [
                ["embed_tokens.weight"],
                ["norm.weight"],
            ]
            for layer_index in range(config.num_hidden_layers):
                layer_mappings = [
                    [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.k_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.v_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.q_proj.bias", None],
                    [f"layers.{layer_index}.self_attn.k_proj.bias", None],
                    [f"layers.{layer_index}.self_attn.v_proj.bias", None],
                    [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                    [f"layers.{layer_index}.input_layernorm.weight"],
                    [f"layers.{layer_index}.post_attention_layernorm.weight"],
                ]
                model_mappings.extend(layer_mappings)

            init_name_mappings(mappings=model_mappings)
            # base-model prefix "Qwen2MoEModel"
            if "Qwen2Model" not in config.architectures:
                for mapping in model_mappings:
                    mapping[0] = "model." + mapping[0]
                    mapping[1] = "qwen2." + mapping[1]
                if not config.tie_word_embeddings:
                    model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

            mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
            return mappings

3.1.3 模型组网正确性验证
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【基本流程】

1. 定义PyTorch模型，加载权重，固定seed，基于numpy生成随机数，转换为PyTorch可以处理的tensor，送入网络，获取输出。
2. 定义PaddlePaddle模型，加载权重，固定seed，基于numpy生成随机数，转换为PaddlePaddle可以处理的tensor，送入网络，获取输出。
3. 排查diff，小于阈值，即可完成自测。

【示例代码】

.. code-block:: python
    :linenos:

    import numpy as np
    import paddle
    import torch
    from transformers import Qwen2Config as Qwen2Config_hf
    from transformers import Qwen2ForCausalLM as Qwen2ForCausalLM_hf

    from paddlenlp.transformers import Qwen2Config, Qwen2ForCausalLM

    def eval_model_convert():
        paddle_input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        torch_input_ids = torch.LongTensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

        # paddle model
        paddle_ckpt_path = "Qwen/Qwen2-0.5B"
        config_paddle = Qwen2Config.from_pretrained(paddle_ckpt_path)
        model_paddle = Qwen2ForCausalLM.from_pretrained(paddle_ckpt_path, config=config_paddle, dtype="float32")

        # torch model
        torch_ckpt_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-0___5B"
        config_torch = Qwen2Config_hf.from_pretrained(torch_ckpt_path, trust_remote_code=True)
        config_torch.dtype = "float32"
        model_torch = Qwen2ForCausalLM_hf.from_pretrained(torch_ckpt_path, config=config_torch, trust_remote_code=True)

        model_paddle.eval()
        model_torch.eval()
        
        out_paddle = model_paddle(paddle_input_ids)[0]
        out_torch = model_torch(torch_input_ids, return_dict=False)[0]

        assert np.allclose(out_paddle.numpy(), out_torch.detach().numpy(), rtol=1e-5, atol=1e-3)
        
    eval_model_convert()

【注意事项】

* 模型在前向对齐验证时，需要调用model.eval()方法，保证组网中的随机量被关闭，比如BatchNorm、Dropout等。
* 给定相同的输入数据，为保证可复现性，如果有随机数生成，固定相关的随机种子。
* 输出diff可以使用np.max(np.abs(o1 - o2))进行计算，一般小于1e-5的话，可以认为前向没有问题。如果最终输出结果diff较大，可以使用二分的方法进行排查，比如说BERT，包含1个embdding层、12个transformer-block以及最后的MLM head层，那么完成模型组网和权重转换之后，如果模型输出没有对齐，可以尝试输出中间某一个transformer-block的tensor进行对比，如果相同，则向后进行排查；如果不同，则继续向前进行排查，以此类推，直到找到导致没有对齐的操作。
* 在验证精度时需设置环境变量，避免算子的随机性，环境变量如下：

.. code-block:: shell
    :linenos:

    # 通用环境变量，避免随机性
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_embedding_deterministic=1
    export FLAGS_cudnn_deterministic=1

    # 并行计算环境变量，避免随机性
    export Flags_mp_aysnc_allreduce=1
    export Flags_skip_mp_c_identity=1
    export FLAGS_shard_norm_align_dp=0
    export FLAGS_shard_use_reduce=1
    export FLAGS_sync_before_allreduce=1

3.1.4 分布式组网对齐
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【基本流程】

基本流程同 3.1.3 模型组网正确性验证。此外，在模型初始化时，需创建分布式并行环境，并使用paddle.distributed.launch进行启动运行，示例命令如下：

.. code-block:: shell
    :linenos:

    python -m paddle.distributed.launch --devices 0,1 compare_torch_with_paddle.py

【示例代码】

.. code-block:: python
    :linenos:

    import numpy as np
    import paddle
    import torch
    from padiff import auto_diff
    from transformers import Qwen2Config as Qwen2Config_hf
    from transformers import Qwen2ForCausalLM as Qwen2ForCausalLM_hf
    from paddle.distributed import fleet
    from paddlenlp.transformers import Qwen2Config, Qwen2ForCausalLM

    def eval_model_convert_parallel(mp_degree=1):
        paddle_input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        torch_input_ids = torch.LongTensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": mp_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()

        # paddle model
        paddle_ckpt_path = "Qwen/Qwen2-0.5B"
        config_paddle = Qwen2Config.from_pretrained(paddle_ckpt_path)
        config_paddle.tensor_parallel_degree = hcg.get_model_parallel_world_size()
        config_paddle.tensor_parallel_rank = hcg.get_model_parallel_rank()
        config_paddle.tensor_parallel_output = False
        model_paddle = Qwen2ForCausalLM.from_pretrained(paddle_ckpt_path, config=config_paddle, dtype="float32")

        # torch model
        torch_ckpt_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-0___5B"
        config_torch = Qwen2Config_hf.from_pretrained(torch_ckpt_path, trust_remote_code=True)
        config_torch.dtype = "float32"
        model_torch = Qwen2ForCausalLM_hf.from_pretrained(torch_ckpt_path, config=config_torch, trust_remote_code=True)

        model_paddle.eval()
        model_torch.eval()

        # 手动验证
        out_paddle = model_paddle(paddle_input_ids)[0]
        out_torch = model_torch(torch_input_ids, return_dict=False)[0]
        assert np.allclose(out_paddle.numpy(), out_torch.detach().numpy(), rtol=1e-5, atol=1e-4)

    eval_model_convert_parallel(mp_degree=2)

【注意事项】

* 在验证精度时需设置环境变量，避免算子的随机性，环境变量如下：

.. code-block:: shell
    :linenos:
    
    # 通用环境变量，避免随机性
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_embedding_deterministic=1
    export FLAGS_cudnn_deterministic=1

    # 并行计算环境变量，避免随机性
    export Flags_mp_aysnc_allreduce=1
    export Flags_skip_mp_c_identity=1
    export FLAGS_shard_norm_align_dp=0
    export FLAGS_shard_use_reduce=1
    export FLAGS_sync_before_allreduce=1

3.2 前向对齐&反向对齐-对齐工具验证
------------------------------------------

【基本流程】

上述手动验证方式对开发者而言较为繁琐，可采用自动验证PaDiff进行验证。PaDiff 是基于 PaddlePaddle 与 PyTorch 的模型精度对齐工具。传入 Paddle 或 Torch 模型，对齐训练中间结果以及训练后的模型权重，并提示精度 diff 第一次出现的位置。

PaDiff: https://github.com/PaddlePaddle/PaDiff

【使用方式】

.. code-block:: python
    :linenos:

    import numpy as np
    import paddle
    import torch
    from padiff import auto_diff
    from transformers import Qwen2Config as Qwen2Config_hf
    from transformers import Qwen2ForCausalLM as Qwen2ForCausalLM_hf

    from paddlenlp.transformers import Qwen2Config, Qwen2ForCausalLM


    def eval_model_convert():
        paddle_input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        torch_input_ids = torch.LongTensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

        # paddle model
        paddle_ckpt_path = "Qwen/Qwen2-0.5B"
        config_paddle = Qwen2Config.from_pretrained(paddle_ckpt_path)
        model_paddle = Qwen2ForCausalLM.from_pretrained(paddle_ckpt_path, config=config_paddle, dtype="float32")

        # torch model
        torch_ckpt_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-0___5B"
        config_torch = Qwen2Config_hf.from_pretrained(torch_ckpt_path, trust_remote_code=True)
        config_torch.dtype = "float32"
        model_torch = Qwen2ForCausalLM_hf.from_pretrained(torch_ckpt_path, config=config_torch, trust_remote_code=True)

        model_paddle.eval()
        model_torch.eval()

        # 手动验证
        out_paddle = model_paddle(paddle_input_ids)[0]
        out_torch = model_torch(torch_input_ids, return_dict=False)[0]
        assert np.allclose(out_paddle.numpy(), out_torch.detach().numpy(), rtol=1e-5, atol=1e-4)

        # 使用padiff验证
        inp = ({"input_ids": torch_input_ids, 
                "use_cache": False, 
                "output_attentions": False,
                "output_hidden_states": False,
                "return_dict": False}, 
            {"input_ids": paddle_input_ids})
        # diff_phase 可以设置为forward，backword和both
        auto_diff(model_torch, model_paddle, inp, atol=1e-4, rtol=1e3, auto_init=False, diff_phase="both", compare_mode="strict")

    eval_model_convert()


精度对齐情况参考，可作为验证标准

+------------------+------+-----------------------+---------------------+-------------------------------------+
|      model       | size | logits diff (float32) | loss diff (float32) | each tensor in all layers (float32) |
+==================+======+=======================+=====================+=====================================+
| Qwen/Qwen2-0.5B  | 0.5B |         1e-4          |        1e-5         |                1e-4                 |
+------------------+------+-----------------------+---------------------+-------------------------------------+
| Qwen/Qwen2-1.5B  | 1.5B |         1e-3          |        1e-5         |                1e-3                 |
+------------------+------+-----------------------+---------------------+-------------------------------------+
|  Qwen/Qwen2-7B   |  7B  |         1e-3          |        1e-5         |                1e-3                 |
+------------------+------+-----------------------+---------------------+-------------------------------------+
| Qwen/Qwen1.5-14B | 14B  |         1e-4          |        1e-5         |                1e-4                 |
+------------------+------+-----------------------+---------------------+-------------------------------------+


3.3 模型训练对齐
------------------------------------------

【基本流程】

完成前面的步骤之后，就可以开始全量数据的训练对齐任务了。按照下面的步骤进行训练对齐。

1. 准备train/eval data, loader, model
2. 模型初始化
3. 加载配置，开始训练，迭代得到最终模型与评估指标。

【注意事项】

#. 【强烈】建议先做完反向对齐之后再进行模型训练对齐，二者之间的不确定量包括：数据集、PaddlePaddle与参考代码在模型training mode下的区别，初始化参数。
#. 在训练对齐过程中，受到较多随机量的影响，精度有少量diff是正常的，以SST-2数据集的分类为例，diff在0.15%以内可以认为是正常的，这里可以根据不同的任务，适当调整对齐检查的阈值(ReprodDiffHelper.report函数中的diff_threshold参数)。
#. 训练过程中的波动是正常的，如果最终收敛结果不一致，可以从以下方面进行排查：

  * 仔细排查Dropout、BatchNorm以及其他组网模块及超参是否无误。
  * 基于参考代码随机生成一份预训练模型，转化为PaddlePaddle的模型，并使用PaddlePaddle加载训练，对比二者的收敛曲线与最终结果，排查初始化影响。
  * 使用参考代码的Dataloader生成的数据，进行模型训练，排查train dataloader的影响。


参考文档:

1. https://github.com/PaddlePaddle/PaDiff
2. https://github.com/PaddlePaddle/models/blob/release/2.2/docs/lwfx/ArticleReproduction_NLP.md
