Precision Alignment
==========================================

1. Overview
==========================================

1.1 Background
------------------------------------------
Model precision alignment is crucial for subsequent work, ensuring that the same model produces stable and consistent output results under identical environments and parameter configurations. This provides a solid foundation for subsequent data analysis, decision-making, and system optimization.

1.2 Prerequisites
------------------------------------------
Based on precision alignment acceptance criteria, the following preparations are recommended:

* Prepare training/validation datasets for model training and evaluation.
* Prepare PyTorch model architecture as the baseline for model precision.
* Prepare validation hardware: For fp16 model parameters, use V100, A100, etc. For bf16 parameters, use A100 or similar compute cards.

2. Workflow
==========================================
The overall workflow includes model structure alignment, small dataset preparation, initial forward alignment, loss function alignment, optimizer alignment, learning rate alignment, regularization strategy alignment, initial backward alignment, training data alignment, and training alignment. For large models using parallel strategies, additional steps include parallel model structure alignment, parallel forward alignment, and parallel backward alignment.

2.1 Process Overview
------------------------------------------
The overall workflow for model precision validation is shown below:

.. figure:: https://github.com/user-attachments/assets/e20aeed6-fc54-49ca-95c9-8e9863416796
  :width: 300px
  :alt: align_workflow
  :align: center


3. Model Alignment Process
==========================================

3.1 Model Structure Alignment
------------------------------------------

Three main steps for model structure alignment:

* Network structure code conversion
* Weight conversion
* Model architecture validation

3.1.1 Network Structure Code Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【Basic Process】

PyTorch APIs are generally similar to PaddlePaddle APIs. Refer to the `PyTorch Latest Release vs Paddle Develop API Mapping Table`_ for manual conversion of some network code.

.. _PyTorch Latest Release vs Paddle Develop API Mapping Table: https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html

【Automatic Code Conversion Tool】

`PaConvert Automatic Code Conversion Tool`
_ Can automatically convert code trained or inferred by other deep learning frameworks into PaddlePaddle code, facilitating quick and automated model code migration.

Currently only supports automatic conversion of PyTorch code. Support for other deep learning frameworks will be added later.
During conversion, we try to maintain the original code style and structure, converting API interfaces from other frameworks to PaddlePaddle APIs.

.. _PaConvert Code Auto-Conversion Tool : https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/convert_from_pytorch/paconvert_introduction_cn.html

【Large Model Network Structure Examples】

* Llama: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/llama/modeling.py
* Qwen2: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/qwen2/modeling.py

3.1.2 Weight Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【Basic Process】

After completing the network code conversion, model weights need to be converted.

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

The model structure needs to implement the _get_name_mapping method, which identifies parameters that need transposing in linear layers to adapt to Paddle's nn.Linear parameters. Refer to Qwen model structure:

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

3.1.3 Model Network Correctness Verification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【Basic Process】

1. Define PyTorch model, load weights, fix seed, generate random numbers based on numpy, convert to tensor processable by PyTorch, feed into network, obtain output.
2. Define PaddlePaddle model, load weights, fix seed, generate random numbers based on numpy, convert to tensor processable by PaddlePaddle, feed into network, obtain output.
3. Check diff; if below threshold, verification is successful.

【Example Code】

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

【Notes】

* When verifying forward alignment, call model.eval() to disable randomness in network components like BatchNorm and Dropout.
* For reproducibility, fix random seeds if random numbers are involved.
* Output diff can be calculated using np.max(np.abs(o1 - o2)). Generally, if diff <1e-5, forward pass is considered correct. If output diff is large, use binary search to locate the problematic operation.
* Set environment variables to avoid operator randomness:

.. code-block:: shell
    :linenos:

    # General environment variables
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_embedding_deterministic=1
    export FLAGS_cudnn_deterministic=1

    # Distributed training environment variables
    export Flags_mp_aysnc_allreduce=1
    export Flags_skip_mp_c_identity=1
    export FLAGS_shard_norm_align_dp=0
    export FLAGS_shard_use_reduce=1
    export FLAGS_sync_before_allreduce=1

3.1.4 Distributed Network Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

【Basic Process】

The basic process is similar to section 3.1.3. Additionally, during model initialization, create a distributed environment and use paddle.distributed.launch to start training. Example command:

.. code-block:: shell
    :linenos:

    python -m paddle.distributed.launch --devices 0,1 compare_torch_with_paddle.py

【Example Code】

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

        # Manual verification
        out_paddle = model_paddle(paddle_input_ids)[0]
        out_torch = model_torch(torch_input_ids, return_dict=False)[0]
        assert np.allclose(out_paddle.numpy(), out_torch.detach().numpy(), rtol=1e-5, atol=1e-4)

    eval_model_convert_parallel(mp_degree=2)

【Notes】

* Set environment variables to avoid operator randomness:

.. code-block:: shell
    :linenos:
    
    # General environment variables
    export NVIDIA_TF32_OVERRIDE=0
    export FLAGS_embedding_deterministic=1
    export FLAGS_cudnn_deterministic=1

    # Distributed training environment variables
    export Flags_mp_aysnc_allreduce=1
    export Flags_skip_mp_c_identity=1
    export FLAGS_shard_norm_align_dp=0
    export FLAGS_shard_use_reduce=1
    export FLAGS_sync_before_allreduce=1

3.2 Forward & Backward Alignment - Alignment Tool Verification
------------------------------------------

【Basic Process】

Instead of manual verification, use automated tool PaDiff for alignment. PaDiff is a model precision alignment tool between PaddlePaddle and PyTorch. It takes Paddle or Torch models, aligns intermediate training results and final weights, and reports where the first precision diff occurs.

PaDiff: https://github.com/PaddlePaddle/PaDiff

【Usage】

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

        # Manual verification
        out_paddle = model_paddle(paddle_input_ids)[0]
        out_torch = model_torch(torch_input_ids, return_dict=False)[0]
        assert np.allclose(out_paddle.numpy(), out_torch.detach().numpy(), rtol=1e-5, atol=1e-4)

        # Use padiff for verification
        inp = ({"input_ids": torch_input_ids, 
                "use_cache": False, 
                "output_attentions": False,
                "output_hidden_states": False,
                "return_dict": False}, 
            {"input_ids": paddle_input_ids})
        # diff_phase can be forward, backward or both
        auto_diff(model_torch, model_paddle, inp, atol=1e-4, rtol=1e3, auto_init=False, diff_phase="both", compare_mode="strict")

    eval_model_convert()

Precision alignment reference (verification standard):

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

3.3 Model Training Alignment
------------------------------------------

【Basic Process】

After completing previous steps, proceed to full-data training alignment:

1. Prepare train/eval data, data loaders, and model
2. Initialize model
3. Load configuration and start training to obtain final model and evaluation metrics.

【Notes】

#. 【Strongly Recommended】Complete backward alignment before training alignment. Uncertain factors include: dataset differences, framework discrepancies between PaddlePaddle and reference code in training mode, and initialization parameters.
#. During training alignment, some output differences are acceptable. For example, in SST-2 classification task, difference <0.15% is considered normal. Adjust diff_threshold in ReprodDiffHelper.report as needed.
#. Training fluctuations are normal. If final convergence differs, check:

  * Verify Dropout, BatchNorm, and other modules with hyperparameters.
  * Generate a pretrained model using reference code, convert to PaddlePaddle model, and compare convergence curves.
  * Use reference code's DataLoader output for training to exclude data loading effects.

References:

1. https://github.com/PaddlePaddle/PaDiff
2. https://github.com/PaddlePaddle/models/blob/release/2.2/docs/lwfx/ArticleReproduction_NLP.md