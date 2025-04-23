# Running llama2-13b Model on NPU with PaddleNLP
PaddleNLP has been deeply adapted and optimized for the Ascend NPU ([Learn Ascend](https://www.hiascend.com/zh/ecosystem/industry)) to run the llama2-13B model. This toolkit achieves near-seamless switching between Ascend NPU and GPU by maintaining consistent training and inference interfaces.

Technical highlights:
- **Full Training Strategy Adaptation** Supports 4D hybrid parallelism, flexibly adapting to various training strategies.
- **Extremely Optimized Training Performance** 95% of communication is overlapped with computation, with software-hardware co-design delivering ultimate performance.
- **Low-Barrier Performance Tuning** Automatic distributed strategy optimization capability shields hardware complexity while enabling users to easily explore computing limits.
- **Extremely Compressed Inference Cost** Supports layer-level operator fusion for inference, with fused operators now supporting dynamic insertion.

<!-- Performance image placeholder -->
<!-- <div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/da10e972-260c-4925-bf49-1e0aefd2a65c">
</div> -->

The diagram below shows the module dependency graph for running llama2-13b training and inference on NPU, which will clarify subsequent installation steps.
<!-- Training performance image placeholder -->

## 🚀 Quick Start 🚀

### (0) Before starting, you need an Ascend NPU machine with the following system requirements:

| Chip Type | Driver Version | CANN Version |
| --- | --- | --- |
| Ascend 910 | 23.0.3 | CANN 8.0.RC1 |

**Note: This example uses an 8-card machine and demonstrates the workflow through fine-tuning training + inference.**

**Note: To verify if your machine has Ascend 910B chips, simply run the following command in the system environment and check the output:**
```
lspci | grep d802

# Example: $ lspci | grep d802, output as follows
28:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
29:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
38:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
39:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
48:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
49:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
59:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
5a:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
98:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
99:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
b8:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
b9:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
c8:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
c9:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
d9:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
da:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
```

### (1) Environment Preparation: (This will take 5-15 minutes)
1. Pull the image
```# Note this image is for development environment only; the image does not contain precompiled PaddlePaddle packages
docker pull registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py39
```

2. Start the container with the following command. You can specify visible Ascend NPU cards for the container using ASCEND_RT_VISIBLE_DEVICES
```
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py39 /bin/bash
```

3. Install PaddlePaddle
```
# PaddlePaddle, the deep learning framework, provides fundamental computing capabilities
python -m pip install paddlepaddle==3.0.0b0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

4. Install paddleCustomDevice
```
# paddleCustomDevice is the custom hardware backend implementation for PaddlePaddle, providing NPU operator implementations.
python -m pip install paddle-custom-npu==3.0.0b0 -i https://www.paddlepaddle.org.cn/packages/stable/npu/
# For source compilation and installation, please refer to https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md
```

5. Clone PaddleNLP repository and install dependencies
```
# PaddleNLP is a natural language processing and large language model (LLM) development library based on PaddlePaddle, containing various large models implemented with PaddlePaddle, including the llama2-13B model. To better utilize PaddleNLP, you need to clone the entire repository.
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
python -m pip install -r requirements.txt
python -m pip install -e .
```

6. Install paddlenlp_ops
```
# The PaddleNLP repository includes built-in Ascend-specific fused operators to help users enjoy the ultimate compression of inference costs
```
cd csrc/npu
python setup.py build bdist_wheel
pip install dist/paddlenlp_ops-0.0.0-cp39-cp39-linux_x86_64.whl
cd -
```

### (2) Data Preparation: (This will take 2-5 minutes)
sft is the fine-tuning strategy. We provide an advertising generation dataset demo for your debugging:
```
# Fine-tuning: For testing convenience, we provide an advertising generation dataset ready to use:
cd llm/devices/npu/llama
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz
tar -zxvf AdvertiseGen.tar.gz
```
The supported fine-tuning data format is a json file where each line contains a dictionary with the following fields:
- `src`: `str, List(str)`, represents the model's input instruction (instruction), prompt, or task description.
- `tgt`: `str, List(str)`, represents the model's expected output.
Sample data:
```
{"src": "Type#Dress*Color#Blue*Style#Fresh*Pattern#Bowknot", "tgt": "The dress features 3D bowknot decorations with blue ribbon accents, creating a full, layered silhouette while adding a touch of sweetness. This design highlights the girl's fresh and charming demeanor."}
...
# You can prepare your own fine-tuning data following this format.
```

### (3) Training: (This will take approximately 4 hours)
We provide three entry scripts for Pretrain/SFT/LoRA in this directory, optimized with parallel strategies and configurations for 8x910 chip training resources. Detailed steps to launch fine-tuning:
```
# Run SFT strategy
bash llama_npu_sft_N1C8.sh
```

### (4) Inference: (This will take 10-15 minutes)
Before inference, prepare the configuration file. Under the merged parameter path (tutorial path: `./output/sft_bf16_llama_N1C8`), modify `config.json` as follows:
```
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
 eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 13824,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "num_key_value_heads": 40,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.31.0",
  "use_cache": false,
  "vocab_size": 32000
}
```

To ensure minimal inference costs with extreme compression, we employ static graph implementations. Therefore, it is necessary to export the static graph model from the dynamically trained model. Execute the following command to perform the export:

```
bash export_npu.sh ./output/sft_bf16_llama_N1C8/ ./inference
```

Finally, we execute inference through the static graph model:

```
# Execute inference code
bash predict_npu.sh ./inference
```

Upon successful execution, the generated inference results can be observed. A sample output is as follows:
"温故而知新" (wēn gù ér zhī xīn) is a Chinese idiom that means "to review the old and know the new." It is often used to describe the idea of gaining new insights through revisiting past knowledge.

The characters break down as:
- 温 (wēn): "to review/revisit"
- 故 (gù): "the old/past"
- 知 (zhī): "to know/understand"
- 新 (xīn): "the new"

This proverb originates from Confucius' *Analects* (《论语》): "温故而知新，可以为师矣" - "He who reviews old knowledge and gains new understanding can become a teacher."

The idiom emphasizes:
1. The value of historical knowledge
2. The process of deriving new understanding from existing foundations
3. The importance of continuous learning through reflection and synthesis

In modern contexts, it's used to advocate for:
- Learning from past experiences
- Re-examining historical data with fresh perspectives
- Combining traditional wisdom with innovative thinking
##  💪🏼 Features 💪🏼

- **Communication Hiding Technology**
When enabling tensor parallelism during model training, numerous communication (AllReduce/ReduceScatter/AllGather) + matrix multiplication (Matmul) operator combinations emerge. The 910 chip provides an efficient parallel mechanism to hide communication overhead.
<!-- Placeholder for principle diagram -->
By setting `FLAGS_NPU_MC2=1`, the communication-computation fusion pass is activated, effectively hiding the majority of tensor parallelism communication overhead within computations, thereby enhancing training performance.
<!-- Placeholder for performance chart -->
<!-- Placeholder for profiling chart -->

- **Unified Abstraction for Custom Operators**
In practice, we observe that fusion operators significantly impact large model training performance. To comprehensively support high-performance operators on Ascend and other hardware while maintaining concise network code, we provide a unified [custom operator interface implementation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/llama/fusion_ops.py). Current implementations cover common `fusion_rope`, `fusion_rms_norm`, and `fusion_flash_attention` for NPU, GPU, XPU, and GCU.

- **Layer-level Operator Fusion**
Layer-level operator fusion significantly enhances computational efficiency. The fused operators support dynamic insertion. During inference execution, the following logs can be observed:
```
--- Running IR pass [remove_residual_in_fused_bias_residual_layernorm]
--- Running IR pass [remove_residual_in_rms_norm]
--- Running IR pass [remove_blha_get_max_len]
--- Running IR pass [llama_fuse_attention_layer_begin]
--- Running IR pass [llama_fuse_attention_layer_end]
--- Running IR pass [llama_fuse_attention_layer]
--- Running IR pass [llama_fuse_lm_head_with_slice]
--- Running IR pass [llama_fuse_lm_head]
--- Running IR pass [llama_fuse_get_padding_offset]
```
