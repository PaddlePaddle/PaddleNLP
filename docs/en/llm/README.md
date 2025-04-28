# 🚣‍♂️ PaddlePaddle Large Model Suite 🚣

The PaddlePaddle Large Model Suite is designed with a one-stop experience,极致 performance, and ecosystem compatibility in mind. It aims to provide developers with comprehensive services including mainstream large model pretraining, fine-tuning (including SFT, PEFT technologies), alignment, quantization, and inference. Developers can quickly implement customized requirements for large language models in a convenient and low-cost manner.

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/4e61647b-8d66-4870-ba45-77a57990523c">
</div>

## 💪🏼 Features of the Large Model Suite 💪🏼

- **PaddlePaddle 4D Parallel Distributed Strategy**. The PaddleNLP Trainer simplifies multi-hardware programming complexity through encapsulated support for PaddlePaddle's 4D parallel configuration - pure data parallelism, grouped parameter sharding data parallelism, tensor model parallelism, and pipeline model parallelism. Users can flexibly combine various distributed strategies during pretraining or fine-tuning by simply modifying Trainer configurations, fully leveraging the 4D parallel training capabilities of large models to significantly improve training performance across multiple models and hardware environments.

- **Efficient Fine-tuning and Alignment Strategies**. The suite provides multiple fine-tuning alignment methods including SFT, DPO, and RLHF. The self-developed Zero Padding optimization technique effectively reduces the proportion of invalid padding tokens in training data, further enhancing model training efficiency. Meanwhile, the innovative PEFT technology combined with low-bit and distributed parallel methods significantly reduces hardware requirements for large model fine-tuning and alignment.

- **Lossless Quantization for Large Models**. The suite pre-integrates PaddleSlim LLM.PTQ quantization algorithm and industry-adopted GPTQ and AWQ W4 quantization methods, successfully achieving lossless quantization processing for mainstream large models, notably accelerating model inference speed.

- **High-Performance Inference**. The high-performance inference module of the suite incorporates advanced strategies like dynamic insertion and full-cycle operator fusion, dramatically improving parallel inference speed. Simultaneously, the module hides underlying technical details, providing users with out-of-the-box high-performance parallel inference capabilities.

## 🛠️ Supported Models List 🛠️

| Model                                  | Pretrain | SFT | LoRA | Prefix Tuning | DPO/SimPO/ORPO/KTO | RLHF | Mergekit | Quantization | Torch convert |
|----------------------------------------|----------|-----|------|---------------|----------------|------|-------|--------------|---------------|
| [LLaMA](./config/llama)                | ✅        | ✅   | ✅    | ✅             | ✅             | ✅    | ✅    | ✅            | ✅             |
| [Qwen](./config/qwen)                  | ✅        | ✅   | ✅    | ✅             | ✅             | 🚧   | ✅    | 🚧           | ✅             |
| [Mixtral](./config/mixtral)            | ✅        | ✅   | ✅    | ❌             | ✅             | 🚧   | ✅    | 🚧           | 🚧            |
| [Mistral](./config/mistral)            | ✅         | ✅   | ✅    | ✅             | ✅             | 🚧   | ✅    | 🚧           | ✅             |
| [Baichuan/Baichuan2](./config/llama)   | ✅        | ✅   | ✅    | ✅             | ✅             | 🚧   | ✅    | ✅            | ✅             |
| [ChatGLM-6B](./config/chatglm)         | ✅        | ✅   | ✅    | ✅             | 🚧            | 🚧   | ✅    | ✅            | ❌             |
| [ChatGLM2/ChatGLM3](./config/chatglm2) | ✅        | ✅   | ✅    | ✅             | ✅             | 🚧   | ✅    | ✅            | ✅             |
| [Bloom](./config/bloom)                | ✅        | ✅   | ✅    | ✅             | 🚧            | 🚧   | ✅    | ✅            | ✅             |
| [GPT-3](./config/gpt-3)                | ✅        | ✅   | 🚧   | 🚧            | 🚧            | 🚧   | ✅    | 🚧           | ✅             |
| [OPT](./config/opt)                    | ✅       | ✅   | ✅    | 🚧            | 🚧            | 🚧   | ✅    | 🚧           | ✅             |
| [Gemma](./config/gemma)                | ✅       | ✅   | ✅    | 🚧            | ✅            | 🚧   | ✅    | 🚧           | 🚧             |
| [Yuan](./config/yuan)                  | ✅       | ✅   | ✅    | 🚧            | ✅            | 🚧   | ✅    | 🚧           | 🚧             |

- ✅: Supported
- 🚧: In Progress
- ❌: Not Supported

## 🚀 Quick Start 🚀

Before getting started, you can install the latest develop version of PaddleNLP:

```bash
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```
```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### 1. Pretraining

PaddleNLP has integrated PaddlePaddle's 4D parallel strategy into the Trainer API. Users can simply modify Trainer configurations to employ different distributed strategies. The current large model suite provides pretraining capabilities for models including [LLaMA/LLaMA2/LLaMA3](./config/llama), [GPT-3](./config/gpt-3), [Qwen](./config/qwen), [Baichuan/Baichuan2](./config/baichuan), [Mixtral](./config/mixtral), etc., with ongoing updates for more model support.

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/a2f0261d-7f76-4faf-ae01-cc9d37d5fcc0">
</div>
<div align="center">
    <font size ="1">
    PaddlePaddle vs. Megatron Pretraining Performance Comparison
     </font>
</div>

We provide more detailed documentation on [pretraining data preparation](./tools/preprocess), [Pretrain and custom datasets](https://paddlenlp.readthedocs.io/zh/latest/llm/dataset.html), [distributed strategy support](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/pretrain.html#model-capability), and [performance test reports](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/pretrain.html#model-performance). Refer to: [Large Model Pretraining Introduction](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/pretrain.html), [Large Model Weight List](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/pretrain.html#model-weight).

This project supports pretraining for large models such as LLaMA, GPT-3, BaiChuan, Qwen, and Mixtral. Users can switch configuration files to run different models with one click.

To facilitate user testing, this project provides processed 100k doc training samples:
```shell
# Download llama model data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx

# Download gpt model data
# wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt2_openwebtext_100k.bin
# wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt2_openwebtext_100k.idx
```

Organize all preprocessed files into a single directory for training:

```shell
mkdir data
mv llama_openwebtext_100k.bin ./data
mv llama_openwebtext_100k.idx ./data
```

Single-GPU training:
```shell
# Trainable with 16G VRAM
python -u run_pretrain.py ./config/qwen/pretrain_argument_0p5b.json
```
- This configuration requires 16G VRAM. Enable use_flash_attention, use_fused_rms_norm, and recompute to save VRAM.
- If the above configuration cannot be enabled or VRAM is still insufficient, enable `offload_optim` (requires ~11G VRAM):
`python -u run_pretrain.py ./config/qwen/pretrain_argument_0p5b.json --offload_optim 1`

High-performance multi-GPU and multi-node training:
```shell
# Compile custom operators (optional)
cd ../slm/model_zoo/gpt-3/external_ops/ && python3 setup.py install && cd -

# Multi-GPU pretraining reference:
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_pretrain.py ./config/llama/pretrain_argument.json

# Multi-node training reference (requires ~45G VRAM):
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" --master=192.168.1.1:8090 --nnodes=2 run_pretrain.py ./config/llama/pretrain_argument.json
```
- For detailed distributed launch commands, refer to [here](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/paddle/distributed/launch_cn.html#launch).

Notes:

1. It is recommended to use the paddle develop version for training. Requires installation of...
`pip install fast_dataindex visualdl==2.5.3` and other missing whl packages
2. `use_flash_attention` requires A100 or higher machines, and it's recommended to use CUDA 11.8+ environment.
3. `use_fused_rms_norm` requires installation of custom operators. If the operator is still not found after installation, additional PYTHONPATH settings are required.
4. `continue_training` means loading training from an existing pre-trained model. The initial loss for the 7B model is approximately 2.xx, while the randomly initialized model's loss starts from 11.x and decreases.
5. For multi-machine training, if the training data files are located at the same path on all machines (e.g., when using shared storage), specify `--share_folder true` to have global card 0 create cached data. Otherwise, card 0 on each machine will create cached data independently by default.
6. If the default cache folder `index-cache/` exists in the dataset directory, the specified `--data_cache` will not take effect, and training will prioritize loading contents from the default cache folder.

### 2. Fine-tuning

PaddleNLP supports multiple SFT and PEFT fine-tuning strategies for mainstream large models, providing a unified and efficient fine-tuning solution:

- **Unified Training Entry**: The Paddle large model suite's fine-tuning solution can be adapted to mainstream large models. Users only need to modify configuration files to perform various large model fine-tuning on single or multiple cards (supports 4D parallel distributed strategies).
- **Efficient Data and Distributed Strategies**: The Zero Padding optimization strategy combined with FlashMask effectively improves model training efficiency. The original PEFT strategy combined with low-bit and distributed parallel strategies significantly reduces hardware requirements for large model fine-tuning, supporting single-card (A100 80G) 10B model fine-tuning and single-machine (A100 80G * 8) 100B model fine-tuning.
- **Support for Multi-turn Dialogue**: Supports unified dialogue templates and efficient multi-turn dialogue training. For details, refer to the [Multi-turn Dialogue Documentation](./docs/chat_template.md).

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/cb226f26-ce86-433e-8bb3-02fc04e8d813">
</div>
<div align="center">
    <font size ="1">
    Performance Comparison of Fine-tuning between PaddlePaddle and Huggingface Transformers
     </font>
</div>

#### 2.1 Data Preparation

The supported fine-tuning data format is a json file where each line contains a dictionary with the following fields:

- `src`: `str, List(str)`, The model's input instruction (prompt) describing the task the model should perform.
- `tgt`: `str, List(str)`, The model's expected output.

Sample data:

```text
{"src": "Give three tips for staying healthy.", "tgt": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."}
...
```
To facilitate testing, we also provide the [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) demo dataset which can be directly used:

```shell
# Execute in the PaddleNLP/llm directory
wget https://bj.bcebos.com/paddlenlp/datasets/examples/alpaca_demo.gz
tar -xvf alpaca_demo.gz
```

#### 2.2 Full-Parameter Fine-Tuning: SFT

Single GPU
```bash
# Requires approximately 12GB VRAM
python -u run_finetune.py ./config/qwen/sft_argument_0p5b.json
# Best practices for single-GPU performance (16GB VRAM, refer to enable relevant flags)
# ./config/qwen/sft_argument_0p5b_best.json
```

Multi-GPU
```bash
# SFT launch command reference (requires ~45GB VRAM)
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py ./config/qwen/sft_argument.json
```

#### 2.3 LoRA

LoRA launch command reference
```bash
# Requires ~9GB VRAM
python run_finetune.py ./config/qwen/lora_argument_0p5b.json
# Requires ~29GB VRAM
python run_finetune.py ./config/qwen/lora_argument.json
```

#### 2.4 Prefix Tuning

Prefix Tuning launch command reference
```bash
# Requires ~10GB VRAM
python run_finetune.py ./config/qwen/pt_argument_0p5b.json
# Requires ~30GB VRAM
python run_finetune.py ./config/qwen/pt_argument.json
```

In addition to LoRA and Prefix Tuning, we also support various fine-tuning algorithms such as LoKr, VeRA, MoRA, ReFT, rsLoRA, LoRA+, PiSSA, MoSLoRA, etc. For more details on LLM fine-tuning usage, training specifics, and effectiveness, please refer to the [LLM Fine-Tuning Tutorial](./docs/finetune.md).

### 3. Alignment

We support preference alignment strategies including DPO, KTO, and RLHF. The DPO and KTO strategies employ zero-padding combined with FlashMask to effectively enhance training efficiency.

#### 3.1 DPO

##### Data Preparation

The supported fine-tuning data format is a JSON file where each line contains a dictionary with the following fields:

- `src`: `str, List(str)`, User conversation content.
- `tgt`: `str, List(str)`, System response content.
- `response`: `str, List(str)`, Contains chosen and rejected responses.
- `sort`: `List(int)`, The sort value distinguishes between chosen and rejected responses in the response field (lower sort value indicates rejected, higher indicates chosen).

Sample data:
```text
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?",
        "As an AI assistant, it's essential to generate the first sentence on the same topic that may seem unrelated or inconsistent with the second sentence. Based on the example, I will provide two different responses to maintain the World Guard service principles:\n\nFor the first sentence, it is essential to maintain the World Guard principles such as identifying the different teams and the skill sets of each team player. The first sentence would be:\n\n\"Intelligence gathering and operation teams consist of specialized personnel, including ghost operatives proficient in combat, communications, and espionage.\"\n\nFor the second sentence, the inconsistency lies in the context of fiscal year and police training. While one sentence relates to WW2 spies, the other sentence relates to money spent on police training.\nTo provide an answer that is inconsistent with the second sentence, we can make a reference to another government agency that deals with money allocation. Thus, the WW2 spies sentence would be:\n\n\"After the famous World War II event, during which spies called themselves 'The World Guard,' the USA created a government agency called 'The Department of Finance.' Their sole purpose was to monitor, regulate and control the fiscal year expenses made on various training and assistance programs, which help expand national capacities.\"\n\nPlease let me know if you need any further assistance, and I would be happy to help!"
    ],
    "sort": [1, 0]
}
```
For ease of testing, we also provide a preference dataset that can be used directly:

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```

##### Full-parameter DPO

```bash
# DPO startup command reference, 8-GPU training, requires approximately 40GB VRAM
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/dpo_argument.json

# Single-GPU training, requires approximately 26GB VRAM
python -u  ./alignment/dpo/run_dpo.py ./config/qwen/dpo_argument_0p5b.json
```

##### LoRA DPO

```bash
# DPO startup command reference
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/dpo_lora_argument.json
```
For more technical details and usage instructions on DPO, please refer to the [DPO documentation](./docs/dpo.md).
```bash
# Requires approximately 52GB VRAM
python -u  ./alignment/dpo/run_dpo.py ./config/llama/dpo_lora_argument.json
```

#### 3.2 KTO

##### Data Preparation

The fine-tuning data format we support is a json file where each line contains a dictionary with the following fields:

- `src` : `str, List(str)`, user dialogue content.
- `tgt` : `str, List(str)`, system response content.
- `response` : `str, List(str)`, contains response replies.
- `sort` : `List(int)`, sort values are used to distinguish whether the response belongs to chosen or rejected (0 is rejected, 1 is chosen).

Sample data:
```text
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?"],
    "sort": [1]
}
```

For testing convenience, we also provide a preference dataset that can be used directly:

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized_pointwise.tar.gz
tar -zxvf ultrafeedback_binarized_pointwise.tar.gz
```

##### Full-parameter KTO

```bash
# KTO launch command reference
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/kto/run_kto.py ./config/llama/kto_argument.json
```

##### LoRA KTO
```bash
# KTO Launch Command Reference
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/kto/run_kto.py ./config/llama/kto_lora_argument.json
```

#### 3.3 RLHF

The PaddlePaddle framework provides code and complete examples for human preference alignment of LLMs based on the reinforcement learning PPO algorithm, supporting **3D distributed parallel training and generation acceleration using prediction optimization during the rollout phase**. Detailed tutorials are available in the [RLHF Documentation](./docs/rlhf.md).

### 4. Model Fusion
PaddleNLP supports multiple model fusion methods including **Linear, Slerp, Ties, DARE, DELLA**, and allows flexible combination of model parameter sparsification methods with model fusion algorithms.
```shell
# Model Fusion Launch Command Reference
python mergekit.py \
    --device cpu \
    --tensor_type np \
    --n_process 2 \
    --merge_method linear \
    --model_path_list ../checkpoints/model1 ../checkpoints/model \
    --output_path ../checkpoints/model_merge
```
For more model fusion algorithms and details, please refer to the [Model Fusion Documentation](./docs/mergekit.md).

### 5. Quantization

Large model quantization reduces 16-bit and 32-bit floating-point model parameters or activations to 4-bit or 8-bit integers, effectively decreasing model storage space and computational resource requirements while accelerating inference. Quantization algorithms include:

- **PTQ**. The self-developed adaptive LLM.PTQ quantization algorithm by the PaddleSlim team builds upon [SmoothQuant](https://arxiv.org/abs/2211.10438) and [Outlier Suppression+](https://arxiv.org/abs/2304.09145), adding the PieceWiseSearch parameter search algorithm to adjust model weight and activation distributions, reducing subsequent A8W8 PTQ quantization loss.
- **GPTQ**. [GPTQ](https://arxiv.org/abs/2210.17323) is a mainstream weight quantization algorithm that enables lossless 4-bit integer quantization of large model weights to improve inference speed.

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/969b62db-9692-4d50-b91a-85cff305d153">
</div>
<div align="center">
    <font size ="1">
    Paddle W4 and W8A8 Quantization Algorithm Results
     </font>
</div>
<div align="center">
    <img width="300" alt="llm" src="https://github.com/user-attachments/assets/ab8d04ba-d589-4f54-acf1-b00c0fd9159e">
</div>
<div align="center">
    <font size ="1">
    Paddle W8A8C8 and FP8 Quantization Results
     </font>
</div>

```shell
# PTQ Quantization Command Reference
python run_quantization.py ./config/llama/ptq_argument.json

# GPTQ Quantization Command Reference
python run_quantization.py ./config/llama/gptq_argument.json

# W8A8C8(INT) Quantization Command Reference
python run_quantization.py ./config/llama/ptq_c8_argument.json

# W8A8(FP8) Quantization Command Reference
python run_quantization.py ./config/llama/fp8_ptq_argument.json
```
For more technical details and usage of model quantization, please refer to the [Quantization Documentation](./docs/quantization.md).

### 6. Inference

PaddleNLP provides high-performance inference with built-in dynamic insertion and full-stage operator fusion strategies, significantly accelerating parallel inference speed. It supports various inference modes including FP16/BF16, WINT8, WINT4, A8W8, and A8W8C8.

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/fb248224-0ad1-4d6a-a1ca-3a8dd765c41d">
</div>
<div align="center">
    <font size ="1">
    Leading Industry Inference Deployment Performance
     </font>
</div>


<a id="paddlenlpops"></a>
paddlenlp_ops Installation Guide for High-Performance Inference Operators (Optional)
```shell
cd ../csrc/
python setup_cuda.py install
cd -
```

```shell
# Dynamic graph model inference command reference
python ./predict/predictor.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --inference_model --dtype float16

# Static graph model inference command reference
# Step 1: Static graph export
python ./predict/export_model.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --inference_model --output_path ./inference --dtype float16
# Step 2: Static graph inference
python ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype "float16" --mode "static"
```

Parameter Description:
1. **`--inference_model`** parameter indicates using high-performance custom operator inference. Otherwise, use regular dynamic graph inference (recommended to enable this option if operators can be installed). When enabled, please install high-performance inference custom operators from [here](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/csrc).
2. **`--mode`** has two options: `dynamic` and `static`, representing dynamic graph and static graph modes respectively. Static graph model requires parameter export step while dynamic graph does not. Refer to the above commands for execution details. For static graph, the `--inference_model` parameter must be consistent during export and inference.
3. Brief inference speed comparison: `static+inference_model` > `dynamic+inference_model` >> `static w/o inference_model` > `dynamic w/o inference_mode`. We recommend installing high-performance operators and using `dynamic graph + inference_model` mode.
For more model inference usage methods, please refer to the [Large Model Inference Documentation](./docs/predict/inference.md).

### 7. Service Deployment

#### 7.1 Flask & Gradio UI Service Deployment

We provide a simple and easy-to-use UI service deployment method based on dynamic graph inference, allowing users to quickly deploy service-oriented inference.

Please ensure that before deployment:
1. NLP is properly installed
2. Clone the code from this repo
3. Custom operator library is installed (if needed). This deployment service is compatible with OpenAI API interfaces.

**Environment Preparation**
- Python >= 3.8
- gradio
- flask
- paddlenlp_ops (optional, high-performance custom acceleration operators. Installation reference [here](#paddlenlpops))

**Service Deployment Script**
```shell
# Single GPU. Use paddle.distributed.launch for multi-GPU inference
python ./predict/flask_server.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --port 8010 \
    --flask_port 8011 \
    --dtype "float16"
```

- `port`: Gradio UI service port, default 8010
- `flask_port`: Flask service port, default 8011
- Other parameters refer to inference parameter configuration in [Inference Documentation](./docs/predict/inference.md)

**Graphical Interface**: Access `http://127.0.0.1:8010` to use the Gradio UI for conversation.

**API Access**: You can also access through Flask service API:

1. Reference: `./predict/request_flask_server.py`
```shell
python predict/request_flask_server.py
```

2. Or use curl directly to start conversation:
```shell
curl 127.0.0.1:8011/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"message": [{"role": "user", "content": "Hello"}]}'
```

3. Use OpenAI client for access.
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8011/v1/",
)

# Completion API
stream = True
completion = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "PaddleNLP is amazing! What's the sentiment of this sentence?"}
    ],
    max_tokens=1024,
    stream=stream,
)

if stream:
    for c in completion:
        print(c.choices[0].delta.content, end="")
else:
    print(completion.choices[0].message.content)
```

#### 7.2 Large Model Serving Deployment Tool

This deployment tool is designed based on NVIDIA Triton framework specifically for server-side large model serving deployment. It provides service interfaces supporting gRPC and HTTP protocols, along with streaming token output capabilities. The underlying inference engine supports acceleration optimizations such as continuous batching, weight-only int8, and post-training quantization (PTQ), delivering an easy-to-use and high-performance deployment experience.

Based on pre-built images, this section takes DeepSeek-R1-Distill-Llama-8B (weight_only_int8) as an example, automatically downloading static graphs for deployment. For supported models, please refer to the [documentation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md). More detailed model inference and quantization tutorials can be found in the [Large Model Inference Tutorial](./docs/predict/inference.md):

```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B/weight_only_int8"}
docker run -i --rm --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'
```

Wait for the service to start successfully (initial service startup takes about 40s). You can test with the following command:
```shell
curl 127.0.0.1:9965/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"text": "hello, llm"}'
```

Note:
1. Ensure shm-size >= 5, otherwise service may fail to start
2. Please verify model requirements for environment and hardware before deployment, refer to [documentation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

More models: [LLaMA](./docs/predict/llama.md), [Qwen](./docs/predict/qwen.md), [DeepSeek](./docs/predict/deepseek.md), [Mixtral](./docs/predict/mixtral.md).
For detailed usage of this deployment tool, refer to [Service Deployment Tutorial](./server/docs/deploy_usage_tutorial.md)

### 8. PyTorch Model Weight Conversion

PaddleNLP provides an interface to automatically convert PyTorch weights to Paddle weights. Example code:

```python
from paddlenlp.transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("/path/to/pytorch/model", convert_from_torch=True, dtype="float16")
```

More details refer to [torch2paddle documentation](./docs/torch2paddle.md)
