**简体中文**🀄 | [English🌎](./README_en.md)

<p align="center">
  <img src="https://user-images.githubusercontent.com/1371212/175816733-8ec25eb0-9af3-4380-9218-27c154518258.png" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleNLP?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleNLP?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleNLP?color=3af"></a>
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/dm/paddlenlp?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleNLP?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleNLP?color=ccf"></a>
</p>

<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#模型支持> 模型支持 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#社区交流> 社区交流 </a>
</h4>

**PaddleNLP**是一款基于飞桨深度学习框架的大语言模型(LLM)开发套件，支持在多种硬件上进行高效的大模型训练、无损压缩以及高性能推理。PaddleNLP 具备**简单易用**和**性能极致**的特点，致力于助力开发者实现高效的大模型产业级应用。

## News 📢

* **2024.06.27 [PaddleNLP v3.0 Beta](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0-beta0)**：拥抱大模型，体验全升级。统一大模型套件，实现国产计算芯片全流程接入；全面支持飞桨4D 并行配置、高效精调策略、高效对齐算法、高性能推理等大模型产业级应用流程；自研极致收敛的 RsLoRA+算法、自动扩缩容存储机制 Unified Checkpoint 和通用化支持的 FastFFN、FusedQKV 助力大模型训推；主流模型持续支持更新，提供高效解决方案。

* **2024.04.24 [PaddleNLP v2.8](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.8.0)**：自研极致收敛的 RsLoRA+算法，大幅提升 PEFT 训练收敛速度以及训练效果；引入高性能生成加速到 RLHF PPO 算法，打破 PPO 训练中生成速度瓶颈，PPO 训练性能大幅领先。通用化支持 FastFFN、FusedQKV 等多个大模型训练性能优化方式，大模型训练更快、更稳定。

* **2024.01.04 [PaddleNLP v2.7](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.7.1)**： 大模型体验全面升级，统一大模型入口。统一预训练、精调、压缩、推理以及部署等环节的实现代码，到 `PaddleNLP/llm`目录。全新[大模型套件文档](https://paddlenlp.readthedocs.io/zh/latest/)，一站式指引用户从大模型入门到业务部署上线。自动扩缩容存储机制 Unified Checkpoint，大大提高大模型存储的通用性。高效微调升级，支持了高效微调+LoRA 同时使用，支持了 QLoRA 等算法。

* **2023.08.15 [PaddleNLP v2.6](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.6.0)**： 发布[全流程大模型套件](./llm)，涵盖预训练，精调，压缩，推理以及部署等各个环节，为用户提供端到端的大模型方案和一站式的开发体验；内置[4D 并行分布式 Trainer](./docs/trainer.md)，[高效微调算法 LoRA/Prefix Tuning](./llm#33-lora), [自研 INT8/INT4量化算法](./llm#6-量化)等等；全面支持[LLaMA 1/2](./llm/config/llama), [BLOOM](./llm/config/bloom), [ChatGLM 1/2](./llm/config/chatglm), [OPT](./llm/config/opt)等主流大模型

## 特性

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleNLP/assets/15797489/983d1ee1-1acc-4e01-a341-557cfe43aec5" width="600">
</div>

### <a href=#多硬件训推一体> 🔧 多硬件训推一体 </a>

支持英伟达 GPU、昆仑 XPU、昇腾 NPU、燧原 GCU 和海光 DCU 等多个硬件的大模型训练和推理，套件接口支持硬件快速切换，大幅降低硬件切换研发成本。

### <a href=#高效易用的预训练> 🚀 高效易用的预训练 </a>

支持纯数据并行策略、分组参数切片的数据并行策略、张量模型并行策略和流水线模型并行策略的4D 高性能训练，Trainer 支持分布式策略配置化，降低复杂分布式组合带来的使用成本；
Unified Checkpoint 大模型存储格式在模型参数分布上支持动态扩缩容训练，降低硬件切换带来的迁移成本。

### <a href=#高效精调> 🤗 高效精调 </a>

精调算法深度结合零填充数据流和 FlashMask 高性能算子，降低训练无效数据填充和计算，大幅提升精调训练吞吐。

### <a href=#无损压缩和高性能推理> 🎛️ 无损压缩和高性能推理 </a>

大模型套件高性能推理模块内置动态插入和全环节算子融合策略，极大加快并行推理速度。底层实现细节封装化，实现开箱即用的高性能并行推理能力。

------------------------------------------------------------------------------------------

## 模型支持

* 模型参数已支持 LLaMA 系列、Baichuan 系列、Bloom 系列、ChatGLM 系列、Gemma 系列、Mistral 系列、OPT 系列和 Qwen 系列，详细列表👉【LLM】模型参数支持列表如下：

|                                        模型系列                                         | 模型名称                                                                                                                                                                                                                                                                                                                                                                                      |
|:---------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    [LLaMA](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)     | facebook/llama-7b, facebook/llama-13b, facebook/llama-30b, facebook/llama-65b                                                                                                                                                                                                                                                                                                                 |
|    [LLama2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)    | meta-llama/Llama-2-7b, meta-llama/Llama-2-7b-chat, meta-llama/Llama-2-13b, meta-llama/Llama-2-13b-chat, meta-llama/Llama-2-70b, meta-llama/Llama-2-70b-chat                                                                                                                                                                                                                                   |
|    [LLama3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)    | meta-llama/Meta-Llama-3-8B, meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-70B, meta-llama/Meta-Llama-3-70B-Instruct                                                                                                                                                                                                                                                            |
| [Baichuan](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/baichuan)  | baichuan-inc/Baichuan-7B, baichuan-inc/Baichuan-13B-Base, baichuan-inc/Baichuan-13B-Chat                                                                                                                                                                                                                                                                                                      |
| [Baichuan2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/baichuan) | baichuan-inc/Baichuan2-7B-Base, baichuan-inc/Baichuan2-7B-Chat, baichuan-inc/Baichuan2-13B-Base, baichuan-inc/Baichuan2-13B-Chat                                                                                                                                                                                                                                                              |
|    [Bloom](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/bloom)     | bigscience/bloom-560m, bigscience/bloom-560m-bf16, bigscience/bloom-1b1, bigscience/bloom-3b, bigscience/bloom-7b1, bigscience/bloomz-560m, bigscience/bloomz-1b1, bigscience/bloomz-3b, bigscience/bloomz-7b1-mt, bigscience/bloomz-7b1-p3, bigscience/bloomz-7b1, bellegroup/belle-7b-2m                                                                                                    |
|  [ChatGLM](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/chatglm/)  | THUDM/chatglm-6b, THUDM/chatglm-6b-v1.1                                                                                                                                                                                                                                                                                                                                                       |
| [ChatGLM2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/chatglm2)  | THUDM/chatglm2-6b                                                                                                                                                                                                                                                                                                                                                                             |
| [ChatGLM3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/chatglm2)  | THUDM/chatglm3-6b                                                                                                                                                                                                                                                                                                                                                                             |
|    [Gemma](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/gemma)     | google/gemma-7b, google/gemma-7b-it, google/gemma-2b, google/gemma-2b-it                                                                                                                                                                                                                                                                                                                      |
|  [Mistral](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/mistral)   | mistralai/Mistral-7B-Instruct-v0.3, mistralai/Mistral-7B-v0.1                                                                                                                                                                                                                                                                                                                                 |
|  [Mixtral](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/mixtral)   | mistralai/Mixtral-8x7B-Instruct-v0.1                                                                                                                                                                                                                                                                                                                                                          |
|      [OPT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/opt)       | facebook/opt-125m, facebook/opt-350m, facebook/opt-1.3b, facebook/opt-2.7b, facebook/opt-6.7b, facebook/opt-13b, facebook/opt-30b, facebook/opt-66b, facebook/opt-iml-1.3b, opt-iml-max-1.3b                                                                                                                                                                                                  |
|     [Qwen](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)     | qwen/qwen-7b, qwen/qwen-7b-chat, qwen/qwen-14b, qwen/qwen-14b-chat, qwen/qwen-72b, qwen/qwen-72b-chat,                                                                                                                                                                                                                                                                                        |
|   [Qwen1.5](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)    | Qwen/Qwen1.5-0.5B, Qwen/Qwen1.5-0.5B-Chat, Qwen/Qwen1.5-1.8B, Qwen/Qwen1.5-1.8B-Chat, Qwen/Qwen1.5-4B, Qwen/Qwen1.5-4B-Chat, Qwen/Qwen1.5-7B, Qwen/Qwen1.5-7B-Chat, Qwen/Qwen1.5-14B, Qwen/Qwen1.5-14B-Chat, Qwen/Qwen1.5-32B, Qwen/Qwen1.5-32B-Chat, Qwen/Qwen1.5-72B, Qwen/Qwen1.5-72B-Chat, Qwen/Qwen1.5-110B, Qwen/Qwen1.5-110B-Chat, Qwen/Qwen1.5-MoE-A2.7B, Qwen/Qwen1.5-MoE-A2.7B-Chat |
|    [Qwen2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)     | Qwen/Qwen2-0.5B, Qwen/Qwen2-0.5B-Instruct, Qwen/Qwen2-1.5B, Qwen/Qwen2-1.5B-Instruct, Qwen/Qwen2-7B, Qwen/Qwen2-7B-Instruct, Qwen/Qwen2-72B, Qwen/Qwen2-72B-Instruct, Qwen/Qwen2-57B-A14B, Qwen/Qwen2-57B-A14B-Instruct                                                                                                                                                                       |

* 4D 并行和算子优化已支持 LLaMA 系列、Baichuan 系列、Bloom 系列、ChatGLM 系列、Gemma 系列、Mistral 系列、OPT 系列和 Qwen 系列，【LLM】模型4D 并行和算子支持列表如下：


| 模型名称/并行能力支持 | 数据并行 | 张量模型并行 |          | 参数分片并行 |        |        | 流水线并行 |
|:---------------------:|:--------:|:------------:|:--------:|:------------:|:------:|:------:|:----------:|
|                       |          |   基础能力   | 序列并行 |    stage1    | stage2 | stage3 |            |
|         Llama         |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|        Llama2         |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|        Llama3         |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|         Qwen          |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|        Qwen1.5        |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|         Qwen2         |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|     Mixtral(moe)      |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     🚧     |
|        Mistral        |    ✅     |      ✅       |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|  Baichuan(同 llama)   |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|       Baichuan2       |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|        ChatGLM        |    ✅     |      ✅       |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|       ChatGLM2        |    ✅     |      🚧      |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|       ChatGLM3        |    ✅     |      🚧      |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|         Bloom         |    ✅     |      ✅       |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|      GPT-2/GPT-3      |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|          OPT          |    ✅     |      ✅       |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|         Gemma         |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |

* 大模型预训练、精调（包含 SFT、PEFT 技术）、对齐、量化已支持 LLaMA 系列、Baichuan 系列、Bloom 系列、ChatGLM 系列、Mistral 系列、OPT 系列和 Qwen 系列，【LLM】模型预训练、精调、对齐、量化支持列表如下：

| 模型名称/能力支持  | Pretrain | SFT | LoRA | Prefix Tuning | DPO | RLHF | Quantization | Torch convert |
|:------------------:|:--------:|:---:|:----:|:-------------:|:---:|:----:|:------------:|:-------------:|
|       LLaMA        |    ✅     |  ✅  |  ✅   |       ✅       |  ✅  |  ✅   |      ✅       |       ✅       |
|        Qwen        |    ✅     |  ✅  |  ✅   |       ✅       |  ✅  |  🚧  |      🚧      |       ✅       |
|      Mixtral       |    ✅     |  ✅  |  ✅   |       ❌       | 🚧  |  🚧  |      🚧      |      🚧       |
|      Mistral       |    ❌     |  ✅  |  ✅   |       ✅       |  ✅  |  🚧  |      🚧      |       ✅       |
| Baichuan/Baichuan2 |    ✅     |  ✅  |  ✅   |       ✅       |  ✅  |  🚧  |      ✅       |       ✅       |
|     ChatGLM-6B     |    ❌     |  ✅  |  ✅   |       ✅       | 🚧  |  🚧  |      ✅       |       ❌       |
| ChatGLM2/ChatGLM3  |    ❌     |  ✅  |  ✅   |       ✅       | 🚧  |  🚧  |      ✅       |       ✅       |
|       Bloom        |    ❌     |  ✅  |  ✅   |       ✅       | 🚧  |  🚧  |      ✅       |       ✅       |
|       GPT-3        |    ✅     |  ✅  |  🚧  |      🚧       | 🚧  |  🚧  |      🚧      |       ✅       |
|        OPT         |    🚧    |  ✅  |  ✅   |      🚧       | 🚧  |  🚧  |      🚧      |       ✅       |

------------------------------------------------------------------------------------------

## 安装

### 环境依赖

* python >= 3.8
* paddlepaddle >= 3.0.0b0

### pip 安装

```shell
pip install --upgrade paddlenlp==3.0.0b0
```

或者可通过以下命令安装最新 develop 分支代码：

```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

更多关于 PaddlePaddle 和 PaddleNLP 安装的详细教程请查看[Installation](./docs/get_started/installation.rst)。

------------------------------------------------------------------------------------------

## 快速开始

### 大模型文本生成

PaddleNLP 提供了方便易用的 Auto API，能够快速的加载模型和 Tokenizer。这里以使用 `Qwen/Qwen2-0.5B` 模型做文本生成为例：

```python
>>> from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype="float16")
>>> input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
>>> outputs = model.generate(**input_features, max_length=128)
>>> print(tokenizer.batch_decode(outputs[0]))
['我是一个AI语言模型，我可以回答各种问题，包括但不限于：天气、新闻、历史、文化、科学、教育、娱乐等。请问您有什么需要了解的吗？']
```

### 大模型预训练

```shell
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
cd .. # change folder to PaddleNLP/llm
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./config/llama/pretrain_argument.json
```

### 大模型 SFT 精调

```shell
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz && tar -zxvf AdvertiseGen.tar.gz
cd .. # change folder to PaddleNLP/llm
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/sft_argument.json
```

更多大模型全流程步骤，请参考[飞桨大模型套件](./llm)介绍。

更多 PaddleNLP 内容可参考：

* [精选模型库](./legacy/model_zoo)，包含优质预训练模型的端到端全流程使用。
* [多场景示例](./legacy/examples)，了解如何使用 PaddleNLP 解决 NLP 多种技术问题，包含基础技术、系统应用与拓展应用。
* [交互式教程](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)，在🆓免费算力平台 AI Studio 上快速学习 PaddleNLP。

------------------------------------------------------------------------------------------

## 社区交流

* 微信扫描二维码并填写问卷，即可加入交流群与众多社区开发者以及官方团队深度交流.

<div align="center">
    <img src="https://user-images.githubusercontent.com/11987277/245085922-0aa68d24-00ff-442e-9c53-2f1e898151ce.png" width="150" height="150" />
</div>

## Citation

如果 PaddleNLP 对您的研究有帮助，欢迎引用

```bibtex
@misc{=paddlenlp,
    title={PaddleNLP: An Easy-to-use and High Performance NLP Library},
    author={PaddleNLP Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleNLP}},
    year={2021}
}
```

## Acknowledge

我们借鉴了 Hugging Face 的[Transformers](https://github.com/huggingface/transformers)🤗关于预训练模型使用的优秀设计，在此对 Hugging Face 作者及其开源社区表示感谢。

## License

PaddleNLP 遵循[Apache-2.0开源协议](./LICENSE)。
