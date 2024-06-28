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

**PaddleNLP**是一款基于飞桨深度学习框架的大语言模型(LLM)开发套件，支持在多种硬件上进行高效的大模型训练、无损压缩以及高性能推理。PaddleNLP具备**简单易用**和**性能极致**的特点，致力于助力开发者实现高效的大模型产业级应用。

## News 📢

* **2024.06.27 [PaddleNLP v3.0 Beta](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0)**：拥抱大模型，体验全升级。统一大模型工具链，实现国产计算芯片全流程接入；全面支持飞桨4D并行配置、高效精调策略、高效对齐算法、高性能推理等大模型产业级应用流程；自研极致收敛的RsLoRA+算法、自动扩缩容存储机制Unified Checkpoint和通用化支持FastFFN、FusedQKV助力大模型训推；主流模型持续支持更新，提供高效解决方案。

* **2024.04.24 [PaddleNLP v2.8](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.8.0)**：自研极致收敛的RsLoRA+算法，大幅提升PEFT训练收敛速度以及训练效果；引入高性能生成加速到RLHF PPO算法，打破 PPO 训练中生成速度瓶颈，PPO训练性能大幅领先。通用化支持 FastFFN、FusedQKV等多个大模型训练性能优化方式，大模型训练更快、更稳定。

* **2024.01.04 [PaddleNLP v2.7](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.7.1)**： 大模型体验全面升级，统一工具链大模型入口。统一预训练、精调、压缩、推理以及部署等环节的实现代码，到 `PaddleNLP/llm`目录。全新[大模型工具链文档](https://paddlenlp.readthedocs.io/zh/latest/llm/finetune.html)，一站式指引用户从大模型入门到业务部署上线。自动扩缩容存储机制 Unified Checkpoint，大大提高大模型存储的通用性。高效微调升级，支持了高效微调+LoRA同时使用，支持了QLoRA等算法。

* **2023.08.15 [PaddleNLP v2.6](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.6.0)**： 发布[全流程大模型工具链](./llm)，涵盖预训练，精调，压缩，推理以及部署等各个环节，为用户提供端到端的大模型方案和一站式的开发体验；内置[4D并行分布式Trainer](./docs/trainer.md)，[高效微调算法LoRA/Prefix Tuning](./llm#33-lora), [自研INT8/INT4量化算法](./llm#6-量化)等等；全面支持[LLaMA 1/2](./llm/llama), [BLOOM](.llm/bloom), [ChatGLM 1/2](./llm/chatglm), [GLM](./llm/glm), [OPT](./llm/opt)等主流大模型


## 特性

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleNLP/assets/15797489/983d1ee1-1acc-4e01-a341-557cfe43aec5" width="600">
</div>

### <a href=#多硬件训推一体> 🔧 多硬件训推一体 </a>
支持英伟达GPU、昆仑XPU、昇腾NPU、燧原GCU和海光DCU等多个硬件的大模型训练和推理，套件接口支持硬件快速切换，大幅降低硬件切换研发成本。

### <a href=#高效易用的预训练> 🚀 高效易用的预训练 </a>
支持数据、分片、张量、流水线并行的4D高性能训练，Trainer支持分布式策略配置化，降低复杂分布式组合带来的使用成本；
Unified Checkpoint大模型存储格式在模型参数分布上支持动态扩缩容训练，降低硬件切换带来的迁移成本。

### <a href=#高效精调与高效对齐> 🤗 高效精调与高效对齐 </a>
精调和对齐算法深度结合零填充数据流和FlashMask高性能算子，降低训练无效数据填充和计算，大幅提升精调和对齐训练吞吐。

### <a href=#无损压缩和高性能推理> 🎛️ 无损压缩和高性能推理 </a>
大模型套件高性能推理模块内置动态插入和全环节算子融合策略，极大加快并行推理速度。底层实现细节封装化，实现开箱即用的高性能并行推理能力。

------------------------------------------------------------------------------------------

## 模型支持

| Model                                      | Pretrain | SFT | LoRA | Prefix Tuning | DPO | RLHF | Quantization | Weight convert |
|--------------------------------------------|:--------:|:---:|:----:|:-------------:|:---:|:----:|:------------:|:--------------:|
| [LLaMA](./llm/config/llama)                |    ✅     |  ✅  |  ✅   |       ✅       |  ✅  |  ✅   |      ✅       |       ✅        |
| [Qwen](./llm/config/qwen)                  |    ✅     |  ✅  |  ✅   |       ✅       |  ✅  |  🚧  |      🚧      |       ✅        |
| [Mixtral](./llm/config/mixtral)            |    ✅     |  ✅  |  ✅   |       ❌       | 🚧  |  🚧  |      🚧      |       🚧       |
| [Baichuan/Baichuan2](./llm/config/llama)   |    ✅     |  ✅  |  ✅   |       ✅       |  ✅  |  🚧  |      ✅       |       ✅        |
| [ChatGLM-6B](./llm/config/chatglm)         |    ❌     |  ✅  |  ✅   |       ✅       | 🚧  |  🚧  |      ✅       |       ❌        |
| [ChatGLM2/ChatGLM3](./llm/config/chatglm2) |    ❌     |  ✅  |  ✅   |       ✅       | 🚧  |  🚧  |      ✅       |       ✅        |
| [Bloom](./llm/config/bloom)                |    ❌     |  ✅  |  ✅   |       ✅       | 🚧  |  🚧  |      ✅       |       ✅        |
| [GPT-3](./llm/config/gpt-3)                |    ✅     |  ✅  |  🚧  |      🚧       | 🚧  |  🚧  |      🚧      |       ✅        |
| [OPT](./llm/config/opt)                    |    🚧    |  ✅  |  ✅   |      🚧       | 🚧  |  🚧  |      🚧      |       ✅        |

* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported

详细列表👉[模型参数支持](https://github.com/PaddlePaddle/PaddleNLP/issues/8663)

------------------------------------------------------------------------------------------

## 安装

### 环境依赖

- python >= 3.8
- paddlepaddle >= 3.0.0b0

### pip安装

```shell
pip install --upgrade paddlenlp
```

或者可通过以下命令安装最新 develop 分支代码：

```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

更多关于PaddlePaddle和PaddleNLP安装的详细教程请查看[Installation](./docs/get_started/installation.rst)。

------------------------------------------------------------------------------------------

## 快速开始

### 大模型文本生成

PaddleNLP提供了方便易用的Auto API，能够快速的加载模型和Tokenizer。这里以使用 `Qwen/Qwen2-0.5B` 模型做文本生成为例：

```python
>>> from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype="float16")
>>> input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
>>> outputs = model.generate(**input_features, max_length=128)
>>> tokenizer.batch_decode(outputs[0])
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

### 大模型SFT精调
```shell
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz && tar -zxvf AdvertiseGen.tar.gz
cd .. # change folder to PaddleNLP/llm
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/sft_argument.json
```

更多大模型全流程步骤，请参考[大模型全流程工具链](./llm)。

更多PaddleNLP内容可参考：
- [精选模型库](./legacy/model_zoo)，包含优质预训练模型的端到端全流程使用。
- [多场景示例](./legacy/examples)，了解如何使用PaddleNLP解决NLP多种技术问题，包含基础技术、系统应用与拓展应用。
- [交互式教程](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)，在🆓免费算力平台AI Studio上快速学习PaddleNLP。

------------------------------------------------------------------------------------------

## 社区交流

- 微信扫描二维码并填写问卷，回复小助手关键词（NLP）之后，即可加入交流群领取福利

  - 与众多社区开发者以及官方团队深度交流。
  - 10G重磅NLP学习大礼包！

<div align="center">
    <img src="https://user-images.githubusercontent.com/11987277/245085922-0aa68d24-00ff-442e-9c53-2f1e898151ce.png" width="150" height="150" />
</div>

## Citation

如果PaddleNLP对您的研究有帮助，欢迎引用

```
@misc{=paddlenlp,
    title={PaddleNLP: An Easy-to-use and High Performance NLP Library},
    author={PaddleNLP Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleNLP}},
    year={2021}
}
```

## Acknowledge

我们借鉴了Hugging Face的[Transformers](https://github.com/huggingface/transformers)🤗关于预训练模型使用的优秀设计，在此对Hugging Face作者及其开源社区表示感谢。

## License

PaddleNLP遵循[Apache-2.0开源协议](./LICENSE)。
