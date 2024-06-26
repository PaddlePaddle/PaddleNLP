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
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#特性> 特性 </a> |
  <a href=#社区交流> 社区交流 </a>
</h4>

**PaddleNLP**是一款基于飞桨深度学习框架的大语言模型(LLM)开发套件，支持在多种硬件上进行高效的大模型训练、无损压缩以及快速推理。PaddleNLP具备**简单易用**和**性能极致**的特点，致力于助力开发者实现高效的大模型产业级应用。

## News 📢

* **2024.06.27 [PaddleNLP v3.0 Beta](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0)**：拥抱大模型，体验全升级。统一大模型工具链，实现国产计算芯片全流程接入；全面支持飞桨4D并行配置、高效精调策略、高效对齐算法、高性能推理等大模型产业级应用流程；自研极致收敛的RsLoRA+算法、全断点存储机制Unified Checkpoint和通用化支持FastFNN、FusedQKV助力大模型训推；主流模型持续支持更新，提供高效解决方案。


* **2024.04.24 [PaddleNLP v2.8](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.8.0)**：自研极致收敛的RsLoRA+算法，大幅提升PEFT训练收敛速度以及训练效果；引入高性能生成加速到RLHF PPO算法，打破 PPO 训练中生成速度瓶颈，PPO训练性能大幅领先。通用化支持 FastFNN、FusedQKV等多个大模型训练性能优化方式，大模型训练更快、更稳定。

* **2024.01.04 [PaddleNLP v2.7](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.7.1)**： 大模型体验全面升级，统一工具链大模型入口。统一预训练、精调、压缩、推理以及部署等环节的实现代码，到 `PaddleNLP/llm`目录。全新[大模型工具链文档](https://paddlenlp.readthedocs.io/zh/latest/llm/finetune.html)，一站式指引用户从大模型入门到业务部署上线。全断点存储机制 Unified Checkpoint，大大提高大模型存储的通用性。高效微调升级，支持了高效微调+LoRA同时使用，支持了QLoRA等算法。

* **2023.08.15 [PaddleNLP v2.6](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.6.0)**： 发布[全流程大模型工具链](./llm)，涵盖预训练，精调，压缩，推理以及部署等各个环节，为用户提供端到端的大模型方案和一站式的开发体验；内置[4D并行分布式Trainer](./docs/trainer.md)，[高效微调算法LoRA/Prefix Tuning](./llm#33-lora), [自研INT8/INT4量化算法](./llm#6-量化)等等；全面支持[LLaMA 1/2](./llm/llama), [BLOOM](.llm/bloom), [ChatGLM 1/2](./llm/chatglm), [GLM](./llm/glm), [OPT](./llm/opt)等主流大模型


## 特性

<div align="center">
    <img src="https://private-user-images.githubusercontent.com/15797489/343082084-dafcd65a-c7b8-4458-9eaf-429fa81ce3d8.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTkzOTIxNjAsIm5iZiI6MTcxOTM5MTg2MCwicGF0aCI6Ii8xNTc5NzQ4OS8zNDMwODIwODQtZGFmY2Q2NWEtYzdiOC00NDU4LTllYWYtNDI5ZmE4MWNlM2Q4LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjI2VDA4NTEwMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM1ZDAwNzE1NGU3MDYwYTQ3ZWU2ZmM5Y2FiNGI2MGE5MDlmYjM1NDc2NmY4YmU3YTkzZjg1YjVkOTcxMzFlYjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.o639_JGZ51Zhiog-GQvgEjlNyWJ3WhuCMs4H8RwP8Tg" width="600">
</div>




### <a href=#多硬件训推一体> 🔧 多硬件训推一体 </a>
支持英伟达GPU、昆仑芯XPU、昇腾NPU、海光DCU、燧原GCU 等多个硬件的大模型训练和推理，同时套件接口支持硬件快速切换，大幅降低硬件切换的研发成本。

### <a href=#高效易用的预训练> 🚀 高效易用的预训练 </a>
支持数据、完全分片数据、张量、流水线并行的高性能训练，同时PaddleNLP Trainer支持分布式策略配置化，降低复杂分布式组合带来的使用成本；
PaddleNLP unified checkpoint大模型存储格式在模型参数分布上支持动态扩缩容训练，降低硬件切换带来的迁移成本。

### <a href=#高效精调与高效对齐> 🤗 高效精调与高效对齐 </a>
精调和对齐算法深度结合零填充数据流和FlashMask高性能算子，降低训练无效数据填充和无效数据计算，大幅提升精调和对齐训练吞吐。

### <a href=#无损压缩和高性能推理> 🎛️ 无损压缩和高性能推理 </a>
大模型套件高性能推理模块内置动态插入和全环节算子融合策略，极大加快并行推理的速度。同时隐藏了底层实现的细节，实现了开箱即用高性能并行推理能力。

------------------------------------------------------------------------------------------

## 安装

### 环境依赖

- python >= 3.8
- paddlepaddle >= 3.0beta

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

PaddleNLP提供了方便易用的Auto API，能够快速的加载模型和Tokenizer。这里以使用 `linly-ai/chinese-llama-2-7b` 大模型做文本生成为例：

```python
>>> from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("linly-ai/chinese-llama-2-7b")
>>> model = AutoModelForCausalLM.from_pretrained("linly-ai/chinese-llama-2-7b", dtype="float16")
>>> input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
>>> outputs = model.generate(**input_features, max_length=128)
>>> tokenizer.batch_decode(outputs[0])
['\n你好！我是一个AI语言模型，可以回答你的问题和提供帮助。']
```

### 一键UIE预测

PaddleNLP提供[一键预测功能](./docs/model_zoo/taskflow.md)，无需训练，直接输入数据即可开放域抽取结果。这里以信息抽取-命名实体识别任务，UIE模型为例：

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
>>> ie = Taskflow('information_extraction', schema=schema)
>>> pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))
[{'时间': [{'end': 6,
          'probability': 0.9857378532924486,
          'start': 0,
          'text': '2月8日上午'}],
  '赛事名称': [{'end': 23,
            'probability': 0.8503089953268272,
            'start': 6,
            'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
  '选手': [{'end': 31,
          'probability': 0.8981548639781138,
          'start': 28,
          'text': '谷爱凌'}]}]
```

更多PaddleNLP内容可参考：
- [大模型全流程工具链](./llm)，包含主流中文大模型的全流程方案。
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
