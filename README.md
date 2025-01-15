**简体中文**🀄 | [English🌎](./README_en.md)

<p align="center">
  <img src="https://user-images.githubusercontent.com/1371212/175816733-8ec25eb0-9af3-4380-9218-27c154518258.png" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="https://paddlenlp.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/paddlenlp/badge/?version=latest">
    <a href="https://github.com/PaddlePaddle/PaddleNLP/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleNLP?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleNLP?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleNLP?color=3af"></a>
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/dm/paddlenlp?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleNLP?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleNLP?color=ccf"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
</p>

<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#模型支持> 模型支持 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#社区交流> 社区交流 </a>
</h4>

**PaddleNLP**是一款基于飞桨深度学习框架的大语言模型(LLM)开发套件，支持在多种硬件上进行高效的大模型训练、无损压缩以及高性能推理。PaddleNLP 具备**简单易用**和**性能极致**的特点，致力于助力开发者实现高效的大模型产业级应用。

<a href="https://trendshift.io/repositories/2246" target="_blank"><img src="https://trendshift.io/api/badge/repositories/2246" alt="PaddlePaddle%2FPaddleNLP | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

## News 📢
* **2024.12.16 [PaddleNLP v3.0 Beta3](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0-beta3)**：大模型功能全新升级，新增了 Llama-3.2、DeepSeekV2模型，升级了 TokenizerFast，快速分词，重构了 SFTTrainer，一键开启 SFT 训练。此外，PaddleNLP 还支持了优化器状态的卸载和重载功能，实现了精细化的重新计算，训练性能提升7%。在 Unified Checkpoint 方面，进一步优化了异步保存逻辑，新增 Checkpoint 压缩功能，可节省78.5%存储空间。
最后，在大模型推理方面，升级 Append Attention，支持了 FP8量化，支持投机解码。

* **2024.12.13 📚《飞桨大模型套件 Unified Checkpoint 技术》**，加速模型存储95%，节省空间78%。支持全分布式策略调整自适应转换，提升模型训练的灵活性与可扩展性。训练-压缩-推理统一存储协议，无需手动转换提升全流程体验。Checkpoint 无损压缩结合异步保存，实现秒级存储并降低模型存储成本。适用于智能制造、指挥交通、医疗健康、金融服务等产业实际场景。12月24日（周二）19：00直播为您详细解读该技术如何优化大模型训练流程。报名链接：https://www.wjx.top/vm/huZkHn9.aspx?udsid=787976

* **2024.11.28 📚《FlashRAG-Paddle | 基于 PaddleNLP 的高效开发与评测 RAG 框架》**，为文本更快更好构建准确嵌入表示、加速推理生成速度。PaddleNLP 支持超大 Batch 嵌入表示学习与多硬件高性能推理，涵盖 INT8/INT4量化技术及多种高效注意力机制优化与 TensorCore 深度优化。内置全环节算子融合技术，使得 FlashRAG 推理性能相比 transformers 动态图提升70%以上，结合检索增强知识输出结果更加准确，带来敏捷高效的使用体验。直播时间：12月3日（周二）19：00。报名链接：https://www.wjx.top/vm/eaBa1vA.aspx?udsid=682361



<details><summary> <b>点击展开</b> </summary><div>

* **2024.08.08 📚《飞桨产业级大语言模型开发利器 PaddleNLP 3.0 重磅发布》**，训压推全流程贯通，主流模型全覆盖。大模型自动并行，千亿模型训推全流程开箱即用。提供产业级高性能精调与对齐解决方案，压缩推理领先，多硬件适配。覆盖产业级智能助手、内容创作、知识问答、关键信息抽取等应用场景。直播时间：8月22日（周四）19：00。报名链接：https://www.wjx.top/vm/Y2f7FFY.aspx?udsid=143844

* **2024.06.27 [PaddleNLP v3.0 Beta](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0-beta0)**：拥抱大模型，体验全升级。统一大模型套件，实现国产计算芯片全流程接入；全面支持飞桨4D 并行配置、高效精调策略、高效对齐算法、高性能推理等大模型产业级应用流程；自研极致收敛的 RsLoRA+算法、自动扩缩容存储机制 Unified Checkpoint 和通用化支持的 FastFFN、FusedQKV 助力大模型训推；主流模型持续支持更新，提供高效解决方案。

* **2024.04.24 [PaddleNLP v2.8](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.8.0)**：自研极致收敛的 RsLoRA+算法，大幅提升 PEFT 训练收敛速度以及训练效果；引入高性能生成加速到 RLHF PPO 算法，打破 PPO 训练中生成速度瓶颈，PPO 训练性能大幅领先。通用化支持 FastFFN、FusedQKV 等多个大模型训练性能优化方式，大模型训练更快、更稳定。
</div></details>

## 特性

### <a href=#多硬件训推一体> 🔧 多硬件训推一体 </a>

支持英伟达 GPU、昆仑 XPU、昇腾 NPU、燧原 GCU 和海光 DCU 等多个硬件的大模型和自然语言理解模型训练和推理，套件接口支持硬件快速切换，大幅降低硬件切换研发成本。
当前支持的自然语言理解模型：[多硬件自然语言理解模型列表](./docs/model_zoo/model_list_multy_device.md)

### <a href=#高效易用的预训练> 🚀 高效易用的预训练 </a>

支持纯数据并行策略、分组参数切片的数据并行策略、张量模型并行策略和流水线模型并行策略的4D 高性能训练，Trainer 支持分布式策略配置化，降低复杂分布式组合带来的使用成本；
[Unified Checkpoint 大模型存储工具](./llm/docs/unified_checkpoint.md)可以使得训练断点支持机器资源动态扩缩容恢复。此外，异步保存，模型存储可加速95%，Checkpoint 压缩，可节省78.5%存储空间。

### <a href=#高效精调> 🤗 高效精调 </a>

精调算法深度结合零填充数据流和 [FlashMask](./llm/docs/flashmask.md) 高性能算子，降低训练无效数据填充和计算，大幅提升精调训练吞吐。

### <a href=#无损压缩和高性能推理> 🎛️ 无损压缩和高性能推理 </a>

大模型套件高性能推理模块内置动态插入和全环节算子融合策略，极大加快并行推理速度。底层实现细节封装化，实现开箱即用的高性能并行推理能力。

## 文档
更多详细文档, 请访问 [PaddleNLP Documentation](https://paddlenlp.readthedocs.io/).

------------------------------------------------------------------------------------------

## 模型支持

* 模型参数已支持 LLaMA 系列、Baichuan 系列、Bloom 系列、ChatGLM 系列、Gemma 系列、Mistral 系列、OPT 系列和 Qwen 系列，详细列表👉【LLM】模型参数支持列表如下：

|                                          模型系列                                           | 模型名称                                                                                                                                                                                                                                                                                                                                                                                      |
|:-------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      [LLaMA](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)       | facebook/llama-7b, facebook/llama-13b, facebook/llama-30b, facebook/llama-65b                                                                                                                                                                                                                                                                                                                 |
|      [Llama2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)      | meta-llama/Llama-2-7b, meta-llama/Llama-2-7b-chat, meta-llama/Llama-2-13b, meta-llama/Llama-2-13b-chat, meta-llama/Llama-2-70b, meta-llama/Llama-2-70b-chat                                                                                                                                                                                                                                   |
|      [Llama3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)      | meta-llama/Meta-Llama-3-8B, meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-70B, meta-llama/Meta-Llama-3-70B-Instruct                                                                                                                                                                                                                                                            |
|     [Llama3.1](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)     | meta-llama/Meta-Llama-3.1-8B, meta-llama/Meta-Llama-3.1-8B-Instruct, meta-llama/Meta-Llama-3.1-70B, meta-llama/Meta-Llama-3.1-70B-Instruct, meta-llama/Meta-Llama-3.1-405B, meta-llama/Meta-Llama-3.1-405B-Instruct, meta-llama/Llama-Guard-3-8B                                                                                                                                              |
|     [Llama3.2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)     | meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-Guard-3-1B                                                                                                                                                                                                                                             |
|     [Llama3.3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)     | meta-llama/Llama-3.3-70B-Instruct                                                                                                                                                                                                                                                                                                                                                             |
|   [Baichuan](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/baichuan)    | baichuan-inc/Baichuan-7B, baichuan-inc/Baichuan-13B-Base, baichuan-inc/Baichuan-13B-Chat                                                                                                                                                                                                                                                                                                      |
|   [Baichuan2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/baichuan)   | baichuan-inc/Baichuan2-7B-Base, baichuan-inc/Baichuan2-7B-Chat, baichuan-inc/Baichuan2-13B-Base, baichuan-inc/Baichuan2-13B-Chat                                                                                                                                                                                                                                                              |
|      [Bloom](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/bloom)       | bigscience/bloom-560m, bigscience/bloom-560m-bf16, bigscience/bloom-1b1, bigscience/bloom-3b, bigscience/bloom-7b1, bigscience/bloomz-560m, bigscience/bloomz-1b1, bigscience/bloomz-3b, bigscience/bloomz-7b1-mt, bigscience/bloomz-7b1-p3, bigscience/bloomz-7b1, bellegroup/belle-7b-2m                                                                                                    |
|    [ChatGLM](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/chatglm/)    | THUDM/chatglm-6b, THUDM/chatglm-6b-v1.1                                                                                                                                                                                                                                                                                                                                                       |
|   [ChatGLM2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/chatglm2)    | THUDM/chatglm2-6b                                                                                                                                                                                                                                                                                                                                                                             |
|   [ChatGLM3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/chatglm2)    | THUDM/chatglm3-6b                                                                                                                                                                                                                                                                                                                                                                             |
| [DeepSeekV2](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/config/deepseek-v2) | deepseek-ai/DeepSeek-V2, deepseek-ai/DeepSeek-V2-Chat, deepseek-ai/DeepSeek-V2-Lite, deepseek-ai/DeepSeek-V2-Lite-Chat, deepseek-ai/DeepSeek-Coder-V2-Base, deepseek-ai/DeepSeek-Coder-V2-Instruct, deepseek-ai/DeepSeek-Coder-V2-Lite-Base, deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct                                                                                                      |
| [DeepSeekV3](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/config/deepseek-v2) | deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-V3-Base                                                                                                                                                                                                                                                                                                                                         |
|      [Gemma](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/gemma)       | google/gemma-7b, google/gemma-7b-it, google/gemma-2b, google/gemma-2b-it                                                                                                                                                                                                                                                                                                                      |
|    [Mistral](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/mistral)     | mistralai/Mistral-7B-Instruct-v0.3, mistralai/Mistral-7B-v0.1                                                                                                                                                                                                                                                                                                                                 |
|    [Mixtral](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/mixtral)     | mistralai/Mixtral-8x7B-Instruct-v0.1                                                                                                                                                                                                                                                                                                                                                          |
|        [OPT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/opt)         | facebook/opt-125m, facebook/opt-350m, facebook/opt-1.3b, facebook/opt-2.7b, facebook/opt-6.7b, facebook/opt-13b, facebook/opt-30b, facebook/opt-66b, facebook/opt-iml-1.3b, opt-iml-max-1.3b                                                                                                                                                                                                  |
|       [Qwen](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)       | qwen/qwen-7b, qwen/qwen-7b-chat, qwen/qwen-14b, qwen/qwen-14b-chat, qwen/qwen-72b, qwen/qwen-72b-chat,                                                                                                                                                                                                                                                                                        |
|     [Qwen1.5](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)      | Qwen/Qwen1.5-0.5B, Qwen/Qwen1.5-0.5B-Chat, Qwen/Qwen1.5-1.8B, Qwen/Qwen1.5-1.8B-Chat, Qwen/Qwen1.5-4B, Qwen/Qwen1.5-4B-Chat, Qwen/Qwen1.5-7B, Qwen/Qwen1.5-7B-Chat, Qwen/Qwen1.5-14B, Qwen/Qwen1.5-14B-Chat, Qwen/Qwen1.5-32B, Qwen/Qwen1.5-32B-Chat, Qwen/Qwen1.5-72B, Qwen/Qwen1.5-72B-Chat, Qwen/Qwen1.5-110B, Qwen/Qwen1.5-110B-Chat, Qwen/Qwen1.5-MoE-A2.7B, Qwen/Qwen1.5-MoE-A2.7B-Chat |
|      [Qwen2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)       | Qwen/Qwen2-0.5B, Qwen/Qwen2-0.5B-Instruct, Qwen/Qwen2-1.5B, Qwen/Qwen2-1.5B-Instruct, Qwen/Qwen2-7B, Qwen/Qwen2-7B-Instruct, Qwen/Qwen2-72B, Qwen/Qwen2-72B-Instruct, Qwen/Qwen2-57B-A14B, Qwen/Qwen2-57B-A14B-Instruct                                                                                                                                                                       |
|    [Qwen2-Math](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)    | Qwen/Qwen2-Math-1.5B, Qwen/Qwen2-Math-1.5B-Instruct, Qwen/Qwen2-Math-7B, Qwen/Qwen2-Math-7B-Instruct, Qwen/Qwen2-Math-72B, Qwen/Qwen2-Math-72B-Instruct, Qwen/Qwen2-Math-RM-72B                                                                                                                                                                                                               |
|     [Qwen2.5](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)      | Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B, Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B, Qwen/Qwen2.5-14B-Instruct, Qwen/Qwen2.5-32B, Qwen/Qwen2.5-32B-Instruct, Qwen/Qwen2.5-72B, Qwen/Qwen2.5-72B-Instruct                                                                     |
|   [Qwen2.5-Math](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)   | Qwen/Qwen2.5-Math-1.5B, Qwen/Qwen2.5-Math-1.5B-Instruct, Qwen/Qwen2.5-Math-7B, Qwen/Qwen2.5-Math-7B-Instruct, Qwen/Qwen2.5-Math-72B, Qwen/Qwen2.5-Math-72B-Instruct, Qwen/Qwen2.5-Math-RM-72B                                                                                                                                                                                                 |
|  [Qwen2.5-Coder](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen/)   | Qwen/Qwen2.5-Coder-1.5B, Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-7B, Qwen/Qwen2.5-Coder-7B-Instruct                                                                                                                                                                                                                                                                              |
|      [Yuan2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/yuan/)       | IEITYuan/Yuan2-2B, IEITYuan/Yuan2-51B, IEITYuan/Yuan2-102B                                                                                                                                                                                                                                                                                                                                    |

* 4D 并行和算子优化已支持 LLaMA 系列、Baichuan 系列、Bloom 系列、ChatGLM 系列、Gemma 系列、Mistral 系列、OPT 系列和 Qwen 系列，【LLM】模型4D 并行和算子支持列表如下：


| 模型名称/并行能力支持 | 数据并行 | 张量模型并行 |          | 参数分片并行 |        |        | 流水线并行 |
|:---------------------:|:--------:|:------------:|:--------:|:------------:|:------:|:------:|:----------:|
|                       |          |   基础能力   | 序列并行 |    stage1    | stage2 | stage3 |            |
|         Llama         |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|         Qwen          |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|        Qwen1.5        |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|         Qwen2         |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|     Mixtral(moe)      |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     🚧     |
|        Mistral        |    ✅     |      ✅       |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|       Baichuan        |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|       Baichuan2       |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|        ChatGLM        |    ✅     |      ✅       |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|       ChatGLM2        |    ✅     |      🚧      |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|       ChatGLM3        |    ✅     |      🚧      |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|         Bloom         |    ✅     |      ✅       |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|      GPT-2/GPT-3      |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|          OPT          |    ✅     |      ✅       |    🚧    |      ✅       |   ✅    |   ✅    |     🚧     |
|         Gemma         |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     ✅      |
|         Yuan2         |    ✅     |      ✅       |    ✅     |      ✅       |   ✅    |   ✅    |     🚧     |

* 大模型预训练、精调（包含 SFT、PEFT 技术）、对齐、量化已支持 LLaMA 系列、Baichuan 系列、Bloom 系列、ChatGLM 系列、Mistral 系列、OPT 系列和 Qwen 系列，【LLM】模型预训练、精调、对齐、量化支持列表如下：


| Model                                      | Pretrain | SFT | LoRA | FlashMask | Prefix Tuning | DPO/SimPO/ORPO/KTO | RLHF | Mergekit | Quantization |
|--------------------------------------------|:--------:|:---:|:----:|:---------:|:-------------:|:------------------:|:----:|:--------:|:------------:|
| [Llama](./llm/config/llama)                |    ✅     |  ✅  |  ✅   |     ✅     |       ✅       |         ✅          |  ✅   |    ✅     |      ✅       |
| [Qwen](./llm/config/qwen)                  |    ✅     |  ✅  |  ✅   |     ✅     |       ✅       |         ✅          |  🚧  |    ✅     |      🚧      |
| [Mixtral](./llm/config/mixtral)            |    ✅     |  ✅  |  ✅   |    🚧     |      🚧       |         ✅          |  🚧  |    ✅     |      🚧      |
| [Mistral](./llm/config/mistral)            |    ✅     |  ✅  |  ✅   |    🚧     |       ✅       |         ✅          |  🚧  |    ✅     |      🚧      |
| [Baichuan/Baichuan2](./llm/config/llama)   |    ✅     |  ✅  |  ✅   |     ✅     |       ✅       |         ✅          |  🚧  |    ✅     |      ✅       |
| [ChatGLM-6B](./llm/config/chatglm)         |    ✅     |  ✅  |  ✅   |    🚧     |       ✅       |         🚧         |  🚧  |    ✅     |      ✅       |
| [ChatGLM2/ChatGLM3](./llm/config/chatglm2) |    ✅     |  ✅  |  ✅   |    🚧     |       ✅       |         ✅          |  🚧  |    ✅     |      ✅       |
| [Bloom](./llm/config/bloom)                |    ✅     |  ✅  |  ✅   |    🚧     |       ✅       |         🚧         |  🚧  |    ✅     |      ✅       |
| [GPT-3](./llm/config/gpt-3)                |    ✅     |  ✅  |  🚧  |    🚧     |      🚧       |         🚧         |  🚧  |    ✅     |      🚧      |
| [OPT](./llm/config/opt)                    |    ✅     |  ✅  |  ✅   |    🚧     |      🚧       |         🚧         |  🚧  |    ✅     |      🚧      |
| [Gemma](./llm/config/gemma)                |    ✅     |  ✅  |  ✅   |    🚧     |      🚧       |         ✅          |  🚧  |    ✅     |      🚧      |
| [Yuan](./llm/config/yuan)                  |    ✅     |  ✅  |  ✅   |    🚧     |      🚧       |         ✅          |  🚧  |    ✅     |      🚧      |
* [大模型推理](./llm/docs/predict/inference.md)已支持 LLaMA 系列、Qwen 系列、Mistral 系列、ChatGLM 系列、Bloom 系列和 Baichuan 系列，支持 Weight Only INT8及 INT4推理，支持 WAC（权重、激活、Cache KV）进行 INT8、FP8量化的推理，【LLM】模型推理支持列表如下：

|          模型名称/量化类型支持           | FP16/BF16 | WINT8 | WINT4 | INT8-A8W8 | FP8-A8W8 | INT8-A8W8C8 |
|:----------------------------------------:|:---------:|:-----:|:-----:|:---------:|:--------:|:-----------:|
|   [LLaMA](./llm/docs/predict/llama.md)   |     ✅     |   ✅   |   ✅   |     ✅     |    ✅     |      ✅      |
|    [Qwen](./llm/docs/predict/qwen.md)    |     ✅     |   ✅   |   ✅   |     ✅     |    ✅     |      ✅      |
|  [Qwen-Moe](./llm/docs/predict/qwen.md)  |     ✅     |   ✅   |   ✅   |    🚧     |    🚧    |     🚧      |
| [Mixtral](./llm/docs/predict/mixtral.md) |     ✅     |   ✅   |   ✅   |    🚧     |    🚧    |     🚧      |
|                 ChatGLM                  |     ✅     |   ✅   |   ✅   |    🚧     |    🚧    |     🚧      |
|                  Bloom                   |     ✅     |   ✅   |   ✅   |    🚧     |    🚧    |     🚧      |
|                 BaiChuan                 |     ✅     |   ✅   |   ✅   |     ✅     |    ✅     |     🚧      |

## 安装

### 环境依赖

* python >= 3.8
* paddlepaddle >= 3.0.0b0

如果您尚未安装 PaddlePaddle，请参考 [飞桨官网](https://www.paddlepaddle.org.cn/) 进行安装。

### pip 安装

```shell
pip install --upgrade paddlenlp==3.0.0b3
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
>>> print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
['我是一个AI语言模型，我可以回答各种问题，包括但不限于：天气、新闻、历史、文化、科学、教育、娱乐等。请问您有什么需要了解的吗？']
```

### 大模型预训练

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP # 如已clone或下载PaddleNLP可跳过
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
cd .. # change folder to PaddleNLP/llm
# 如需使用use_fused_rms_norm=true，需要前往slm/model_zoo/gpt-3/external_ops安装fused_ln
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./config/llama/pretrain_argument.json --use_fused_rms_norm false
```

### 大模型 SFT 精调

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP # 如已clone或下载PaddleNLP可跳过
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz && tar -zxvf AdvertiseGen.tar.gz
cd .. # change folder to PaddleNLP/llm
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/sft_argument.json
```

更多大模型全流程步骤，请参考[飞桨大模型套件](./llm)介绍。
另外我们还提供了快速微调方式, 无需 clone 源代码：

```python
from paddlenlp.trl import SFTConfig, SFTTrainer
from datasets import load_dataset

dataset = load_dataset("ZHUI/alpaca_demo", split="train")

training_args = SFTConfig(output_dir="Qwen/Qwen2.5-0.5B-SFT", device="gpu")
trainer = SFTTrainer(
    args=training_args,
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
)
trainer.train()
```

更多 PaddleNLP 内容可参考：

* [精选模型库](./slm/model_zoo)，包含优质预训练模型的端到端全流程使用。
* [多场景示例](./slm/examples)，了解如何使用 PaddleNLP 解决 NLP 多种技术问题，包含基础技术、系统应用与拓展应用。
* [交互式教程](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)，在🆓免费算力平台 AI Studio 上快速学习 PaddleNLP。

------------------------------------------------------------------------------------------

## 社区交流

* 微信扫描二维码并填写问卷，即可加入交流群与众多社区开发者以及官方团队深度交流.

<div align="center">
    <img src="https://github.com/user-attachments/assets/3a58cc9f-69c7-4ccb-b6f5-73e966b8051a" width="150" height="150" />
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
