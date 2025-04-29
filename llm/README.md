# 🚣‍♂️ 飞桨大模型套件 🚣

飞桨大模型套件以一站式体验、极致性能和生态兼容性为设计理念，致力于为开发者提供业界主流的大模型预训练、精调（包含 SFT、PEFT 技术）、对齐、量化和推理等全方位服务。开发者能够以便捷、低成本的方式快速实现大语言模型的定制化需求。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/4e61647b-8d66-4870-ba45-77a57990523c">
</div>

## 💪🏼 大模型套件特色 💪🏼

- **飞桨4D 并行分布式策略**。 PaddleNLP Trainer 通过封装支持飞桨的4D 并行配置，即纯数据并行策略、分组参数切片的数据并行策略、张量模型并行策略和流水线模型并行策略，从而简化了多硬件编程的复杂性。用户仅需通过修改 Trainer 的配置，就能灵活组合各种预训练或精调过程中的分布式策略，充分发挥大模型的4D 并行训练能力，进而在多模型、多硬件环境下显著提升训练性能。
- **高效精调对齐策略**。飞桨大模型套件提供了包括 SFT、DPO、RLHF 在内的多种精调对齐方法。套件中自研的 Zero Padding 零填充优化技术，有效降低了训练数据中无效填充标记（pad token）的比例，进一步增强了模型训练的效率。同时，独创的 PEFT 技术结合低比特和分布式并行方法，显著降低了进行大模型精调对齐的硬件要求。
- **大模型无损量化**。大模型套件预先集成了 PaddleSlim LLM.PTQ 量化算法，以及业界广泛采用的 GPTQ 和 AWQ 的 W4量化方法，成功实现了对主流大模型的无损量化处理，显著加速了模型的推理速度。
- **高性能推理**。大模型套件的高性能推理模块内置了动态插入和全环节算子融合的高级策略，这极大地提升了并行推理的速度。同时，该模块隐藏了底层技术细节，为用户提供了开箱即用的高性能并行推理功能。


## 🛠️ 支持模型列表 🛠️

| Model                                  | Pretrain | SFT | LoRA | Prefix Tuning | DPO/SimPO/ORPO/KTO | RLHF | Mergekit | Quantization | Torch convert |
|----------------------------------------|----------|-----|------|---------------|----------------|------|-------|--------------|---------------|
| [LLaMA](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)                | ✅        | ✅   | ✅    | ✅             | ✅             | ✅    | ✅    | ✅            | ✅             |
| [Qwen](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen)                  | ✅        | ✅   | ✅    | ✅             | ✅             | 🚧   | ✅    | 🚧           | ✅             |
| [Mixtral](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/mixtral)            | ✅        | ✅   | ✅    | ❌             | ✅             | 🚧   | ✅    | 🚧           | 🚧            |
| [Mistral](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/mistral)            | ✅         | ✅   | ✅    | ✅             | ✅             | 🚧   | ✅    | 🚧           | ✅             |
| [Baichuan/Baichuan2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)   | ✅        | ✅   | ✅    | ✅             | ✅             | 🚧   | ✅    | ✅            | ✅             |
| [ChatGLM-6B](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/chatglm)         | ✅        | ✅   | ✅    | ✅             | 🚧            | 🚧   | ✅    | ✅            | ❌             |
| [ChatGLM2/ChatGLM3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/chatglm2) | ✅        | ✅   | ✅    | ✅             | ✅             | 🚧   | ✅    | ✅            | ✅             |
| [Bloom](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/bloom)                | ✅        | ✅   | ✅    | ✅             | 🚧            | 🚧   | ✅    | ✅            | ✅             |
| [GPT-3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/gpt-3)                | ✅        | ✅   | 🚧   | 🚧            | 🚧            | 🚧   | ✅    | 🚧           | ✅             |
| [OPT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/opt)                    | ✅       | ✅   | ✅    | 🚧            | 🚧            | 🚧   | ✅    | 🚧           | ✅             |
| [Gemma](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/gemma)                | ✅       | ✅   | ✅    | 🚧            | ✅            | 🚧   | ✅    | 🚧           | 🚧             |
| [Yuan](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/yuan)                  | ✅       | ✅   | ✅    | 🚧            | ✅            | 🚧   | ✅    | 🚧           | 🚧             |

- ✅: Supported
- 🚧: In Progress
- ❌: Not Supported

## 🚀 快速开始 🚀

开始之前，您可以安装先 PaddleNLP 最新 develop 版本:
```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### 1. 预训练

PaddleNLP 将飞桨4D 并行策略加入到 Trainer API 中， 用户只需修改 Trainer 配置即可使用不同的分布式策略。目前大模型套件提供[LLaMA/LLaMA2/LLaMA3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/llama)、[GPT-3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/gpt-3)、[Qwen](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/qwen)、[Baichuan/Baichuan2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/baichuan)、[Mixtral](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/config/mixtral) 等模型预训练功能，更多模型支持持续更新中。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/a2f0261d-7f76-4faf-ae01-cc9d37d5fcc0">
</div>
<div align="center">
    <font size ="1">
    飞桨与 Megatron 预训练性能比对
     </font>
</div>

我们在此处提供了更详细的[预训练数据制作](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/tools/preprocess)，[Pretrain 和自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/llm/dataset.html)，[分布式策略支持情况](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/pretrain.html#model-capability)，[性能测试报告文档](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/pretrain.html#model-performance)，参见: [大模型预训练介绍](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/pretrain.html), [大模型权重列表](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/pretrain.html#model-weight)。

此项目支持了 LLaMA、GPT-3、BaiChuan、Qwen 和 Mixtral 等大模型的预训练。用户切换配置 config 文件，即可一键运行。

为了方便用户运行测试本模型，本项目提供了处理好的100k 条 doc 的训练样本：

```shell
# llama 模型数据下载
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx

# gpt 模型数据下载
# wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt2_openwebtext_100k.bin
# wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt2_openwebtext_100k.idx
```

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```shell
mkdir data
mv llama_openwebtext_100k.bin ./data
mv llama_openwebtext_100k.idx ./data
```
单卡训练:
```shell
# 16G 显存可训练
python -u run_pretrain.py ./config/qwen/pretrain_argument_0p5b.json
```
- 该配置16G 显存可训练，可以开启 use_flash_attention,use_fused_rms_norm,recompute 进一步省显存
- 如果上述配置无法开启，或显存依然不够，可以开启`offload_optim`,此时显存约为11G  `python -u run_pretrain.py ./config/qwen/pretrain_argument_0p5b.json  --offload_optim  1`

高性能、多卡、多机训练:
```shell
# 编译自定义算子，可选
cd ../slm/model_zoo/gpt-3/external_ops/ && python3 setup.py install && cd -

# 多卡模型预训练参考:
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_pretrain.py ./config/llama/pretrain_argument.json
# 多机训练参考: 占用45G显存左右
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7"  --master=192.168.1.1:8090 --nnodes=2  run_pretrain.py ./config/llama/pretrain_argument.json
```
- 更详细的分布式启动命令请参考[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/paddle/distributed/launch_cn.html#launch)。

注意：

1. 建议使用 paddle develop 版本训练，需要安装`pip install fast_dataindex visualdl==2.5.3`等相关缺失 whl 包
2. `use_flash_attention` 需要在 A100 以上机器开启，建议使用 cuda11.8以上环境。
3. `use_fused_rms_norm` 需要安装自定义算子。如果安装后仍然找不到算子，需要额外设置 PYTHONPATH
4. `continue_training` 表示从现有的预训练模型加载训练。7b 模型初始 loss 大概为2.xx, 随机初始化模型 loss 从11.x 左右下降。
5. 多机训练时，若各机器使用的训练数据文件位置相同（例如挂载共享硬盘情况），请指定`--share_folder true`使全局0号卡制作缓存数据。否则默认各台机器的0号卡独立制作缓存数据，
6. 若数据集文件夹中存在默认缓存文件夹`index-cache/`，则额外指定的`--data_cache`不生效，训练时优先加载默认缓存文件夹中的内容。

### 2. 精调

PaddleNLP 支持多个主流大模型的 SFT、PEFT 等精调策略，提供统一、高效精调方案：

- **统一训练入口**。飞桨大模型套件精调方案可适配业界主流大模型，用户只需修改配置文件，即能在单卡或多卡（支持4D 并行分布式策略）进行多种大模型精调。
- **高效数据和分布式策略**。Zero Padding 零填充优化策略结合 FlashMask 策略有效提升模型训练效率。独创 PEFT 结合低比特和分布式并行策略，大幅降低大模型精调硬件门槛，支持单卡（A100 80G）百亿模型微调、单机（A100 80G * 8）千亿模型微调。
- **支持多轮对话**。支持统一对话模板，支持多轮对话高效训练，详参[多轮对话文档](./docs/chat_template.md)。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/cb226f26-ce86-433e-8bb3-02fc04e8d813">
</div>
<div align="center">
    <font size ="1">
    飞桨与 Huggingface Transformers 微调性能比对
     </font>
</div>

#### 2.1 数据准备

我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：

```text
{"src": "Give three tips for staying healthy.", "tgt": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."}
...
```

为了方便测试，我们也提供了[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)demo 数据集可以直接使用：

```shell
# 在 PaddleNLP/llm 目录执行
wget https://bj.bcebos.com/paddlenlp/datasets/examples/alpaca_demo.gz
tar -xvf alpaca_demo.gz
```

#### 2.2 全参精调：SFT

单卡
```bash
# 需要12G显存左右
python -u run_finetune.py ./config/qwen/sft_argument_0p5b.json
# 单卡性能最佳实践，16G显存，可以参考打开开关。
# ./config/qwen/sft_argument_0p5b_best.json
```

多卡
```bash
# SFT 启动命令参考，需要45G显存左右
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py ./config/qwen/sft_argument.json
```

#### 2.3 LoRA

LoRA 启动命令参考
```bash
# 需要9G左右显存
python run_finetune.py ./config/qwen/lora_argument_0p5b.json
# 需要29G左右显存
python run_finetune.py ./config/qwen/lora_argument.json
```

#### 2.4 Prefix Tuning

Prefix Tuning 启动命令参考
```bash
# 需要10G左右显存
python run_finetune.py ./config/qwen/pt_argument_0p5b.json
# 需要30G左右显存
python run_finetune.py ./config/qwen/pt_argument.json
```

除了 LoRA、Prefix Tuning 外，还支持 LoKr、VeRA、MoRA、ReFT、rsLoRA、LoRA+、PiSSA、MoSLoRA 等多种精调算法，更多大模型精调使用文档、训练细节和效果请参见[大模型精调教程](./docs/finetune.md)。

### 3. 对齐

我们支持 DPO、KTO、RLHF 等偏好对齐策略。DPO、KTO 策略采用 zero_padding 策略，结合 FlashMask 策略，有效提升模型训练效率。

#### 3.1 DPO

##### 数据准备

我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `src` : `str, List(str)`, 用户对话内容。
- `tgt` : `str, List(str)`, 系统回复内容。
- `response` : `str, List(str)`, 包含 chosen 和 rejected 回复。
- `sort` : `List(int)`, sort 值用于区分 response 中 chosen 和 rejected（sort 值小的是 rejected，sort 值大的是 chosen）。。

样例数据：

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
...
```

为了方便测试，我们也提供了偏好数据集可以直接使用：

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```

##### 全参 DPO


```bash
# DPO 启动命令参考, 8卡训练， 需要大概40G显存
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/dpo_argument.json

# 单卡训练，大概需要26G显存左右
python -u  ./alignment/dpo/run_dpo.py ./config/qwen/dpo_argument_0p5b.json
```

##### LoRA DPO

```bash
# DPO 启动命令参考
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/dpo_lora_argument.json
```
更多 DPO 技术细节和使用说明详见[DPO 文档](./docs/dpo.md)。
```bash
# 单卡执行, 需要52G左右显存
python -u  ./alignment/dpo/run_dpo.py ./config/llama/dpo_lora_argument.json
```

#### 3.2 KTO

##### 数据准备

我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `src` : `str, List(str)`, 用户对话内容。
- `tgt` : `str, List(str)`, 系统回复内容。
- `response` : `str, List(str)`, 包含 resoinse 回复。
- `sort` : `List(int)`, sort 值用于区分 response 属于 chosen 和 rejected（0是 rejected，1是 chosen）。

样例数据：

```text
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?"],
    "sort": [1]
}
...
```

为了方便测试，我们也提供了偏好数据集可以直接使用：

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized_pointwise.tar
tar -xvf ultrafeedback_binarized_pointwise.tar.gz
```

##### 全参 KTO

```bash
# KTO 启动命令参考
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/kto/run_kto.py ./config/llama/kto_argument.json
```
##### LoRA KTO

```bash
# KTO 启动命令参考
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/kto/run_kto.py ./config/llama/kto_lora_argument.json
```

#### 3.3 RLHF

飞桨大模型套件提供了提供了基于强化学习 PPO 算法对 LLM 进行人类偏好对齐的代码及完整使用示例，支持**3D 分布式并行训练以及 rollout 阶段使用预测优化进行生成加速**。详细使用教程详见[RLHF 文档](./docs/rlhf.md)。

### 4. 模型融合
PadlleNLP 支持多种模型融合方法，包括**Linear、Slerp、Ties、DARE、DELLA**，并支持模型参数稀疏化方法与模型融合算法的灵活组合使用。
```shell
# 模型融合启动命令参考
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./llm/tools/mergekit.py \
    --tensor_type pd \
    --merge_method linear \
    --model_path_str "../checkpoints/model1,../checkpoints/model2" \
    --output_path ../checkpoints/model_merge

```

更多模型融合算法与细节详见[模型融合文档](./docs/mergekit.md)。

### 5. 量化

大模型量化将16位、32位浮点数的模型参数或激活量化为4位或8位整数能够有效降低模型存储空间和计算资源需求，同时加速推理速度。量化算法包含：

- **PTQ**。PaddleSlim 团队自研的自适应 LLM.PTQ 量化算法，在[SmoothQuant](https://arxiv.org/abs/2211.10438)和[Outlier Suppression+](https://arxiv.org/abs/2304.09145)基础上新增 PieceWiseSearch 参数搜索算法，对模型权重和激活分布进行调整，减少后续 A8W8 PTQ 量化损失。
- **GPTQ**。[GPTQ](https://arxiv.org/abs/2210.17323)是业界主流的权重量化算法，可以将大模型权重进行4位整数无损量化，提高模型推理速度。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/969b62db-9692-4d50-b91a-85cff305d153">
</div>
<div align="center">
    <font size ="1">
    飞桨 W4和 W8A8量化算法效果展示
     </font>
</div>
<div align="center">
    <img width="300" alt="llm" src="https://github.com/user-attachments/assets/ab8d04ba-d589-4f54-acf1-b00c0fd9159e">
</div>
<div align="center">
    <font size ="1">
    飞桨 W8A8C8和 FP8量化效果展示
     </font>
</div>

```shell
# PTQ 量化启动命令参考
python run_quantization.py ./config/llama/ptq_argument.json

# GPTQ 量化启动命令参考
python run_quantization.py ./config/llama/gptq_argument.json

# W8A8C8(INT)量化启动命令参考
python run_quantization.py ./config/llama/ptq_c8_argument.json

# W8A8(FP8)量化启动命令参考
python run_quantization.py ./config/llama/fp8_ptq_argument.json
```

更多技术细节和模型量化使用详见[量化文档](./docs/quantization.md)。

### 6. 推理

PaddleNLP 提供高性能推理，内置动态插入和全环节算子融合策略，极大加快并行推理的速度，同时支持 FP16/BF16、WINT8、WINT4、A8W8、A8W8C8多种推理方式。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/fb248224-0ad1-4d6a-a1ca-3a8dd765c41d">
</div>
<div align="center">
    <font size ="1">
    推理部署性能业界领先
     </font>
</div>


<a id="paddlenlpops"></a>
paddlenlp_ops 安装高性能推理算子教程（可选）
```shell
cd ../csrc/
python setup_cuda.py install
cd -
```

```shell
# 动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --inference_model --dtype float16

# 静态图模型推理命令参考
# step1 : 静态图导出
python ./predict/export_model.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --inference_model --output_path ./inference --dtype float16
# step2: 静态图推理
python ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype "float16" --mode "static"
```
参数说明：
1. **`--inference_model`** 参数表示使用高性能自定义算子推理，否则使用普通动态图推理(如果可以安装算子，建议打开此开关)。打开时，请前往[此处安装](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/csrc)高性能推理自定义算子，
2. **`--mode`** 有两种模式可选 `dynamic`, `static`。分别表示动态图和静态图模式。静态图模型需要进行参数导出步骤，动态图不需要。具体可以参考上述命令执行。静态图情况下，导出和推理的参数`--inference_model`需要一致。
3. 推理速度简要比较。`static+inference_model` > `dynamic+inference_model` >> `static w/o inference_model` > `dynamic w/o inference_mode`。推荐安装高性能算子，使用 `动态图+inference_model` 模式，方便快捷。


更多模型推理使用方法详见[大模型推理文档](./docs/predict/inference.md)。

### 7. 服务化部署

#### 7.1 Flask & Gradio UI 服务化部署

我们提供了一套基于动态图推理的简单易用 UI 服务化部署方法，用户可以快速部署服务化推理。

请确保，在部署前请确保已正确安装 NLP，clone 本 repo 下位置代码。以及自定义算子库。本部署的服务是兼容 OpenAI API 接口



环境准备

- python >= 3.8
- gradio
- flask
- paddlenlp_ops (可选，高性能自定义加速算子， 安装参考[这里](#paddlenlpops))


服务化部署脚本

```shell
# 单卡，可以使用 paddle.distributed.launch 启动多卡推理
python  ./predict/flask_server.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --port 8010 \
    --flask_port 8011 \
    --dtype "float16"
```

- `port`: Gradio UI 服务端口号，默认8010。
- `flask_port`: Flask 服务端口号，默认8011。
- 其他参数请参见[推理文档](./docs/predict/inference.md)中推理参数配置。

图形化界面: 打开 `http://127.0.0.1:8010` 即可使用 gradio 图形化界面，即可开启对话。
API 访问: 您也可用通过 flask 服务化 API 的形式

1. 可参考：`./predict/request_flask_server.py` 文件访问。
```shell
python predict/request_flask_server.py
```

2. 或者直接使用 curl,调用开始对话
```shell
curl 127.0.0.1:8011/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"message": [{"role": "user", "content": "你好"}]}'
```
3. 使用 OpenAI 客户端调用：
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
        {"role": "user", "content": "PaddleNLP好厉害！这句话的感情色彩是？"}
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


#### 7.2 大模型服务化部署工具

该部署工具是基于英伟达 Triton 框架专为服务器场景的大模型服务化部署而设计。它提供了支持 gRPC、HTTP 协议的服务接口，以及流式 Token 输出能力。底层推理引擎支持连续批处理、weight only int8、后训练量化（PTQ）等加速优化策略，为用户带来易用且高性能的部署体验。

基于预编译镜像部署，本节以 DeepSeek-R1-Distill-Llama-8B（weight_only_int8） 为例，自动下载静态图进行部署，具体支持模型可查看[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)，更细致的模型推理、量化教程可以参考[大模型推理教程](./docs/predict/inference.md)：

```shell

export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B/weight_only_int8"}
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'
```

等待服务启动成功（服务初次启动大概需要40s），可以通过以下命令测试：

```shell
curl 127.0.0.1:9965/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"text": "hello, llm"}'
```

Note:
1. 请保证 shm-size >= 5，不然可能会导致服务启动失败
2. 部署前请确认模型所需要的环境和硬件，请参考[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

更多模型请参考[LLaMA](./docs/predict/llama.md)、[Qwen](./docs/predict/qwen.md)、[DeepSeek](./docs/predict/deepseek.md)、[Mixtral](./docs/predict/mixtral.md)。
更多关于该部署工具的使用方法，请查看[服务化部署流程](./server/docs/deploy_usage_tutorial.md)

### 8. PyTorch 模型权重转换

PaddleNLP 提供了可自动将 PyTorch 相关的权重转化为 Paddle 权重的接口，代码如下：

```python
from paddlenlp.transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("/path/to/pytorch/model", convert_from_torch=True,dtype="float16")
```

更多细节请参考[torch2paddle 文档](./docs/torch2paddle.md)
