# 三步极速蒸馏 DeepSeek R1

## 1. 高吞吐、低成本的飞桨 DeepSeek-R1 推理服务部署

飞桨新一代框架3.0全面升级了大模型推理能力，依托高扩展性的中间表示（PIR）从模型压缩、推理计算、服务部署、多硬件推理全方位深度优化，能够支持众多开源大模型进行高性能推理，并在 DeepSeek V3/R1上取得了突出的性能表现。飞桨框架3.0支持了 DeepSeek V3/R1满血版及其系列蒸馏版模型的 FP8推理，并且提供 INT8量化功能，破除了 Hopper 架构的限制。此外，还引入了4比特量化推理，使得用户可以单机部署，降低成本的同时显著提升系统吞吐近一倍，提供了更为高效、经济的部署方案。

在性能优化方面，我们对 MLA 算子进行多级流水线编排、精细的寄存器及共享内存分配优化，性能相比 FlashMLA 最高可提升23%。综合 FP8矩阵计算调优及动态量化算子优化等基于飞桨的 DeepSeek R1 FP8推理，单机每秒输出 token 数超1000；若采用4比特单机部署方案，每秒输出 token 数可达2000以上！推理性能显著领先其他开源方案。此外，还支持了 MTP 投机解码，突破大批次推理加速，在解码速度保持不变的情况下，吞吐提升144%；吞吐接近的情况下，解码速度提升42%。针对长序列 Prefill 阶段，通过注意力计算动态量化，首 token 推理速度提升37%。

<p align="center">
  <img src="https://github.com/user-attachments/assets/84a90f79-6fb7-434d-857e-cbd964558f02" align="middle"  width="500" />
</p>


DeepSeek-R1 单机 WINT4推理：以1台 H800/A800为例，部署单机4比特量化推理服务。
  * 设置变量 model_name 声明需要下载的模型，具体支持的静态图模型详见文档。
  * 设置模型存储路径 MODEL_PATH，默认挂载至容器内/models 路径下（请确认对存储路径 MODEL_PATH 具有写权限）
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name="deepseek-ai/DeepSeek-R1/weight_only_int4"
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models \
-e "model_name=${model_name}" \
-e "MP_NUM=8" \
-e "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'&& docker logs -f $(docker ps -lq)
```

启动命令后，模型会自动开始下载并部署服务，用户可以 curl 请求判断模型是否成功部署，更多 DeepSeek-R1部署方案细节和请求参数请参考 DeepSeek 部署。
```shell
curl ${ip}:9965/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
      "model":"default",
      "text":"Hello, how are you?"
  }'
```

**详细部署启动命令请参考[DeepSeek 部署](../../docs/predict/deepseek.md)。**


## 2. 超详细的飞桨数据蒸馏-训练-评测方案
我们将数据蒸馏的方案划分为三部分，分别为数据蒸馏、模型训练和模型评估，相关代码已经上传至 PaddleNLP。

## 环境准备

### 安装 PaddleNLP 组件
```shell
pip install --upgrade paddlenlp==3.0.0b4
```

或者可通过以下命令安装最新 develop 分支代码：

```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### 安装高性能算子组件
请参考[文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/csrc)安装高性能算子组件。

如果您正在使用 Python3.10，可以使用如下命令安装预编译版本。
```shell
pip install --pre --upgrade paddlenlp_ops -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### 安装数据蒸馏相关依赖库

```shell
pip install -r requirements.txt
```

### 2.1. 数据蒸馏

我们将使用中文版 GSM8K 数据集进行数据蒸馏，随后将其转换为 JSONL 格式。
```python
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from copy import deepcopy

from datasets import load_dataset

# convert data for distill
# GSM8K
dataset = load_dataset("meta-math/GSM8K_zh")["train"]
dataset.to_json("data/gsm8k_zh/GSM8K_zh.jsonl", force_ascii=False)
```

在准备好数据集后，可以使用下面的命令进行蒸馏。
```shell
python distill_data.py \
    --input_file "./data/gsm8k_zh/GSM8K_zh.jsonl" \
    --output_dir "./data/GSM8K_distilled_zh" \
    --prompt_key "question_zh" \
    --response_key "deepseek_r1_response_zh" \
    --reasoning_key "deepseek_r1_reasoning_zh" \
    --prompt_suffix "\n请一步一步地推理，并将你的最终答案放在\boxed{}中。" \
    --base_urls "http://192.168.0.1:9965/v1,http://192.168.0.2:9965/v1" \
    --model deepseek-r1:671b \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 32768 \
    --concurrency 16
```

调用参数详细解释如下：
```text
--input_file：此参数用于指定输入的 JSONL 文件路径。例如，在之前的案例中，我们使用了 meta-math_gsm8k_zh.jsonl 文件作为输入源。
--output_dir：该参数用于设置输出目录，该目录将用于保存经过蒸馏处理后的 JSONL 文件。此目录还具备断点蒸馏功能，即若程序运行中断，您可以重新运行脚本以继续蒸馏过程，无需从头开始。
--prompt_key：此参数用于指定输入 JSONL 文件中用作 prompt 的字段名称。在 GSM8K 数据集中，我们使用了 question_zh 字段的内容作为问题文本。
--response_key：该参数用于指定输出 JSONL 文件中用作 response 的字段名称。例如，我们选择了 deepseek_r1_response_zh 作为输出回复文本的字段。
--reasoning_key：此参数用于指定输出 JSONL 文件中用作 reasoning 的字段名称。在示例中，我们使用了 deepseek_r1_reasoning_zh 作为输出推理文本的字段。
--prompt_suffix：该参数允许您在 prompt 文本后添加附加内容。对于 DeepSeek R1 处理数学问题，我们建议在 prompt 后添加 "\n请一步一步地推理，并将你的最终答案放在\boxed{}中。"，以符合 DeepSeek R1 的最佳实践要求。
--base_urls：此参数用于指定服务地址，您可以输入多个地址，地址间用英文逗号分隔，以实现负载均衡。如果服务是本地部署的，请设置为 http://127.0.0.1:8XXX/v1。
--api_keys：该参数用于提供服务密钥，您可以输入多个密钥，密钥间用英文逗号分隔，并与 base_urls 中的地址一一对应。如果服务是本地部署的，则无需设置此参数。
--model：此参数用于指定蒸馏的目标模型名称，例如 deepseek-r1:671b。
--temperature：该参数用于设置生成温度，通过调整该参数，您可以控制生成内容的随机性程度。
--top_p：此参数用于设置生成概率阈值，通过调整该参数，您可以控制生成内容的多样性。
--max_tokens：该参数用于指定生成内容的最大 token 数限制，以确保输出内容在合理范围内。
--concurrency：此参数用于设置并发数，即同时向 API 发起的请求数量，您可以根据需要调整此参数，以控制请求处理的并行度。
```
运行过程中可以看到如下的进度条，表示蒸馏进度。蒸馏过程中会在 output_dir 目录下生成三个文件，分别是蒸馏后的数据集、请求 API 时的日志和当前蒸馏状态。
```text
[deepseek-r1:7b] Data Distilling Progress:   0%|▊           | 36/8792 [05:57<15:29:58,  6.37s/it]

tree ./data/GSM8K_distilled_zh
./data/GSM8K_distilled_zh
├── distilled-GSM8K_zh.jsonl
├── distilled-GSM8K_zh.log
└── distilled-GSM8K_zh.status

0 directories, 3 files
```

由于我们采用了8路并发，导致最终蒸馏后的数据集是乱序的，如果我们需要恢复原始数据集的顺序，可以使用以下代码：
```python
from datasets import load_dataset, Dataset

ds = load_dataset("json", data_files="./data/GSM8K_distilled_zh/distilled-GSM8K_zh.jsonl", split="train")
df = ds.to_pandas()
sorted_df = df.sort_values(by='line_num')
sorted_ds = Dataset.from_pandas(sorted_df)
sorted_ds.to_json("./data/GSM8K_distilled_zh/sorted-distilled-GSM8K_zh.jsonl", force_ascii=False)
```
最终我们将会得到如下字段的数据集，新增了 deepseek_r1_reasoning , deepseek_r1_response, deepseek_r1_completion_tokens 和 deepseek_r1_prompt_tokens 字段，分别对应 DeepSeek R1 的推理过程和回答。
```json
{
    "line_num":0,
    "question":"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "answer":"Natalia sold 48\/2 = <<48\/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    "answer_only":"72",
    "split":"train",
    "question_zh":"Natalia在四月份向她的48个朋友出售了夹子，然后在五月份卖出了四月份的一半。Natalia在四月和五月总共卖了多少个夹子？",
    "answer_zh":"Natalia在五月份卖出了48\/2 = 24个夹子。\nNatalia在四月和五月总共卖出了48+24 = 72个夹子。",
    "deepseek_r1_response_zh":"\n\n首先，Natalia 在四月份向她的 48 个朋友出售了夹子，这意味着她在四月份总共卖出了 48 个夹子。\n\n接下来，在五月份，她卖出的夹子数量是四月份的一半。因此，五月份卖出的数量为：\n\\[ \\frac{48}{2} = 24 \\]\n\n为了计算四月和五月总共卖出的夹子数量，我们将这两个月的销售量相加：\n\\[ 48 + 24 = 72 \\]\n\n最终，Natalia 在四月和五月总共卖出了 72 个夹子。\n\n\\[\n\\boxed{72}\n\\]",
    "deepseek_r1_reasoning_zh":"<think>\n嗯，好的，我现在来仔细思考一下这个问题该怎么解决。\n\n题目说，Natalia在四月份向她的48个朋友每个卖了夹子，然后在五月份卖出了四月份的一半数量。问题是她在四月和五月总共卖了多少个夹子。\n\n首先，我需要理解题目的意思是否正确。题目说的是“卖给了四月份的一半”，这里的“一半”是指卖的数量还是价格呢？不过从上下文来看，应该是销售数量的“一半”。也就是说，在四月份她卖出48个夹子，那么五月份她卖出的就是48的一半，也就是24个夹子。\n\n那接下来我要计算的是四月和五月总共卖了多少个夹子。那就是把四月的数量加上五月的数量，也就是48加24对吧？\n\n让我再确认一下步骤是否正确：\n\n1. 四月份销售数量：48个。\n2. 五月份销售数量是四月份的一半，所以5月份就是48 ÷ 2 = 24个。\n3. 总销售量就是4月和5月的销售量之和，即48 + 24。\n\n计算一下：48加24等于72。嗯，看起来没问题。\n\n不过，我是不是哪里有问题呢？比如，题目里说“向她的48个朋友出售了夹子”，这可能意味着每个朋友买了一个夹子吗？也就是说，四月份总共卖出了48个夹子对吧？\n\n然后，五月份卖出的数量是四月份的一半，也就是24个。那么加起来的话确实是72个。\n\n嗯，这样应该是对的吧。让我再检查一下是否理解正确题目要求。题目问的是总共的数量，而不是平均数或者其他什么的，所以没问题。\n\n还有一点需要注意：是否有其他隐藏的信息或者陷阱？比如，这里的“卖出”是指每个朋友一个夹子吗？或者有没有可能在四月份她卖了48个夹子给朋友们，而五月份又卖了多少呢？\n\n不过再读一遍题目，题目说：“Natalia在四月份向她的48个朋友出售了夹子”，那么这个数量应该是总共卖出的数目。也就是说，四月份卖出了48个夹子。\n\n然后，“在五月份卖出了四月份的一半”，所以五月份就卖出了24个。\n\n那么，总共就是48+24=72个夹子对吧？没错，没有问题。\n\n嗯，看来我的思路是对的。这样计算下来，Natalia在四月和五月一共卖了72个夹子。\n\n哦，等一下，我是不是哪里弄错了？比如，“四个朋友”或者“五月份卖出的数量”，会不会是其他意思呢？\n\n不，题目里明确说：“Natalia在四月份向她的48个朋友出售了夹子。”所以她这四月份总共卖了48个夹子。然后，“在五月份卖出了四月份的一半数量”，也就是24个。\n\n嗯，没错，看来没问题。\n\n那总数就是72个夹子。\n\n**答案**\nNatalia在四月和五月总共卖了\\boxed{72}个夹子。\n<\/think>"
}
......
```

接下来我们可以将蒸馏后的数据进一步进行蒸馏训练， 此处采用的数据为我们蒸馏后数据集 [PaddlePaddle/GSM8K_distilled_zh](https://huggingface.co/datasets/PaddlePaddle/GSM8K_distilled_zh)，用户可自行修改输入输出数据为蒸馏后数据， PaddleNLP 支持的格式如下:
* src : str, List(str), 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
* tgt : str, List(str), 模型的输出。
样例数据：
```json
{"src": "Give three tips for staying healthy.", "tgt": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."}
......
```
```python
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from copy import deepcopy

from datasets import load_dataset

# PaddlePaddle/GSM8K_distilled_zh
dataset = load_dataset("PaddlePaddle/GSM8K_distilled_zh")
dataset["train"].to_json("data/gsm8k_distilled_zh/GSM8K_distilled_zh-train.json", force_ascii=False)
dataset["test"].to_json("data/gsm8k_distilled_zh/GSM8K_distilled_zh-test.json", force_ascii=False)

# make data for sft
def process_data_zh(example):
    src = example.get("question_zh", "")
    content = example.get("deepseek_r1_response_zh", "")
    reasoning_content = example.get("deepseek_r1_reasoning_zh", "")
    tgt = reasoning_content + content
    return {"src": src, "tgt": tgt}

def process_data_en(example):
    src = example.get("question", "")
    content = example.get("deepseek_r1_response", "")
    reasoning_content = example.get("deepseek_r1_reasoning", "")
    tgt = reasoning_content + content
    return {"src": src, "tgt": tgt}

# construct Chinese sft dataset
paddlenlp_datatset = deepcopy(dataset)
paddlenlp_datatset["train"] = paddlenlp_datatset["train"].map(
    process_data_zh, remove_columns=paddlenlp_datatset["train"].column_names
)
paddlenlp_datatset["test"] = paddlenlp_datatset["test"].map(
    process_data_zh, remove_columns=paddlenlp_datatset["test"].column_names
)
paddlenlp_datatset["train"].to_json("data/gsm8k_distilled_zh_sft/train.json", force_ascii=False)
paddlenlp_datatset["test"].to_json("data/gsm8k_distilled_zh_sft/dev.json", force_ascii=False)

# construct English sft dataset
paddlenlp_datatset = deepcopy(dataset)
paddlenlp_datatset["train"] = paddlenlp_datatset["train"].map(
    process_data_en, remove_columns=paddlenlp_datatset["train"].column_names
)
paddlenlp_datatset["test"] = paddlenlp_datatset["test"].map(
    process_data_en, remove_columns=paddlenlp_datatset["test"].column_names
)
paddlenlp_datatset["train"].to_json("data/gsm8k_distilled_en_sft/train.json", force_ascii=False)
paddlenlp_datatset["test"].to_json("data/gsm8k_distilled_en_sft/dev.json", force_ascii=False)

```
最终我们将会得到如下字段的数据集
```json
{"src":"Natalia在四月份向她的48个朋友出售了夹子，然后在五月份卖出了四月份的一半。Natalia在四月和五月总共卖了多少个夹子？","tgt":"<think>\n嗯，好的，我现在要解决这个问题。首先，题目是说Natalia在四月份向她的48个朋友卖了夹子，然后在五月份卖出了四月份的一半。然后问四月份和五月份总共卖了多少个夹子。对吧？\n\n首先，我需要理清楚每个月的销售量。四月份的时候，她向48个朋友卖了夹子。这里可能需要注意，是每个朋友买了一个夹子，还是总共卖了48个夹子？题目里说“向她的48个朋友出售了夹子”，可能这里的数量不是很明确。不过通常这种题目如果没有特别说明的话，可能指的是她卖给每个朋友一个夹子，所以四月份她卖了48个夹子，对吗？或者说，她向48个朋友出售夹子，每个朋友买了多少呢？这里可能需要再仔细看一下题目。\n\n不过题目里并没有提到每个朋友买的数量，所以可能默认每个朋友买了一个，所以四月份卖了48个夹子。然后五月份卖出了四月份的一半，也就是五月份卖了48的一半，也就是24个夹子。那么四月份和五月份总共卖了48加24，等于72个夹子。对吧？\n\n不过，等等，这里可能有没有理解错的地方。让我再仔细看一下题目。题目说“Natalia在四月份向她的48个朋友出售了夹子”，然后五月份卖出了四月份的一半。这里的“四月份的一半”可能指的是四月份销售量的一半，也就是四月份卖了48个，五月份卖了24个，所以总共有72个。对吧？\n\n不过，有没有可能四月份的时候，她向48个朋友各卖了一个夹子，所以四月份总共卖了48个，然后五月份卖出了四月份数量的一半，也就是24个，所以总销量是48+24=72。没错，应该是这样。\n\n不过，也有可能题目中的“四月份的一半”指的是五月份卖出的夹子数量是四月份朋友数量的一半，也就是四月份的朋友数量是48，五月份的朋友数量是24，然后每个朋友买了一个夹子，所以四月份卖了48个，五月份卖了24个，总共72个。不过这样的话，结果还是一样的，对吗？\n\n不过，再仔细想想，题目中的“四月份向她的48个朋友出售了夹子”，然后五月份卖出了四月份的一半。这里的“四月份的一半”可能指的是四月份卖出数量的一半，而不是朋友数量的一半。所以四月份卖出的是48个，五月份卖出的是24个，总销量72个。没错，这样计算是对的。\n\n不过，有没有可能四月份的时候，她卖给每个朋友多个夹子，所以总销量更多？但题目里没有提到每个朋友买了多少，所以只能假设每个朋友买了一个，所以四月份是48个，五月份是24个，总共72个。是的，应该是这样。\n\n所以，最终的答案应该是72，对吗？不过，再检查一遍，四月份48，五月份24，加起来72，没错。所以，答案应该是72，用数学符号框起来。\n\n不过，再确认一下，题目中的“四月份的一半”是否指的是五月份的销售量是四月份销售量的一半，是的，没错。所以五月份是24，总销量72。没错，没问题。\n<\/think>\n\nNatalia在四月份卖出的夹子数量为48个。五月份卖出的夹子数量是四月份的一半，即48 ÷ 2 = 24个。因此，四月和五月总共卖出的夹子数量为48 + 24 = 72个。\n\n\\boxed{72}"}
......
```

### 2.2. 模型训练
通过对模型蒸馏后的模型进行精调即可让模型具备思考能力，蒸馏后的数据相对较长，需要依赖训练套件的长文训能力。PaddleNLP 在精调（微调）训练进行了极致优化性能，并支持了128K 长上下文训练。

飞桨框架凭借其独有的 FlashMask 高性能变长注意力掩码计算技术，结合 PaddleNLP 中的 Zero Padding 零填充数据流优化策略，实现了显存开销的大幅缩减。这一创新使得精调训练代码能够无缝地从8K 扩展至前所未有的128K 长文本训练，训练效率相较于 LLama-Factory 更是实现了显著提升，高达1.8倍。

在这里，我们运用了全参监督精调（SFT）算法来精细调整小型模型的参数。这一流程极为高效便捷，仅需要输入模型和数据集即可完成微调、模型压缩等任务。此外，我们提供了一键式启动多卡训练、混合精度训练、梯度累积、断点重启以及日志显示等一系列功能。同时，我们还对训练过程中的通用配置进行了封装，涵盖了优化器选择、学习率调度等关键环节。若您希望探索更多精调算法，请查阅大模型精调的相关配置指南。

使用下面的命令，使用 `Qwen/Qwen2.5-Math-7B` 作为预训练模型进行模型微调，将微调后的模型保存至指定路径中。如果在 GPU 环境中使用，可以指定 gpus 参数进行多卡训练，训练配置文件参考`sft_argument.json`。

监督微调方式 （SFT）
```shell
python -u -m paddle.distributed.launch \
    --devices "0,1,2,3,4,5,6,7" \
    ../../run_finetune.py \
    sft_argument.json
```

`sft_argument.json` 配置文件如下, 具体参数配置含义请参考[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/finetune.md):
```json
{
  "model_name_or_path": "Qwen/Qwen2.5-Math-7B",
  "dataset_name_or_path": "./data/gsm8k_distilled_en_sft",
  "output_dir": "./checkpoints/Qwen2.5-Math-7B/gsm8k_distilled/",
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "per_device_eval_batch_size": 1,
  "eval_accumulation_steps": 16,
  "num_train_epochs": 3,
  "learning_rate": 3e-05,
  "warmup_steps": 30,
  "logging_steps": 1,
  "evaluation_strategy": "no",
  "save_strategy": "epoch",
  "src_length": 8192,
  "max_length": 16384,
  "bf16": true,
  "fp16_opt_level": "O2",
  "do_train": true,
  "do_eval": false,
  "eval_steps": 10,
  "disable_tqdm": true,
  "load_best_model_at_end": false,
  "eval_with_do_generation": false,
  "metric_for_best_model": "accuracy",
  "recompute": true,
  "save_total_limit": 1,
  "tensor_parallel_degree": 1,
  "pipeline_parallel_degree": 1,
  "sharding": "stage2",
  "zero_padding": true,
  "unified_checkpoint": true,
  "use_flash_attention": true,
  "flash_mask": true
}
```

### 2.3. 模型评估
我们提供了在 GSM8K 上的评估脚本，便于比较模型在进行微调后在数学处理方面的能力，详细代码可以参考评估脚本。
```shell
python -u -m paddle.distributed.launch \
    --devices 0,1,2,3 \
    distill_eval.py \
    --eval_file ./data/gsm8k_distilled_zh/GSM8K_distilled_zh-test.json \
    --eval_question_key question_zh \
    --eval_answer_key answer_only \
    --eval_prompt "\n请一步一步地推理，并将你的最终答案放在\boxed{}中。" \
    --model_name_or_path "./checkpoints/Qwen/Qwen2.5-Math-7B/gsm8k_distilled/" \
    --inference_model true \
    --dtype bfloat16 \
    --batch_size 32 \
    --use_flash_attention true \
    --src_length 1024 \
    --max_length 3072 \
    --total_max_length 4096 \
    --decode_strategy greedy_search \
    --temperature 0.95 \
    --top_p 0.6 \
    --data_file ./data/gsm8k_distilled_zh/GSM8K_distilled_zh-test.json \
    --eval_results results-gsm8k/Qwen/Qwen2.5-Math-7B/output_zh.json
```

在 GSM8K（0-shot）上，PaddleNLP 蒸馏后的 Qwen2.5-Math-7B 模型（83.82%）相较于原始模型（68.21%）有了显著的性能提升，准确率提高了约15.61个百分点。

| Model Name                           | GSM8K(en, 0-shot) |
|--------------------------------------|:-----------------:|
| Qwen2.5-Math-7B                      |       68.2        |
| DeepSeek-R1-Distill-Qwen-7B          |       77.3        |
| Qwen2.5-Math-7B ( PaddleNLP Distill) |       83.8        |


## 3. 高效快速的飞桨模型本地部署方案

经过模型评估后，用户可将模型用于下游任务部署，PaddleNLP 提供了动态图高性能部署（简单易用）和服务化部署方案（高效稳定）方便用户适配不同场景使用。经过在 L20环境下的测试，我们的服务化部署方案在推理性能方面展现出了卓越的表现，显著超越其他开源方案。

|         distill model         |    框架    | 精度 | 测试并发 | 平均首 token 时延 (s) | 平均整句时延 (s) |  QPS  |  OTPS   | 提升比例 |
|:-----------------------------:|:----------:|:----:|:--------:|:-------------------:|:----------------:|:-----:|:-------:|:--------:|
| DeepSeek-R1-Distill-Qwen-32B  | vLLM-0.7.3 | W8A8 |    64    |        1.29         |      23.71       | 2.63  | 496.76  |          |
|                               | paddlenlp  | W8A8 |    64    |        3.76         |      18.81       | 3.32  | 626.22  |   26%    |
| DeepSeek-R1-Distill-Qwen-14B  | vLLM-0.7.3 | W8A8 |   256    |        1.05         |      40.39       | 6.02  | 1131.94 |          |
|                               | paddlenlp  | W8A8 |   256    |        4.77         |      18.72       | 12.37 | 1873.14 |   65%    |
|  DeepSeek-R1-Distill-Qwen-7B  | vLLM-0.7.3 | W8A8 |   256    |        0.53         |      19.82       | 12.27 | 2310.39 |          |
|                               | paddlenlp  | W8A8 |   256    |        0.45         |      10.46       | 22.23 | 3842.89 |   66%    |
| DeepSeek-R1-Distill-Qwen-1.5B | vLLM-0.7.3 | W8A8 |   256    |        0.23         |       8.34       | 28.11 | 5259.05 |          |
|                               | paddlenlp  | W8A8 |   256    |        2.92         |       6.7        | 36.51 | 6884.9  |   31%    |

在这里我们总结部署流程如下，用户可参考[教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/general_model_inference.md)动手实践：
- 模型动转静：将动态图模型转为静态图模型，便于推理部署。
- 模型服务化部署：将静态图模型部署为服务，便于调用。
