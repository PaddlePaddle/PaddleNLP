# 向量检索模型训练

推荐安装 gpu 版本的[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)，以 cuda12.3的 paddle 为例，安装命令如下：

```
conda install nccl -c conda-forge
conda install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ -c conda-forge
```
安装其他依赖：
```
pip install git+https://github.com/PaddlePaddle/PaddleNLP.git@develop
pip install -r requirements.txt
```

下载 DuReader-Retrieval 中文数据集：
```
cd data
wget https://paddlenlp.bj.bcebos.com/datasets/dureader_dual.train.jsonl
```

## 训练
### 单卡训练
当模型架构为 encoder-only 时，以 RocketQA 为例，模型名称为`rocketqa-zh-base-query-encoder`，采用单卡训练：
```
export CUDA_VISIBLE_DEVICES=0
python train.py --do_train \
              --model_name_or_path rocketqa-zh-base-query-encoder \
              --output_dir rocketqa-zh-base-query-encoder-duretrieval \
              --train_data ./data/dureader_dual.train.jsonl \
              --overwrite_output_dir \
              --fine_tune_type sft \
              --sentence_pooling_method cls \
              --num_train_epochs 3 \
              --per_device_train_batch_size 64 \
              --learning_rate 3e-5 \
              --train_group_size 4 \
              --recompute \
              --passage_max_len 512 \
              --use_matryoshka
```
### 多卡训练
单卡训练效率过低，batch_size 较小，建议使用多卡训练，对于对比学习训练推荐使用大 batch_size，多卡训练，示例命令如下：

```
python -m paddle.distributed.launch --gpus "0,1,2,3" train.py --do_train \
              --model_name_or_path rocketqa-zh-base-query-encoder \
              --output_dir rocketqa-zh-base-query-encoder-duretrieval \
              --train_data ./data/dual.train.json \
              --overwrite_output_dir \
              --fine_tune_type sft \
              --sentence_pooling_method cls \
              --num_train_epochs 3 \
              --per_device_train_batch_size 32 \
              --learning_rate 3e-5 \
              --train_group_size 8 \
              --recompute \
              --passage_max_len 512 \
              --use_matryoshka
```

当模型架构为 decoder-only 时，以[RepLLaMA](https://huggingface.co/castorini/repllama-v1-7b-lora-passage) 和 [NV-Embed-v1](https://huggingface.co/nvidia/NV-Embed-v1) 为例，采用多卡训练：
```
model_name=castorini/repllama-v1-7b-lora-passage 或 nvidia/NV-Embed-v1
output_dir=repllama-v1-7b-duretrieval 或 NV-Embed-v1-duretrieval

python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train.py --do_train \
    --query_instruction_for_retrieval "query: " \
    --passage_instruction_for_retrieval "" \
    --model_name_or_path ${model_name} \
    --output_dir ${output_dir}$ \
    --save_steps 100 \
    --train_data ./data/dureader_dual.train.jsonl  \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --recompute \
    --train_group_size 4 \
    --learning_rate 1e-4 \
    --query_max_len 128 \
    --passage_max_len 4096 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --negatives_cross_device \
    --warmup_steps 100 \
    --do_train \
    --fine_tune_type lora \
    --fp16_opt_level "O2" \
    --sentence_pooling_method last \
    --sharding "stage3 offload" \
    --use_flash_attention \
    --temperature 0.01
```
可配置参数包括：
- `model_name_or_path`: 选择预训练模型，可选 rocketqa-zh-base-query-encoder 等
- `output_dir`: 模型保存路径
- `train_data`: 训练数据集路径，这里使用的是 dureader 中文数据集
- `overwrite_output_dir`: 是否覆盖模型保存路径，默认为 False
- `fine_tune_type`: 训练模式，可选 sft 和 lora, bitfit 等策略
- `sentence_pooling_method`: 句子池化方法，可选 cls 和 mean, cls 为 CLS 层，mean 为平均池化
- `num_train_epochs`: 训练轮数
- `per_device_train_batch_size`: 单卡训练 batch 大小
- `learning_rate`: 学习率
- `train_group_size`: 每个训练集正负样本的数据，默认为8，例如 train_group_size=4，则每个训练集包含1个正样本和3个负样本
- `max_example_num_per_dataset`: 每个训练集的最大样本数，默认为100000000
- `recompute`: 是否重新计算，默认为 False
- `query_max_len`: query 的最大长度，默认为32
- `query_instruction_for_retrieval`: query 的检索指令，默认为 None
- `passage_instruction_for_retrieval`: passage 的检索指令，默认为 None
- `passage_max_len`: passage 的最大长度，默认为512
- `use_matryoshka`: 是否使用俄罗斯套娃策略（matryoshka），默认为 False
- `matryoshka_dims`: 俄罗斯套娃策略的维度，默认为[64, 128, 256, 512, 768]
- `matryoshka_loss_weights`: 俄罗斯套娃策略的损失权重，默认为[1, 1, 1, 1, 1]
- `use_inbatch_neg`: 是否使用 in batch negatives 策略，默认为 False
- `use_flash_attention`: 是否使用 flash attention，默认为 False
- `temperature`: in batch negatives 策略的 temperature 参数，默认为0.02
- `negatives_cross_device`: 跨设备 in batch negatives 策略，默认为 False
- `margin`: in batch negatives 策略的 margin 参数，默认为0.2
- `sharding`: 是否使用 Paddle Sharding 数据并行训练，基础选项应为 stage1、stage2 或 stage3，并且您可以像这样将 CPU 卸载添加到 stage2 或 stage3：stage2 offload 或 stage3 offload
- `fp16_opt_level`: 对于混合精度训练，AMP 优化级别可以选择 ['O0', 'O1', 'O2']。详情请参考 [链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html)。


## 评估
在 T2Ranking 上评估，对 RocketQA 的测试代码示例如下：
```
export CUDA_VISIBLE_DEVICES=0
model_path=rocketqa-zh-base-query-encoder-duretrieval
python evaluation/benchmarks.py --model_type bert \
    --query_model ${model_path} \
    --passage_model ${model_path} \
    --query_max_length 64 \
    --passage_max_length 512 \
```
可配置参数包括：
- `model_type`: 模型的类似，可选 bert 或 roberta 等等
- `query_model`: query 向量模型的路径
- `passage_model`: passage 向量模型的路径
- `query_max_length`: query 的最大长度
- `passage_max_length`: passage 的最大长度
- `evaluate_all`: 是否评估所有的 checkpoint，默认为 False，即只评估指定的 checkpoint
- `checkpoint_dir`: 与`evaluate_all`一起使用

在 MTEB 的 DuRetrieval 上评估，对 RocketQA 的测试代码示例如下：
```
model_path=rocketqa-zh-base-query-encoder-duretrieval
python -u evaluation/eval_mteb.py \
    --base_model_name_or_path ${model_path} \
    --output_folder eval_results/${model_path} \
    --task_name 'DuRetrieval' \
    --eval_batch_size 8 \
    --max_seq_length 2048 \
    --task_split dev
```

对 RepLLaMA 和 NV-Embed 的测试代码示例如下：
```
model_path=repllama-v1-7b-duretrieval 或 NV-Embed-v1-duretrieval
python -u evaluation/eval_mteb.py \
    --base_model_name_or_path ${model_path} \
    --output_folder eval_results/${model_path} \
    --query_instruction "query: " \
    --task_name 'DuRetrieval' \
    --eval_batch_size 8 \
    --max_seq_length 4096 \
    --task_split dev
```
可配置参数包括：
- `base_model_name_or_path`: 模型名称或路径
- `output_folder`: 结果文件存储路径
- `task_name`：任务（数据集）名称，如 DuRetrieval
- `task_split`：测试查询集合，如 test 或 dev
- `query_instruction`：查询前添加的提示文本，如'query: '或 None
- `document_instruction`：文档前添加的提示文本，如'passage: '或 None
- `pooling_method`：获取表示的方式，last 表示取最后 token，mean 表示取平均，cls 表示取`[CLS]`token
- `max_seq_length`: 最大序列长度
- `eval_batch_size`: 模型预测的批次大小（单个 GPU）
- `pad_token`：设置 padding 的 token，可取 unk_token、eos_token 或 pad_token
- `padding_side`：设置 padding 的位置，可取 left 或 right
- `add_bos_token`：是否添加起始符，0表示不添加，1表示添加
- `add_eos_token`：是否添加结束符，0表示不添加，1表示添加

# MTEB 评估
[MTEB](https://github.com/embeddings-benchmark/mteb)
是一个大规模文本嵌入评测基准，包含了丰富的向量检索评估任务和数据集。
本仓库主要面向其中的英文检索任务（Retrieval），并额外支持针对 MSMARCO-Title 的评估。

评估脚本为 `evaluation/eval_mteb.sh`，支持7个模型：
<!-- |  **模&nbsp;型**         | [RocketQA&nbsp;V1](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQA_NAACL2021) | [RocketQA&nbsp;V2](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQAv2_EMNLP2021) | [BGE‑Large‑en‑v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | [RepLLaMA‑passage](https://huggingface.co/castorini/repllama-v1-7b-lora-passage) | [NV‑Embed‑v1](https://huggingface.co/nvidia/NV-Embed-v1) | [BGE‑EN‑ICL](https://huggingface.co/BAAI/bge-en-icl) | [LLARA‑passage](https://huggingface.co/BAAI/LLARA-passage) |
|--------------|-------------|-------------|-------------------|-----------------------------|-------------|------------------------|---------------|
| **最&nbsp;大&nbsp;序&nbsp;列&nbsp;长&nbsp;度**  | 512         |     512     |        512        |            4096             |    4096     |          4096          |     4096      | -->
| 模型                        | 最大序列长度 |
|-----------------------------|--------------|
| [RocketQA&nbsp;V1](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQA_NAACL2021)    |     512      |
| [RocketQA&nbsp;V2](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQAv2_EMNLP2021)  |     512      |
| [BGE‑Large‑en‑v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)                                    |     512      |
| [RepLLaMA‑passage](https://huggingface.co/castorini/repllama-v1-7b-lora-passage)                      |     4096     |
| [NV‑Embed‑v1](https://huggingface.co/nvidia/NV-Embed-v1)                                              |     4096     |
| [BGE‑EN‑ICL](https://huggingface.co/BAAI/bge-en-icl)                                                  |     4096     |
| [LLARA‑passage](https://huggingface.co/BAAI/LLARA-passage)                                            |     4096     |

可支持配置的参数：
- `base_model_name_or_path`: 模型名称或路径
- `output_folder`: 结果文件存储路径
- `task_name`：任务（数据集）名称，如 SciFact
- `task_split`：测试查询集合，如 test 或 dev
- `query_instruction`：查询前添加的提示文本，如'query: '或 None
- `document_instruction`：文档前添加的提示文本，如'passage: '或 None
- `pooling_method`：获取表示的方式，last 表示取最后 token，mean 表示取平均，cls 表示取`[CLS]`token
- `max_seq_length`: 最大序列长度
- `eval_batch_size`: 模型预测的批次大小（单个 GPU）
- `pad_token`：设置 padding 的 token，可取 unk_token、eos_token 或 pad_token
- `padding_side`：设置 padding 的位置，可取 left 或 right
- `add_bos_token`：是否添加起始符，0表示不添加，1表示添加
- `add_eos_token`：是否添加结束符，0表示不添加，1表示添加


评估结果如下，

MSMARCO-Title 数据集, MRR@10, Recall@10, NDCG@10分数：
| 模型                        | MRR@10 | Recall@10 | NDCG@10 |
|-----------------------------|:------:|:---------:|:-------:|
| RocketQA v1                 | 36.94  |   65.67   |  43.51  |
| RocketQA v2                 | 38.88  |   67.06   |  45.28  |
| bge-large-en-v1.5           | 35.30  |   64.24   |  41.96  |
| repllama-v1-7b-lora-passage | 38.24  |   66.26   |  45.13  |
| NV-Embed-v1                 | 38.39  |   68.02   |  45.21  |
| bge-en-icl (zero-shot)      | 42.74  |   71.90   |  49.47  |
| LLARA-passage               | 43.04  |   72.59   |  49.87  |

MTEB-Retrieval 数据集, MRR@10分数：
| 模型                        | 平均分数 | ArguAna | ClimateFEVER | CQADupstackRetrieval | DBPedia |  FEVER | FiQA2018 | HotpotQA | MSMARCO | NFCorpus |   NQ   | QuoraRetrieval | SCIDOCS | SciFact | Touche2020 | TRECCOVID |
|-----------------------------|:--------:|:-------:|:------------:|:--------------------:|:-------:|:------:|:--------:|:--------:|:-------:|:--------:|:------:|:--------------:|:-------:|:-------:|:----------:|:---------:|
| RocketQA v1                 |  52.24   |  38.46  |    28.73     |        31.39         |  72.10  | 68.70  |  40.52   |  73.07   |  35.72  |  48.43   | 48.44  |     74.23      |  24.93  |  56.48  |   52.54    |   89.79   |
| RocketQA v2                 |  50.85   |  36.57  |    25.39     |        28.76         |  69.52  | 67.36  |  37.41   |  71.27   |  37.37  |  49.29   | 45.70  |     71.85      |  23.57  |  51.85  |   58.22    |   88.67   |
| bge‑large‑en‑v1.5           |  61.19   |  57.56  |    43.09     |        41.89         |  77.26  | 85.39  |  52.91   |  84.72   |  35.52  |  56.94   | 48.86  |     88.43      |  38.28  |  71.98  |   44.95    |   90.00   |
| repllama‑v1‑7b‑lora‑passage |  58.00   |  40.16  |    42.07     |        39.53         |  72.62  | 79.58  |  53.37   |  84.29   |  34.55  |  58.04   | 50.81  |     87.43      |  32.33  |  72.19  |   40.18    |   82.87   |
| NV‑Embed‑v1                 |  65.24   |  60.28  |    45.17     |        48.14         |  80.19  | 86.78  |  69.24   |  88.36   |  39.73  |  59.40   | 66.70  |     88.35      |  34.27  |  75.17  |   42.50    |   94.33   |
| bge‑en‑icl (zero‑shot)      |  69.29   |  77.83  |    57.88     |        45.69         |  82.04  | 92.50  |  65.78   |  92.76   |  39.97  |  61.84   | 69.64  |     90.22      |  41.14  |    75.13  |   56.56    |   90.33   |
| LLARA-passage               |  60.11   |  38.77  |    34.58     |        36.19         |  75.50  | 81.02  |  51.72   |  86.36   |  38.81  |  57.69   | 56.85  |     80.58      |  30.15  |  73.17  |   67.20    |   93.07   |

MTEB-Retrieval 数据集, Recall@10分数：
| 模型                        | 平均分数 | ArguAna | ClimateFEVER | CQADupstackRetrieval | DBPedia |  FEVER | FiQA2018 | HotpotQA | MSMARCO | NFCorpus |   NQ   | QuoraRetrieval | SCIDOCS | SciFact | Touche2020 | TRECCOVID |
|-----------------------------|:--------:|:-------:|:------------:|:--------------------:|:-------:|:------:|:--------:|:--------:|:-------:|:--------:|:------:|:--------------:|:-------:|:-------:|:----------:|:---------:|
| RocketQA v1                 |  46.12   |  75.61  |    25.28     |        41.50         |  22.04  | 83.88  |  39.45   |  56.68   |  62.96  |  14.22   | 73.05  |     88.59      |  14.09  |  73.44  |   19.18    |   1.79    |
| RocketQA v2                 |  44.45   |  71.19  |    24.28     |        38.53         |  21.45  | 82.86  |  36.80   |  55.21   |  64.68  |  13.43   | 68.88  |     87.09      |  13.27  |  68.44  |   18.88    |   1.76    |
| bge‑large‑en‑v1.5           |  54.59   |  90.26  |    39.13     |        55.23         |  26.44  | 93.39  |  51.45   |  76.87   |  63.54  |  19.37   | 76.32  |     95.74      |  24.92  |  88.49  |   15.65    |   2.03    |
| repllama‑v1‑7b‑lora‑passage |  52.95   |  78.88  |    40.03     |        52.53         |  25.89  | 92.01  |  52.19   |  69.54   |  63.60  |  19.04   | 78.50  |     95.38      |  19.91  |  88.27  |   16.62    |   1.82    |
| NV‑Embed‑v1                 |  58.78   |  93.95  |    41.07     |        64.66         |  28.67  | 95.24  |  70.62   |  85.19   |  69.15  |  18.45   | 89.16  |     95.92      |  21.27  |  90.02  |   15.94    |   2.36    |
| bge‑en‑icl (zero‑shot)      |  60.62   |  97.08  |    52.19     |        60.38         |  29.81  | 96.92  |  67.42   |  88.33   |  69.53  |  20.42   | 90.96  |     97.02      |  27.33  |   91.05   |   18.81    |   2.11    |
| LLARA-passage               |  52.30   |  76.17  |    32.52     |        47.91         |  26.33  | 90.48  |  51.09   |  71.16   |  67.82  |  17.67   | 81.89  |     92.54      |  18.12  |  86.80  |   21.81    |   2.23    |

MTEB-Retrieval 数据集, NDCG@10分数：
| 模型                        | 平均分数 | ArguAna | ClimateFEVER | CQADupstackRetrieval | DBPedia |  FEVER | FiQA2018 | HotpotQA | MSMARCO | NFCorpus |   NQ   | QuoraRetrieval | SCIDOCS | SciFact | Touche2020 | TRECCOVID |
|-----------------------------|:--------:|:-------:|:------------:|:--------------------:|:-------:|:------:|:--------:|:--------:|:-------:|:--------:|:------:|:--------------:|:-------:|:-------:|:----------:|:---------:|
| RocketQA v1                 |  44.74   |  47.16  |    21.02     |        32.12         |  37.53  | 70.30  |  32.89   |  55.21   |  41.93  |  29.65   | 53.26  |     76.44      |  13.63  |  59.85  |   30.37    |   69.75   |
| RocketQA v2                 |  43.09   |  44.66  |    19.15     |        29.51         |  35.75  | 69.00  |  30.34   |  53.56   |  43.59  |  29.38   | 50.16  |     74.22      |  12.82  |  55.08  |   30.60    |   68.56   |
| bge‑large‑en‑v1.5           |  53.68   |  65.17  |    32.75     |        43.05         |  43.69  | 85.09  |  44.69   |  72.57   |  41.90  |  38.35   | 54.42  |     89.14      |  23.37  |  75.50  |   23.01    |   72.48   |
| repllama‑v1‑7b‑lora‑passage |  51.81   |  49.19  |    32.57     |        40.75         |  41.80  | 81.27  |  45.47   |  67.27   |  41.23  |  37.77   | 59.24  |     88.15      |  18.93  |  75.74  |   23.90    |   73.88   |
| NV‑Embed‑v1                 |  58.86   |  68.30  |    34.37     |        50.27         |  48.29  | 86.58  |  62.90   |  79.92   |  46.48  |  37.98   | 71.22  |     89.20      |  20.16  |  78.30  |   23.98    |   84.91   |
| bge‑en‑icl (zero‑shot)      |  61.62   |  82.34  |    45.33     |        47.27         |  50.60  | 91.91  |  59.13   |  84.90   |  46.78  |  40.66   | 73.85  |     91.03      |  25.46  |  77.91  |   30.71    |   76.38   |
| LLARA-passage               |  52.48   |  47.51  |    26.13     |        37.26         |  44.12  | 81.09  |  43.98   |  69.17   |  45.49  |  37.07   | 61.76  |     82.29      |  17.30  |  76.07  |   36.73    |   81.30   |



## Reference

[1] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, Ali Farhadi: Matryoshka Representation Learning. NeurIPS 2022.

[2] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, Jimmy Lin: Fine-Tuning LLaMA for Multi-Stage Text Retrieval. arXiv 2023.

[3] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighof: C-Pack: Packaged Resources To Advance General Chinese Embedding. SIGIR 2024.

[4] Niklas Muennighoff, Nouamane Tazi, Loic Magne, Nils Reimers: MTEB: Massive Text Embedding Benchmark. EACL 2023.

[5] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping: NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. ICLR 2025.

[6] Zheng Liu, Chaofan Li, Shitao Xiao, Yingxia Shao, Defu Lian: Llama2Vec: Unsupervised Adaptation of Large Language Models for Dense Retrieval. ACL 2024.

[7] Chaofan Li, MingHao Qin, Shitao Xiao, Jianlyu Chen, Kun Luo, Yingxia Shao, Defu Lian, Zheng Liu: Making Text Embedders Few-Shot Learners. ICLR 2025.

[8] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, Haifeng Wang: RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. NAACL 2021

[9] Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, Ji-Rong Wen: RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking. EMNLP 2021
