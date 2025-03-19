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
本仓库主要面向其中的中英文检索任务（Retrieval），并额外支持针对 MSMARCO-Title 的评估。

评估脚本为 `evaluation/eval_mteb.sh`, 支持5个模型：
LLARA ([LLARA-passage](https://huggingface.co/BAAI/LLARA-passage)),
NV-Embed ([NV-Embed-v1](https://huggingface.co/nvidia/NV-Embed-v1)),
BGE-EN-ICL([BGE-EN-ICL](https://huggingface.co/BAAI/bge-en-icl)),
RepLLaMA([repllama-v1-7b-lora-passage](https://huggingface.co/castorini/repllama-v1-7b-lora-passage)),
BGE([bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5))

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


评估结果如下：
| Model                       | Max&nbsp;Length | ArguAna |           |        | ClimateFEVER |           |        | CQADupstackRetrieval |           |        | DBPedia |           |        |  FEVER  |           |        | FiQA2018 |           |        | HotpotQA |           |        | MSMARCO |           |        | NFCorpus |           |        |    NQ   |           |        | QuoraRetrieval |           |        | SCIDOCS |           |        | SciFact |           |        | Touche2020 |           |        | TRECCOVID |           |        |
|-----------------------------|:----------:|:-------:|:---------:|:------:|:------------:|:---------:|:------:|:--------------------:|:---------:|:------:|:-------:|:---------:|:------:|:-------:|:---------:|:------:|:--------:|:---------:|:------:|:--------:|:---------:|:------:|:-------:|:---------:|:------:|:--------:|:---------:|:------:|:-------:|:---------:|:------:|:--------------:|:---------:|:------:|:-------:|:---------:|:------:|:-------:|:---------:|:------:|:----------:|:---------:|:------:|:---------:|:---------:|:------:|
|                             |            | NDCG@10 | Recall@10 | MRR@10 |    NDCG@10   | Recall@10 | MRR@10 |        NDCG@10       | Recall@10 | MRR@10 | NDCG@10 | Recall@10 | MRR@10 | NDCG@10 | Recall@10 | MRR@10 |  NDCG@10 | Recall@10 | MRR@10 |  NDCG@10 | Recall@10 | MRR@10 | NDCG@10 | Recall@10 | MRR@10 |  NDCG@10 | Recall@10 | MRR@10 | NDCG@10 | Recall@10 | MRR@10 |     NDCG@10    | Recall@10 | MRR@10 | NDCG@10 | Recall@10 | MRR@10 | NDCG@10 | Recall@10 | MRR@10 |   NDCG@10  | Recall@10 | MRR@10 |  NDCG@10  | Recall@10 | MRR@10 |
| RocketQA v1                 |    512     |  47.16  |           |        |    21.02     |           |        |        32.12         |           |        |  37.53  |           |        |  70.30  |           |        |  32.89   |           |        |  55.21   |           |        |  41.93  |           |        |  29.65   |           |        |  53.26  |           |        |     76.44      |           |        |  13.63  |           |        |  59.85  |           |        |   30.37    |           |        |   69.75   |           |        |
| RocketQA v2                 |    512     |  44.66  |           |        |    19.15     |           |        |        29.51         |           |        |  35.75  |           |        |  69.00  |           |        |  30.34   |           |        |  53.56   |           |        |  43.59  |           |        |  29.38   |           |        |  50.16  |           |        |     74.22      |           |        |  12.82  |           |        |  55.08  |           |        |   30.60    |           |        |   68.56   |           |        |
| bge‑large‑en‑v1.5          |    512     |  65.17  |   90.26   | 57.56  |    32.75     |   39.13   | 43.09  |        43.05         |   55.23   | 41.89  |  43.69  |   26.44   | 77.26  |  85.09  |   93.39   | 85.39  |  44.69   |   51.45   | 52.91  |  72.57   |   76.87   | 84.72  |  41.90  |   63.54   | 35.52  |  38.35   |   19.37   | 56.94  |  54.42  |   76.32   | 48.86  |     89.14      |   95.74   | 88.43  |  23.37  |   24.92   | 38.28  |  75.50  |   88.49   | 71.98  |   23.01    |   15.65   | 44.95  |   72.48   |   2.03    | 90.00  |
| repllama‑v1‑7b‑lora‑passage |    4096    |  49.19  |   78.88   | 40.16  |    32.57     |   40.03   | 42.07  |        40.75         |   52.53   | 39.53  |  41.80  |   25.89   | 72.62  |  81.27  |   92.01   | 79.58  |  45.47   |   52.19   | 53.37  |  67.27   |   69.54   | 84.29  |  41.23  |   63.60   | 34.55  |  37.77   |   19.04   | 58.04  |  59.24  |   78.50   | 50.81  |     88.15      |   95.38   | 87.43  |  18.93  |   19.91   | 32.33  |  75.74  |   88.27   | 72.19  |   23.90    |   16.62   | 40.18  |   73.88   |   1.82    | 82.87  |
| NV‑Embed‑v1                 |    4096    |  68.30  |   93.95   | 60.28  |    34.37     |   41.07   | 45.17  |        50.27         |   64.66   | 48.14  |  48.29  |   28.67   | 80.19  |  86.58  |   95.24   | 86.78  |  62.90   |   70.62   | 69.24  |  79.92   |   85.19   | 88.36  |  46.48  |   69.15   | 39.73  |  37.98   |   18.45   | 59.40  |  71.22  |   89.16   | 66.70  |     89.20      |   95.92   | 88.35  |  20.16  |   21.27   | 34.27  |  78.30  |   90.02   | 75.17  |   23.98    |   15.94   | 42.50  |   84.91   |   2.36    | 94.33  |
| bge‑en‑icl (zero‑shot)      |    4096    |  82.34  |           |        |    45.33     |           |        |        47.27         |           |        |  50.60  |           |        |  91.91  |           |        |  59.13   |           |        |  84.90   |           |        |  46.78  |           |        |  40.66   |           |        |  73.85  |           |        |     91.03      |           |        |  25.46  |           |        |  77.91  |           |        |   30.71    |           |        |   76.38   |           |        |
| LLARA-passage               |    4096    |  47.51  |   76.17   | 38.77  |    26.13     |   32.52   | 34.58  |        37.26         |   47.91   | 36.19  |  44.12  |   26.33   | 75.50  |  81.09  |   90.48   | 81.02  |  43.98   |   51.09   | 51.72  |  69.17   |   71.16   | 86.36  |  45.49  |   67.82   | 38.81  |  37.07   |   17.67   | 57.69  |  61.76  |   81.89   | 56.85  |     82.29      |   92.54   | 80.58  |  17.30  |   18.12   | 30.15  |  76.07  |   86.80   | 73.17  |   36.73    |   21.81   | 67.20  |   81.30   |   2.23    | 93.07  |



| Model                       | Max Length | MSMARCO-Title |           |        |
|-----------------------------|:----------:|:-------------:|:---------:|:------:|
|                             |            |    NDCG@10    | Recall@10 | MRR@10 |
| RocketQA v1                 |    512     |               |           | 36.90  |
| RocketQA v2                 |    512     |               |           | 38.90  |
| bge-large-en-v1.5           |    512     |     41.96     |   64.24   | 35.30  |
| repllama-v1-7b-lora-passage |    4096    |     45.13     |   68.18   | 38.24  |
| NV-Embed-v1                 |    4096    |     45.21     |   68.02   | 38.39  |
| bge-en-icl (zero-shot)      |    4096    |               |           | 42.77  |
| LLARA-passage               |    4096    |     49.87     |   72.59   | 43.04  |



## Reference

[1] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, Ali Farhadi: Matryoshka Representation Learning. NeurIPS 2022.

[2] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, Jimmy Lin: Fine-Tuning LLaMA for Multi-Stage Text Retrieval. arXiv 2023.

[3] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighof: C-Pack: Packaged Resources To Advance General Chinese Embedding. SIGIR 2024.

[4] Niklas Muennighoff, Nouamane Tazi, Loic Magne, Nils Reimers: MTEB: Massive Text Embedding Benchmark. EACL 2023.

[5] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping: NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. arXiv 2024.

[6] Zheng Liu, Chaofan Li, Shitao Xiao, Yingxia Shao, Defu Lian: Llama2Vec: Unsupervised Adaptation of Large Language Models for Dense Retrieval. ACL 2024

[7] Chaofan Li, MingHao Qin, Shitao Xiao, Jianlyu Chen, Kun Luo, Yingxia Shao, Defu Lian, Zheng Liu: Making Text Embedders Few-Shot Learners. arXiv 2024
