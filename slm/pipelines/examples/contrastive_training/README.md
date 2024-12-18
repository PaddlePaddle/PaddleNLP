# 向量检索模型训练

## 安装

推荐安装 gpu 版本的[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)，以 cuda11.7的 paddle 为例，安装命令如下：

```
conda install nccl -c conda-forge
conda install paddlepaddle-gpu==2.6.1 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
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
model_name=castorini/repllama-v1-7b-lora-passage 或 NV-Embed-v1
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

## MTEB 评估
[MTEB](https://github.com/embeddings-benchmark/mteb)
是一个大规模文本嵌入评测基准，包含了丰富的向量检索评估任务和数据集。
本仓库主要面向其中的中英文检索任务（Retrieval），并以 SciFact 数据集作为主要示例。

评估 NV-Embed 向量检索模型（[NV-Embed-v1](https://huggingface.co/nvidia/NV-Embed-v1)）：
```
export CUDA_VISIBLE_DEVICES=0
python evaluation/eval_mteb.py \
       --base_model_name_or_path NV-Embed-v1 \
       --output_folder en_results/nv-embed-v1 \
       --query_instruction "Given a claim, find documents that refute the claim" \
       --task_name 'SciFact' \
       --eval_batch_size 8
```
结果文件保存在`en_results/nv-embed-v1/SciFact/last/no_model_name_available/no_revision_available/SciFact.json`，包含以下类似的评估结果：
```
'ndcg_at_1': 0.67667,
'ndcg_at_3': 0.73826,
'ndcg_at_5': 0.76662,
'ndcg_at_10': 0.783,
'ndcg_at_20': 0.7936,
'ndcg_at_100': 0.80206,
'ndcg_at_1000': 0.80444
```

评估 RepLLaMA 向量检索模型（[repllama-v1-7b-lora-passage](https://huggingface.co/castorini/repllama-v1-7b-lora-passage)）：
```
export CUDA_VISIBLE_DEVICES=0
python evaluation/eval_mteb.py \
       --base_model_name_or_path castorini/repllama-v1-7b-lora-passage \
       --output_folder en_results/repllama-v1-7b-lora-passage \
       --task_name SciFact \
       --task_split test \
       --query_instruction 'query: ' \
       --document_instruction 'passage: ' \
       --pooling_method last \
       --max_seq_length 512 \
       --eval_batch_size 2 \
       --pad_token unk_token \
       --padding_side right \
       --add_bos_token 0 \
       --add_eos_token 1
```
结果文件保存在`en_results/repllama-v1-7b-lora-passage/SciFact/last/no_revision_available/SciFact.json`，包含以下类似的评估结果：
```
'ndcg_at_1': 0.63,
'ndcg_at_3': 0.71785,
'ndcg_at_5': 0.73735,
'ndcg_at_10': 0.75708,
'ndcg_at_20': 0.7664,
'ndcg_at_100': 0.77394,
'ndcg_at_1000': 0.7794
```

评估 BGE 向量检索模型（[bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)）：
```
export CUDA_VISIBLE_DEVICES=0
python evaluation/eval_mteb.py \
       --base_model_name_or_path BAAI/bge-large-en-v1.5 \
       --output_folder en_results/bge-large-en-v1.5 \
       --task_name SciFact \
       --task_split test \
       --document_instruction 'Represent this sentence for searching relevant passages: ' \
       --pooling_method mean \
       --max_seq_length 512 \
       --eval_batch_size 32 \
       --pad_token pad_token \
       --padding_side right \
       --add_bos_token 0 \
       --add_eos_token 0
```
结果文件保存在`en_results/bge-large-en-v1.5/SciFact/mean/no_revision_available/SciFact.json`，包含以下类似的评估结果：
```
'ndcg_at_1': 0.64667,
'ndcg_at_3': 0.70359,
'ndcg_at_5': 0.7265,
'ndcg_at_10': 0.75675,
'ndcg_at_20': 0.76743,
'ndcg_at_100': 0.77511,
'ndcg_at_1000': 0.77939
```

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


## Reference

[1] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, Ali Farhadi: Matryoshka Representation Learning. NeurIPS 2022.

[2] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, Jimmy Lin: Fine-Tuning LLaMA for Multi-Stage Text Retrieval. arXiv 2023.

[3] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighof: C-Pack: Packaged Resources To Advance General Chinese Embedding. SIGIR 2024.

[4] Niklas Muennighoff, Nouamane Tazi, Loic Magne, Nils Reimers: MTEB: Massive Text Embedding Benchmark. EACL 2023.

[5] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping: NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. arXiv 2024.
