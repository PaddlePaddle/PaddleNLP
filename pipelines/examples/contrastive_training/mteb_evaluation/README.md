# MTEB基准评估
[MTEB](https://github.com/embeddings-benchmark/mteb)
是一个大规模文本嵌入基准，包含了丰富的向量评估任务和数据集，涉及多语言、多领域的向量检索。
本仓库主要面向其中的中英文的检索任务（Retrieval），并以SciFact数据集作为主要示例。

## 环境准备
通过Anaconda创建和配置环境：
```
conda create -n mteb_paddle python=3.8
conda activate mteb_paddle
```
安装相关包：
```
conda install paddlepaddle-gpu==2.6.1 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
conda install nccl -c conda-forge
pip install paddlenlp==2.8.0
pip install mteb==1.12.13
```

## 模型评估
使用评估脚本`eval_mteb.py`：

- `base_model_name_or_path`: 模型名称或路径
- `output_folder`: 结果文件存储路径
- `task_name`：任务（数据集）名称，如SciFact
- `task_split`：测试查询集合，如test或dev
- `query_instruction`：查询前添加的提示文本，如'query: '或None
- `document_instruction`：文档前添加的提示文本，如'passage: '或None
- `pooling_method`：获取表示的方式，last表示取最后token，mean表示取平均，cls表示取`[CLS]`token \
- `max_seq_length`: 最大序列长度
- `eval_batch_size`: 模型预测的批次大小（单个GPU）
- `pad_token`：设置padding的token，可取unk_token、eos_token或pad_token
- `padding_side`：设置padding的位置，可取left或right
- `add_bos_token`：是否添加起始符，0表示不添加，1表示添加
- `add_eos_token`：是否添加结束符，0表示不添加，1表示添加

#### RepLLaMA向量检索模型
RepLLaMA的基础模型是LLaMA-2-7B，它在MS MARCO段落数据上进行了LoRA训练，模型参数开源在 [repllama-v1-7b-lora-passage](https://huggingface.co/castorini/repllama-v1-7b-lora-passage) 。
```
CUDA_VISIBLE_DEVICES=0 python eval_mteb.py \
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

#### BGE向量检索模型
BGE的基础模型是类BERT模型，以 [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) 模型为例。
```
CUDA_VISIBLE_DEVICES=0 python eval_mteb.py \
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

## 参考文献
[1] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, Jimmy Lin: Fine-Tuning LLaMA for Multi-Stage Text Retrieval.

[2] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighof: C-Pack: Packaged Resources To Advance General Chinese Embedding.
