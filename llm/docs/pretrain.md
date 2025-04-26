# 大模型预训练介绍

PaddleNLP 大模型套件支持多种大模型的预训练，包括但不限于 LLaMA v1/v2、GPT-3、BaiChuan 和 Qwen 等,这些模型在自然语言处理领域有着广泛的应用，这些模型在自然语言处理（NLP）领域有广泛的应用，比如文本生成、机器翻译、情感分析等。即使是新手小白，也可以通过以下简单的步骤**快速上手**大模型的预训练。



## 准备工作

在开始之前，您需要安装 PaddleNLP 的最新版本，这里推荐安装 `develop` 版本，因为它包含了最新的功能和改进。

```bash
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

# 如果下载速度较慢，建议使用国内镜像，如百度镜像：
pip install --pre --upgrade paddlenlp -f https://mirror.baidu.com/paddlepaddle/whl/paddlenlp.html
```



接下来，您需要将 PaddleNLP 的代码克隆到本地：

```bash
# 您可以通过以下命令克隆 PaddleNLP 代码到本地：
git clone https://github.com/PaddlePaddle/PaddleNLP.git
# 如果克隆速度较慢，建议使用 Gitee 镜像进行克隆：
git clone https://gitee.com/PaddlePaddle/PaddleNLP.git

# 克隆完成后，进入 llm 目录，这是运行大模型预训练的目录
cd PaddleNLP/llm
```



## 数据制作

在开始预训练之前，您需要准备训练数据。PaddleNLP 提供了多种内置数据集，并支持自定义数据的制作，您可以参考以下文档来准备数据：

- [内置预练数据集及自定义数据制作](../dataset.md)
- [CLUECorpus2020 语料制作](../tools/preprocess/docs/CLUECorpus2020.md)
- [CLUECorpusSmall 语料制作](../tools/preprocess/docs/CLUECorpusSmall.md)
- [OpenWebText2 语料制作](../tools/preprocess/docs/OpenWebText2.md)
- [WuDaoCorpus2.0 Base 语料](../tools/preprocess/docs/WuDaoCorpusBase.md)



## 开始训练

为了方便用户运行测试本模型，本项目提供了处理好的100k 条doc 的训练样本：

```bash
# llama 模型数据下载
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx

# gpt 模型数据下载（可选）
# wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
# wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```bash
# 创建 data 目录
mkdir data

# 移动文件至 data 目录下
mv llama_openwebtext_100k.bin ./data
mv llama_openwebtext_100k.idx ./data
```

编译自定义算子（可选）：

```bash
cd ../slm/model_zoo/gpt-3/external_ops/
python3 setup.py install
cd -
```

运行预训练命令：

```bash
# llama 模型预训练
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./config/llama/pretrain_argument.json

# Qwen 模型预训练
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./config/qwen/pretrain_argument.json
```

预训练成功后，打印信息如下（以 Qwen 为例）：

```bash
Final pre-training config: Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  ...
  "vocab_size": 152064
}
Downloading shards:  25%|█████████████████████▎                                                               | 1/4 [00:43<02:11, 43.73s/it]
Downloading shards:  50%|██████████████████████████████████████████▌                                          | 2/4 [01:27<01:27, 43.84s/it]
Downloading shards:  75%|█████████████████████████████████████████████████████████▍                           | 3/4 [02:10<01:05, 43.92s/it]
Downloading shards: 100%|█████████████████████████████████████████████████████████████████▌                   | 4/4 [03:15<00:00, 43.95s/it]
```



## 注意事项

1. 建议使用 Paddle develop 版本训练，需要安装`pip install fast_dataindex visualdl==2.5.3`等相关缺失 whl 包。
2. `use_flash_attention`需在 A100机器开启，建议使用 CUDA 11.8环境。
3. `use_fused_rms_norm`需要安装[自定义 OP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/gpt-3/external_ops)。如果安装后仍然找不到算子，需要额外设置`PYTHONPATH`。
4. `continue_training`表示从现有的预训练模型加载训练。7B 模型初始 loss 大概为2.xx，随机初始化模型 loss 从11.x 左右下降。
5. 当前脚本为 Sharding 版本，需要4D 并行训练（数据、Sharding、张量、流水线并行）的用户，请参考`run_trainer_tp4pp2.sh`脚本。
6. 多机训练时，若各机器使用的训练数据文件位置相同（例如挂载共享硬盘情况），请指定`--share_folder true`使全局0号卡制作缓存数据。否则默认各台机器的0号卡独立制作缓存数据。
7. 若数据集文件夹中存在默认缓存文件夹`index-cache/`，则额外指定的`--data_cache`不生效，训练时优先加载默认缓存文件夹中的内容。

预训练使用了 PaddleNLP 的Trainer 模块，相关分布式策略使用，请参考[大模型 Trainer 混合并行训练教程](./llm_trainer.rst)。



## 模型预训练支持的分布式能力一览

| Model        | Data Parallelism | Tensor Parallelism | Pipeline Parallelism | Sequence Parallelism | Flash Attention | Selective Recompute | Sharding Stage1 + recompute | Sharding Stage1 + DP | Stage2 + recompute | Stage2 + DP | Stage3 + recompute | Stage3 + DP |
| ------------ | ---------------- | ------------------ | -------------------- | -------------------- | --------------- | ------------------- | --------------------------- | -------------------- | ------------------ | ----------- | ------------------ | ----------- |
| LLaMA-65B    | ✅                | ✅                  | ✅                    | ✅                    | ✅               | ✅                   | ✅                           | ✅                    | ✅                  | ✅           | ✅                  | ✅           |
| LLaMA2-70B   | ✅                | ✅                  | ✅                    | ✅                    | ✅               | ✅                   | ✅                           | ✅                    | ✅                  | ✅           | ✅                  | ✅           |
| BaiChuan-13B | ✅                | ✅                  | ✅                    | ✅                    | ✅               | ✅                   | ✅                           | ✅                    | ✅                  | ✅           | ✅                  | ✅           |
| GPT3         | ✅                | ✅                  | ✅                    | ✅                    | ✅               | ✅                   | ✅                           | ✅                    | ✅                  | ✅           | ✅                  | ✅           |
| Qwen-7B      | ✅                | ✅                  | ✅                    | ⬜                    | ✅               | ✅                   | ⬜                           | ✅                    | ✅                  | ✅           | ✅                  | ✅           |
| Qwen-14B     | ✅                | ✅                  | ✅                    | ⬜                    | ✅               | ✅                   | ⬜                           | ✅                    | ✅                  | ✅           | ✅                  | ✅           |
| OPT 66B      | ✅                | ✅                  | ⬜                    | ⬜                    | ❌               | 🚧                   | ⬜                           | ⬜                    | ⬜                  | ⬜           | ⬜                  | ⬜           |
| Bloom-176B   | ✅                | ✅                  | ⬜                    | ⬜                    | ✅               | 🚧                   | ⬜                           | ⬜                    | ⬜                  | ⬜           | ⬜                  | ⬜           |
| ChatGLM-6B   | ✅                | ✅                  | ⬜                    | ⬜                    | ✅               | 🚧                   | ⬜                           | ⬜                    | ⬜                  | ⬜           | ⬜                  | ⬜           |
| ChatGLM2     | ✅                | ✅                  | ⬜                    | ⬜                    | ❌               | 🚧                   | ⬜                           | ⬜                    | ⬜                  | ⬜           | ⬜                  | ⬜           |
| GLM 130B     | ✅                | ✅                  | ⬜                    | ⬜                    | ❌               | 🚧                   | ⬜                           | ⬜                    | ⬜                  | ⬜           | ⬜                  | ⬜           |

* ✅: 已支持，Supported
* 🚧: 部分支持，In Progress
* ❌: 暂不支持，Not Supported



## 模型权重支持列表

上表中展示的是部分模型权重，支持的所有模型如下：

### LLaMA 系列

- baichuan-inc/Baichuan-13B-Base [中文]
- baichuan-inc/Baichuan-13B-Chat [中文]
- baichuan-inc/Baichuan-7B [中文]
- baichuan-inc/Baichuan2-13B-Base [中文]
- baichuan-inc/Baichuan2-13B-Chat [中文]
- baichuan-inc/Baichuan2-7B-Base [中文]
- baichuan-inc/Baichuan2-7B-Chat [中文]
- facebook/llama-13b [英文]
- facebook/llama-65b [英文]
- facebook/llama-7b [英文]
- FlagAlpha/Llama2-Chinese-13b-Chat [中文]
- FlagAlpha/Llama2-Chinese-7b-Chat [中文]
- idea-ccnl/ziya-llama-13b-v1 [中文]
- linly-ai/chinese-llama-2-13b [中文]
- linly-ai/chinese-llama-2-7b [中文]
- meta-llama/Llama-2-13b [英文]
- meta-llama/Llama-2-13b-chat [英文]
- meta-llama/Llama-2-70b [英文]
- meta-llama/Llama-2-7b [英文]
- meta-llama/Llama-2-7b-chat [英文]

### ChatGLM 系列

- THUDM/chatglm-6b-v1.1 [中文]
- THUDM/chatglm2-6b [中文]

### BLOOM 系列

- bigscience/bloom-7b1 [英文]
- bigscience/bloomz-7b1 [多语言]
- bigscience/bloomz-7b1-mt [多语言]

### Qwen 系列

- qwen/qwen-7b [中文]
- qwen/qwen-7b-chat [中文]
- qwen/qwen-14b [中文]
- qwen/qwen-14b-chat [中文]



## 模型预训练性能

以下测试结果基于以下机器环境：

- **GPU**: A100 80G * 8, CUDA 11.8, NCCL 2.15
- **CPU**: Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz
- **内存**: 1 TB

```bash
paddle commit id: 9b36e53f24ac5f471b20de99e0cc3980f38b44ab
paddlenlp commit id: 0b246a609a3062e3c3256d87193b70277b5b07e0
```



## 模型性能测试汇总

| 模型                              | 序列长度 | 分布式策略    | 速度 (tokens/card/sec) | 显存占用 (MB) | 配置文件                                                 | 测试时间            |
| --------------------------------- | -------- | ------------- | ---------------------- | ------------- | -------------------------------------------------------- | ------------------- |
| FlagAlpha/Llama2-Chinese-13b-Chat | 4096     | tp2sd4_stage2 | 1980.22                | 64323         | ./llama/pretrain-flagalpha_llama2_13b-tp2sd4_stage2.json | 2023-11-27 21:42:38 |
| FlagAlpha/Llama2-Chinese-7b-Chat  | 4096     | tp2sd4_stage2 | 3744.62                | 52092         | ./llama/pretrain-flagalpha_llama2_7b-tp2sd4_stage2.json  | 2023-11-27 21:44:57 |
| baichuan-inc/Baichuan2-13B-Base   | 4096     | sd8_stage2    | 1354.99                | 74767         | ./baichuan/pretrain-baichuan2_13b-sd8_stage2.json        | 2023-11-27 21:51:26 |
| baichuan-inc/Baichuan2-7B-Base    | 4096     | tp2sd4_stage2 | 3542.45                | 58363         | ./baichuan/pretrain-baichuan2_7b-tp2sd4_stage2.json      | 2023-11-27 21:53:58 |
| facebook/llama-13b                | 4096     | tp2sd4_stage2 | 1969.64                | 64278         | ./llama/pretrain-llama_13b-tp2sd4_stage2.json            | 2023-11-27 21:58:03 |
| facebook/llama-7b                 | 4096     | tp2sd4_stage2 | 3754.73                | 52092         | ./llama/pretrain-llama_7b-tp2sd4_stage2.json             | 2023-11-27 22:00:30 |
| idea-ccnl/ziya-llama-13b-v1       | 4096     | tp2sd4_stage2 | 1968.34                | 63983         | ./llama/pretrain-ziya_llama_13b-tp2sd4_stage2.json       | 2023-11-27 22:04:35 |
| linly-ai/chinese-llama-2-7b       | 4096     | tp2sd4_stage2 | 3732.9                 | 51751         | ./llama/pretrain-linly_llama2_7b-tp2sd4_stage2.json      | 2023-11-27 22:06:58 |
| meta-llama/Llama-2-13b            | 4096     | tp2sd4_stage2 | 1975.63                | 64294         | ./llama/pretrain-llama2_13b-tp2sd4_stage2.json           | 2023-11-27 22:11:04 |
| meta-llama/Llama-2-7b             | 4096     | tp2sd4_stage2 | 3755.21                | 52092         | ./llama/pretrain-llama2_7b-tp2sd4_stage2.json            | 2023-11-27 22:13:34 |
| qwen/qwen-7b                      | 4096     | tp2sd4_stage2 | 3607.28                | 65448         | ./qwen/pretrain-qwen_7b-tp2sd4_stage2.json               | 2023-11-27 22:16:04 |

**说明**

- **速度单位**: `tokens/card/sec`，表示每张卡每秒需训练的 token 数。
- **速度波动**: 速度会有小幅波动，例如 `facebook/llama-7b` 和 `meta-llama/Llama-2-7b` 是相同训练配置。
- **显存占用**: 显存占用单位是 MB，使用的是 `max_memory_allocated` 获取显存，实际物理显存会占用更多，大约多 2-3GB。
