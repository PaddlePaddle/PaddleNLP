# 预训练

[LLaMA v1/v2](./llama)、[GPT-3](./gpt-3) 目录中提供了模型预训练的数据准备和训练细节，后续我们将支持更多的模型预训练。


```
# 千问模型预训练
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./qwen/pretrain_argument_stage2.json
```


## 模型预训练支持的分布式能力一览

模型|能力|||||||||||||
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
||Data Parallelism|Tensor Parallelism|Pipeline Parallelism|sequence parallelism|Flash Attention|Sharding Stage1 ||Stage2||Stage3||Selective Recompute|
|||||||recompute|DP|recompute|DP|recompute|DP||
LLaMA-65B   |✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|
LLaMA2-70B  |✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|
BaiChuan-13B|✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|
GPT3        |✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|✅|
Qwen-7B     |✅|✅|✅|󠀠󠀠󠀠⬜|✅|⬜|✅|✅|✅|✅|✅|✅|
Qwen-14B    |✅|✅|✅|⬜|✅|⬜|✅|✅|✅|✅|✅|✅|
OPT 66B     |✅|⬜|⬜|⬜|❌|⬜|⬜|⬜|⬜|⬜|⬜|🚧|
Bloom-176B  |✅|✅|⬜|⬜|❌|⬜|⬜|⬜|⬜|⬜|⬜|🚧|
ChatGLM-6B  |✅|✅|⬜|⬜|❌|⬜|⬜|⬜|⬜|⬜|⬜|🚧|
GLM 130B    |✅|✅|⬜|⬜|❌|⬜|⬜|⬜|⬜|⬜|⬜|🚧|

* ✅: 已支持，Supported
* 🚧: 部分支持，In Progress
* ❌: 暂不支持，Not Supported


## 模型权重支持列表
上表中展示的是部分模型权重，支持的所有模型如下：

```
* LLaMA系列
  - facebook/llama-7b [英文]
  - facebook/llama-13b [英文]
  - facebook/llama-65b [英文]
  - meta-llama/Llama-2-7b [英文]
  - meta-llama/Llama-2-7b-chat [英文]
  - meta-llama/Llama-2-13b [英文]
  - meta-llama/Llama-2-13b-chat [英文]
  - meta-llama/Llama-2-70b [英文]
  - baichuan-inc/Baichuan-7B [中文]
  - baichuan-inc/Baichuan-13B-Base [中文]
  - baichuan-inc/Baichuan-13B-Chat [中文]
  - baichuan-inc/Baichuan2-7B-Base [中文]
  - baichuan-inc/Baichuan2-7B-Chat [中文]
  - baichuan-inc/Baichuan2-13B-Base [中文]
  - baichuan-inc/Baichuan2-13B-Chat [中文]
  - FlagAlpha/Llama2-Chinese-7b-Chat [中文]
  - FlagAlpha/Llama2-Chinese-13b-Chat [中文]
  - idea-ccnl/ziya-llama-13b-v1 [中文]
  - linly-ai/chinese-llama-2-7b [中文]
  - linly-ai/chinese-llama-2-13b [中文]
* ChatGLM系列
  - THUDM/chatglm-6b-v1.1 [中文]
  - THUDM/chatglm2-6b [中文]
* BLOOM系列
  - bigscience/bloom-7b1 [英文]
  - bigscience/bloomz-7b1 [多语言]
  - bigscience/bloomz-7b1-mt [多语言]
* Qwen系列
  - qwen/qwen-7b [中文]
  - qwen/qwen-7b-chat [中文]
  - qwen/qwen-14b [中文]
  - qwen/qwen-14b-chat [中文]
```


## 预训练性能
以下测试结果基于

机器环境： A100 80G * 8, CUDA 11.8, NCCL 2.15

训练配置：单个step, global batch size为 256K token.
```
paddle commit id              : 9bf5a86f13aa85b38715c015693f973290c8f9da
paddlenlp commit id           : 0731766c01ae85891b5bbe770eb397eec2287cbb
```
| 模型 | 序列长度 | 分布式策略 | 速度(`tokens/card/sec`) | 显存占用(`MB^1`) | 配置文件
| :-: | :-: | :-: | :-: | :-: |  :-: |
| `baichuan-inc/Baichuan2-13B-Base` |4096 | sd8 |    1366  | 74767MB | `./llama/pretrain_baichuan2-13b-sd8-stage2.json`
| `baichuan-inc/Baichuan2-7B-Base` |4096 | tp2sd4 |     3570  |58363MB | `./llama/pretrain_baichuan2-7b-tp2sd4-stage2.json`
| `facebook/llama-13b` |4096 | tp2sd4 |     1980  | 64402MB | `./llama/pretrain_llama-13b-tp2sd4-stage2.json`
| `facebook/llama-7b` |4096 | tp2sd4 |     3740  | 52092MB | `./llama/pretrain_llama-7b-tp2sd4-stage2.json`
| `FlagAlpha/Llama2-Chinese-13b-Chat` |4096 | tp2sd4 |     1978 | 64185MB | `./llama/pretrain_flagalpha-llama2-13b-tp2sd4-stage2.json`
| `FlagAlpha/Llama2-Chinese-7b-Chat` |4096 | tp2sd4 |     3760 | 52092MB | `./llama/pretrain_flagalpha-llama2-7b-tp2sd4-stage2.json`
| `idea-ccnl/ziya-llama-13b-v1` |4096 | tp2sd4 |     1973 | 64233MB | `./llama/pretrain_ziya-llama-13b-tp2sd4-stage2.json`
| `linly-ai/chinese-llama-2-7b` |4096 | tp2sd4 |     3742 | 51751MB | `./llama/pretrain_linly-llama2-7b-tp2sd4-stage2.json`
| `meta-llama/Llama-2-13b` |4096 | tp2sd4 |     1980 | 64199MB | `./llama/pretrain_llama2-13b-tp2sd4-stage2.json`
| `meta-llama/Llama-2-7b` |4096 | tp2sd4 |     3756 | 52092MB | `./llama/pretrain_llama2-7b-tp2sd4-stage2.json`
| `qwen/qwen-7b` |4096 | tp2sd4 |     3602 | 65448MB | `./qwen/pretrain_qwen-7b-tp2sd4-stage2.json`

注：
1. 显存占用(MB)使用的是 `max_memory_allocated`, 实际物理显存会占用更多，大约多2-3GB.
2. 速度会有小幅波动，例如 `facebook/llama-7b` 和 `meta-llama/Llama-2-7b` 是相同训练配置。
