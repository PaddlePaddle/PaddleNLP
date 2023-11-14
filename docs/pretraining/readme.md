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


上表中展示的是部分模型权重，实际上：
