# Benchmark Results

### 配置

- 硬件: A100-80G with NVLink, 具体卡数见表
- Torch环境: 见torch/requirements.txt
- 数据: 中文模型使用10k条[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0), 英文模型使用10k条[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)

### Bloom

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|---------------|----------|----------|------------|--------------|---------------------------|-------------|--------------------------|---------|
| Bloomz-7b1-mt | LoRA     | 1        | 4          | fp16 O2      | 2293.46                   | fp16        | 1736.92                  | +32%    |
| Bloomz-7b1-mt | Finetune | 4        | 8          | fp16 O2 MP 4 | 2873.13                   | fp16 ZeRO 3 | 1634.58                  | +76%    |
| Bloomz-7b1-mt | Finetune | 4        | 16         | fp16 O2 MP 4 | 2853.83                   | fp16 ZeRO 3 | 2694.64                  | +6%     |


### Llama

| Model     | Method   | Num GPUs | Batch Size  | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|-----------|----------|----------|-------------|--------------|---------------------------|-------------|--------------------------|---------|
| Llama-7b  | LoRA     | 1        | 4           | fp16 O2      |  1986.63                 | fp16        | 1589.27                  |  +25%  |
| Llama-13b | LoRA     | 1        | 8           | fp16 O2, recompute |  838.78             | fp16, gradient ckpt    |    674.51 | +24%   |

###### Distributed Training Experiments

- Finetuning with 4 GPUs
- FP16: torch cuda amp fp16, paddle fp16 O2 opt level

| Model     | Framework | Setup         | Effective Tokens /s  |
|-----------|-----------|---------------|----------------------|
| LLaMA-7b  | torch     | bsz 8 ZeRO 3  | 1142.07              |
| LLaMA-7b  | torch     | bsz 8 ZeRO 2  | 1331.96              |
| LLaMA-7b  | torch     | bsz 16 ZeRO 3 | 2171.99              |
| LLaMA-7b  | torch     | bsz 16 ZeRO 2 | 2211.25              |
| LLaMA-13b | torch     | bsz 8 ZeRO 3  | 712.53               |
| LLaMA-13b | torch     | bsz 8 ZeRO 2  | OOM                  |

| Model     | Framework  | Setup         | Effective Tokens /s  |
|-----------|------------|---------------|----------------------|
| LLaMA-7b  | paddle     | bsz 8 MP 4  | 2208.06              |
| LLaMA-7b  | paddle     | bsz 8 ZeRO 2  | 2978.36              |
| LLaMA-7b  | paddle     | bsz 8 ZeRO 3  | 2265.16              |
| LLaMA-7b  | paddle     | bsz 16 MP 4  | OOM              |
| LLaMA-7b  | paddle     | bsz 16 ZeRO 2  | 4084.38              |

### TODOs

- Add ChatGLM
- Add Llama-30b and Llama-65b
- Enable Flash Attention
- Test sharding stage 3