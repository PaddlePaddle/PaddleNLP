# Benchmark Results

### 配置

- 硬件: A100-80G with NVLink, 具体卡数见表
- Torch环境: 见torch/requirements.txt
- FP16配置: torch 使用 cuda amp fp16, paddle 使用 fp16 O2 opt level

### Bloom

- 数据: 10k条[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | **Speedup** |
|---------------|----------|----------|------------|--------------|---------------------------|-------------|--------------------------|---------|
| Bloomz-7b1-mt | LoRA     | 1        | 4          |              | 2293.46                   |             | 1980.32                  | **16%**    |
| Bloomz-7b1-mt | Finetune | 4        | 8          | MP 4         | 2873.13                   | ZeRO 3      | 1702.00                  | **69%**    |
| Bloomz-7b1-mt | Finetune | 4        | 16         | MP 4         | 2853.83                   | ZeRO 3      | 2849.90                  | **0%**     |

###### 多卡分布式实验记录

- Finetuning with 4 GPUs

| Model          | Setup           | Paddle Effective Tokens /s | Torch Effective Tokens /s  |  Speedup  |
|----------------|-----------------|----------------------------|----------------------------|-----------|
| Bloomz-7b1-mt  | bsz 8 MP4     |       2873.13           |         N/A                |   N/A     |
| Bloomz-7b1-mt  | bsz 8 ZeRO 3  |       N/A               |     1702.00                |   N/A     |
| Bloomz-7b1-mt  | bsz 8 ZeRO 2  |      2172.40            |     1891.16                |   15%     |
| Bloomz-7b1-mt  | bsz 16 MP4    |      2853.83            |         N/A                |   N/A     |
| Bloomz-7b1-mt  | bsz 16 ZeRO 3 |      N/A                |      2849.90               |   N/A     |
| Bloomz-7b1-mt  | bsz 16 ZeRO 2 |    2604.56 (accumulations 2) |  2719.92              |   -4%     |


### Llama

- 数据: 使用10k条[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

| Model     | Method   | Num GPUs | Batch Size  | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|-----------|----------|----------|-------------|--------------|---------------------------|-------------|--------------------------|---------|
| Llama-7b  | LoRA     | 1        | 4           |              |  1986.63                  |             | 1895.90                  |  **5%**  |
| Llama-13b | LoRA     | 1        | 8           | recompute    |  838.78                   | gradient ckpt |    768.26              |  **9%**  |
| Llama-7b  | Finetune | 4        | 8           | MP4          |  2213.46                  | ZeRO 2      | 1621.52                  |  **36%**  |
| Llama-7b  | Finetune | 4        | 16          | sharding 2   |  2804.19                  | ZeRO 2      | 2465.55                  |  **14%**  |
| Llama-13b | Finetune | 4        | 8           | MP4 recompute|  1651.50                  | ZeRO 3      | 736.19                   |  **124%**  |
| Llama-65b | LoRA     | 4        | 8           | MP4 recompute|  474.36                   | gradient ckpt, bits 4, max_memory_MB 50000, qlora        | 327.75          |  **45%** |
| Llama-65b | LoRA     | 4        | 16          | MP4 recompute|  452.20                   | gradient ckpt, bits 4, max_memory_MB 50000, qlora        | 405.90          |  **11%** |


###### 多卡分布式实验记录

- Finetuning with 4 GPUs

| Model     | Setup         | Paddle Effective Tokens /s | Torch Effective Tokens /s  |  Speedup  |
|-----------|---------------|----------------------------|----------------------------|-----------|
| LLaMA-7b  | bsz 8 MP4     | **2213.46**                |  N/A                       | N/A       |
| LLaMA-7b  | bsz 8 ZeRO 3  | 1256.82                    |  1177.93                   | 7%       |
| LLaMA-7b  | bsz 8 ZeRO 2  | 1781.27                    |  1621.52                   | 10%       |
| LLaMA-7b  | bsz 16 (8*2) MP4 | 2630.30                 |  N/A                       | N/A       |
| LLaMA-7b  | bsz 16 ZeRO 3 | 2115.58                    |  2268.16                   | -7%       |
| LLaMA-7b  | bsz 16 ZeRO 2 | **2804.19**                |  2465.55                   | 14%       |
| LLaMA-13b | bsz 8 MP4 recompute |  **1651.50**         |  N/A                       | N/A       |
| LLaMA-13b | bsz 8 ZeRO 3  | 747.17                     |  736.19                    | 1%        |
| LLaMA-13b | bsz 8 ZeRO 2  | OOM                        |  OOM                       | N/A       |


### ChatGLM

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|---------------|----------|----------|------------|--------------|---------------------------|-------------|--------------------------|---------|
| chatglm-6b    | LoRA     | 1        | 4          |              |        2654.73            |             |       1866.48            | **42%**    |
| chatglm-6b    | Finetune | 4        | 8          |   MP 4       |        3109.83            |   ZeRO 2    |       2124.17            | **46%**    |
| chatglm-6b    | Finetune | 4        | 16         |   MP 4       |        3569.95            |   ZeRO 3    |       3191.35            | **12%**    |


###### 多卡分布式实验记录

- Finetuning with 4 GPUs

| Model     | Setup           | Paddle Effective Tokens /s | Torch Effective Tokens /s  |  Speedup  |
|-----------|-----------------|----------------------------|----------------------------|-----------|
| chatglm-6b  | bsz 8 MP4     |  3109.83                   |         N/A                |   N/A     |
| chatglm-6b  | bsz 8 ZeRO 3  |    N/A                     |         1840.99            |   N/A     |
| chatglm-6b  | bsz 8 ZeRO 2  |    2457.88                 |         2124.17            |   16%     |
| chatglm-6b  | bsz 16 MP4    |    3549.68                 |         N/A                |   N/A     |
| chatglm-6b  | bsz 16 ZeRO 3 |    N/A                     |         3184.26            |   N/A     |
| chatglm-6b  | bsz 16 ZeRO 2 |    3462.83                 |         3151.07            |   10%     |

### TODOs

- Enable Flash Attention
