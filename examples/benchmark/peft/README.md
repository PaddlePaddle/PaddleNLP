# Benchmark Results

### 配置

- 硬件: A100-80G with NVLink, 具体卡数见表
- Torch环境: 见torch/requirements.txt
- FP16配置: torch 使用 cuda amp fp16, paddle 使用 fp16 O2 opt level, intokens 设置为 1024, dataloader_num_workers 设置为 4

### Bloom

- 数据: 10k条[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | **Speedup** |
|---------------|----------|----------|------------|--------------|---------------------------|-------------|--------------------------|---------|
| Bloomz-7b1-mt | LoRA     | 1        | 4          |              | 3344.15                   |             | 1980.32                  | **69%**    |
| Bloomz-7b1-mt | Finetune | 4        | 8          | MP 4         | 7421.09                   | ZeRO 3      | 1702.00                  | **336%**    |
| Bloomz-7b1-mt | Finetune | 4        | 16         | MP 4         | 8214.55                   | ZeRO 3      | 2849.90                  | **188%**     |

###### 多卡分布式实验记录

- Finetuning with 4 GPUs

| Model          | Setup           | Paddle Effective Tokens /s | Torch Effective Tokens /s  |  Speedup  |
|----------------|-----------------|----------------------------|----------------------------|-----------|
| Bloomz-7b1-mt  | bsz 8 MP4       |       7421.09              |         N/A                |   N/A     |
| Bloomz-7b1-mt  | bsz 8 ZeRO 3    |       6063.23              |     1702.00                |   256%    |
| Bloomz-7b1-mt  | bsz 8 ZeRO 2    |      5191.47               |     1891.16                |   175%    |
| Bloomz-7b1-mt  | bsz 16 MP4      |      8214.55               |         N/A                |   N/A     |
| Bloomz-7b1-mt  | bsz 16 ZeRO 3   |      5822.23               |      2849.90               |   104     |
| Bloomz-7b1-mt  | bsz 16 ZeRO 2   |    5572.13                 |     2719.92                |   105%    |


### Llama

- 数据: 使用10k条[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

| Model     | Method   | Num GPUs | Batch Size  | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|-----------|----------|----------|-------------|--------------|---------------------------|-------------|--------------------------|---------|
| Llama-7b  | LoRA     | 1        | 4           |              |  3566.04                  |             | 1895.90                  |  **88%**  |
| Llama-13b | LoRA     | 1        | 8           | recompute    |  1511.58                  | gradient ckpt |    768.26              |  **97%**  |
| Llama-7b  | Finetune | 4        | 8           | MP4          |  3841.61                  | ZeRO 2      | 1621.52                  |  **124%**  |
| Llama-7b  | Finetune | 4        | 16          | sharding 2   |  4602.34                  | ZeRO 2      | 2465.55                  |  **127%**  |
| Llama-13b | Finetune | 4        | 8           | MP4 recompute|  1667.90                  | ZeRO 3      | 736.19                   |  **124%**  |
| Llama-65b | LoRA     | 4        | 8           | MP4 recompute|  888.60                   | gradient ckpt, bits 4, max_memory_MB 50000, qlora        | 327.75          |  **171%** |
| Llama-65b | LoRA     | 4        | 16          | MP4 recompute|  900.78                   | gradient ckpt, bits 4, max_memory_MB 50000, qlora        | 405.90          |  **122%** |


###### 多卡分布式实验记录

- Finetuning with 4 GPUs

| Model     | Setup         | Paddle Effective Tokens /s | Torch Effective Tokens /s  |  Speedup  |
|-----------|---------------|----------------------------|----------------------------|-----------|
| LLaMA-7b  | bsz 8 MP4     | **3841.61**                |  N/A                       | N/A       |
| LLaMA-7b  | bsz 8 ZeRO 3  | 4189.43                    |  1177.93                   | 256%       |
| LLaMA-7b  | bsz 8 ZeRO 2  | 4611.10                    |  1621.52                   | 184%       |
| LLaMA-7b  | bsz 16 (4*4) MP4 | 4829.47                 |  N/A                       | N/A       |
| LLaMA-7b  | bsz 16 ZeRO 3 | 4048.61                    |  2268.16                   | 78%       |
| LLaMA-7b  | bsz 16 ZeRO 2 | **3463.45**                |  2465.55                   | 40%       |
| LLaMA-13b | bsz 8 MP4 recompute |  **2509.50**         |  N/A                       | N/A       |
| LLaMA-13b | bsz 8 ZeRO 3  | 1867.99                    |  736.19                    | 154%        |
| LLaMA-13b | bsz 8 ZeRO 2  | 1201.75                    |  OOM                       | N/A       |


### ChatGLM

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|---------------|----------|----------|------------|--------------|---------------------------|-------------|--------------------------|---------|
| chatglm-6b    | LoRA     | 1        | 4          |              |        3472.79            |             |       1866.48            | **86%**    |
| chatglm-6b    | Finetune | 4        | 8          |   MP 4       |        4564.94            |   ZeRO 2    |       2124.17            | **115%**    |
| chatglm-6b    | Finetune | 4        | 16         |   MP 4       |        4972.21            |   ZeRO 3    |       3191.35            | **56%**    |


###### 多卡分布式实验记录

- Finetuning with 4 GPUs

| Model     | Setup           | Paddle Effective Tokens /s | Torch Effective Tokens /s  |  Speedup  |
|-----------|-----------------|----------------------------|----------------------------|-----------|
| chatglm-6b  | bsz 8 MP4     |  4564.94                   |         N/A                |   N/A     |
| chatglm-6b  | bsz 8 ZeRO 3  |    6480.36                 |         1840.99            |   252%    |
| chatglm-6b  | bsz 8 ZeRO 2  |    4707.74                 |         2124.17            |   122%     |
| chatglm-6b  | bsz 16 MP4    |    4972.21                 |         N/A                |   N/A     |
| chatglm-6b  | bsz 16 ZeRO 3 |    5282.28                 |         3184.26            |   66%     |
| chatglm-6b  | bsz 16 ZeRO 2 |    5751.00                 |         3151.07            |   83%     |

### TODOs

- Enable Flash Attention
