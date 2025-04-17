# Benchmark Results

### 配置

- 硬件: A100-80G with NVLink, 具体卡数见表
- Torch 环境: 见 torch/requirements.txt
- FP16配置: torch 使用 cuda amp fp16, paddle 使用 fp16 O2 opt level, intokens 设置为 1024, 并开启了 flash attention

### Bloom

- 数据: 10k 条[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | **Speedup** |
|---------------|----------|----------|------------|--------------|---------------------------|-------------|--------------------------|---------|
| Bloomz-7b1-mt | LoRA     | 1        | 4          |              | 4097.03                   |             | 1980.32                  | **107%**    |
| Bloomz-7b1-mt | Finetune | 4        | 8          | MP 4         | 4136.69                   | ZeRO 3      | 1702.00                  | **143%**    |
| Bloomz-7b1-mt | Finetune | 4        | 16         | MP 4         | 4359.72                   | ZeRO 3      | 2849.90                  | **53%**     |

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

- 数据: 使用10k 条[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

| Model     | Method   | Num GPUs | Batch Size  | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|-----------|----------|----------|-------------|--------------|---------------------------|-------------|--------------------------|---------|
| Llama-7b  | LoRA     | 1        | 4           |              |  4406.23                  |             | 1895.90                  |  **132%**  |
| Llama-13b | LoRA     | 1        | 4           |              |  1975.94                  |             |    1101.85              |  **79%**  |
| Llama-13b | LoRA     | 1        | 8           | recompute    |  1869.60                  | gradient ckpt |    768.26              |  **143%**  |
| Llama-7b  | Finetune | 4        | 8           | MP4          |  3275.90                  | ZeRO 2      | 1621.52                  |  **102%**  |
| Llama-7b  | Finetune | 4        | 16          | sharding 2   |  6798.72                 | ZeRO 2      | 2465.55                  |  **176%**  |
| Llama-13b | Finetune | 4        | 8           | MP4 recompute|  1938.19                  | ZeRO 3      | 736.19                   |  **127%**  |
| Llama-65b | LoRA     | 4        | 8           | MP4 recompute|  840.57                   | gradient ckpt, bits 4, max_memory_MB 50000, qlora        | 327.75          |  **156%** |
| Llama-65b | LoRA     | 4        | 16          | MP4 recompute|  993.38                   | gradient ckpt, bits 4, max_memory_MB 50000, qlora        | 405.90          |  **122%** |


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
| chatglm-6b    | LoRA     | 1        | 4          |              |        4216.76            |             |       1866.48            | **126%**    |
| chatglm-6b    | Finetune | 4        | 8          |   MP 4       |        3799.78            |   ZeRO 2    |       2124.17            | **79%**    |
| chatglm-6b    | Finetune | 4        | 16         |   MP 4       |        5720.21            |   ZeRO 3    |       3191.35            | **79%**    |


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


### GPT 3

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|---------------|----------|----------|------------|--------------|---------------------------|-------------|--------------------------|---------|
| gpt3-6.7b     | LoRA     | 1        | 4          |              |        3450.06            |             |       1186.74            | **191%**|
| gpt3-13b      | LoRA     | 1        | 4          |              |        2008.40            |             |       969.60             | **107%**|
| gpt3-6.7b     | Finetune | 4        | 8          |   MP 4       |        3301.49            |   ZeRO 2    |       1441.65            | **129%**|
| gpt3-13b      | Finetune | 4        | 8          |   MP 4       |        1890.38            |   ZeRO 2    |       783.26             | **141%**|
| gpt3-6.7b     | Finetune | 4        | 16         |   MP 4       |        4666.19            |   ZeRO 3    |       1756.42            | **166%**|
