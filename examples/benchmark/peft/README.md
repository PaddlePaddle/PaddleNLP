# Benchmark Results

### 配置

- 硬件: A100-80G with NVLink, 具体卡数见表
- Torch环境: 见torch/requirements.txt
- 数据: 中文模型使用10k条[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0), 英文模型使用10k条[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- FP16配置: torch 使用 cuda amp fp16, paddle 使用 fp16 O2 opt level

### Bloom

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|---------------|----------|----------|------------|--------------|---------------------------|-------------|--------------------------|---------|
| Bloomz-7b1-mt | LoRA     | 1        | 4          |              | 2293.46                   |             | 1736.92                  | +32%    |
| Bloomz-7b1-mt | Finetune | 4        | 8          | MP 4         | 2873.13                   | ZeRO 3      | 1634.58                  | +76%    |
| Bloomz-7b1-mt | Finetune | 4        | 16         | MP 4         | 2853.83                   | ZeRO 3      | 2694.64                  | +6%     |


### Llama

| Model     | Method   | Num GPUs | Batch Size  | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|-----------|----------|----------|-------------|--------------|---------------------------|-------------|--------------------------|---------|
| Llama-7b  | LoRA     | 1        | 4           |              |  1986.63                  |             | 1589.27                  |  +25%  |
| Llama-13b | LoRA     | 1        | 8           | recompute    |  838.78                   | gradient ckpt |    674.51              |  +24%  |
| Llama-7b  | Finetune | 4        | 8           | MP4          |  2213.46                  | ZeRO 2      | 1331.96                  |  +66%  |
| Llama-7b  | Finetune | 4        | 16          | sharding 2   |  2804.19                  | ZeRO 2      | 2211.25                  |  +27%  |
| Llama-13b | Finetune | 4        | 8           | MP4 recompute|  1651.50                  | ZeRO 3      | 712.53                   | +132%  |


###### 多卡分布式实验记录

- Finetuning with 4 GPUs

| Model     | Setup         | Paddle Effective Tokens /s | Torch Effective Tokens /s  |  Speedup  |
|-----------|---------------|----------------------------|----------------------------|-----------|
| LLaMA-7b  | bsz 8 MP4     | **2213.46**                |  N/A                       | N/A       |
| LLaMA-7b  | bsz 8 ZeRO 3  | 1256.82                    |  1142.07                   | 10%       |
| LLaMA-7b  | bsz 8 ZeRO 2  | 1781.27                    |  1331.96                   | 34%       |
| LLaMA-7b  | bsz 16 (8*2) MP4 | 2630.30                 |  N/A                       | N/A       |
| LLaMA-7b  | bsz 16 ZeRO 3 | 2115.58                    |  2171.99                   | -3%       |
| LLaMA-7b  | bsz 16 ZeRO 2 | **2804.19**                |  2211.25                   | 27%       |
| LLaMA-13b | bsz 8 MP4 recompute |  **1651.50**         |  N/A                       | N/A       |
| LLaMA-13b | bsz 8 ZeRO 3  | 747.17                     |  712.53                    | 5%        |
| LLaMA-13b | bsz 8 ZeRO 2  | OOM                        |  OOM                       | N/A       |

### TODOs

- Add ChatGLM
- Add Llama-30b and Llama-65b
- Enable Flash Attention
