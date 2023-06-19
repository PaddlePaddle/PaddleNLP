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
| Llama-7b  | Finetune | 4        | 8           | fp16 O2 MP 4 |   2208.06                | fp16 ZeRO 3 |    1142.07          |  +93%   |
| Llama-7b  | Finetune | 4        | 16          | fp16 O2, MP 4 <br> batch size 8 <br> grad accumulation 2 |   2559.84   | fp16 ZeRO 3 |     2171.99     |  +18%   |
| Llama-13b | LoRA     | 1        | 8           | fp16 O2, recompute |  838.78             | fp16, gradient ckpt    |    674.51 | +24%   |
| Llama-13b | Finetune | 4        | 8           | fp16 O2, MP 4 <br> batch size 4 <br> grad accumulation 2 | 1165.84  | fp16 ZeRO 3 |    706.70          | +65% |

### TODOs

- Add ChatGLM
- Add Llama-30b and Llama-65b
- Enable Flash Attention
- Test sharding stage 3