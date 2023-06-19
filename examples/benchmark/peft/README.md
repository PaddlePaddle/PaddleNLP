# Benchmark Results

### 配置

- 硬件: A100-80G with NVLink, 具体卡数见表
- Torch环境: 见torch/requirements.txt
- 数据: 10k条[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)

### Bloom

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle Effective Tokens/s | Torch Setup | Torch Effective Tokens/s | Speedup |
|---------------|----------|----------|------------|--------------|---------------------------|-------------|--------------------------|---------|
| Bloomz-7b1-mt | LoRA     | 1        | 4          | fp16 O2      | 2293.46                   | fp16        | 1736.92                  | +32%    |
| Bloomz-7b1-mt | Finetune | 4        | 8          | fp16 O2 MP 4 | 2873.13                   | fp16 ZeRO 3 | 1634.58                  | +76%    |
| Bloomz-7b1-mt | Finetune | 4        | 16         | fp16 O2 MP 4 | 2853.83                   | fp16 ZeRO 3 | 2694.64                  | +6%     |
