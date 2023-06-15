# Benchmark Results

### 配置

- 硬件: A100-80G with NVLink, 具体卡数见表
- Torch环境: 见torch/requirements.txt
- 数据: 10k条[BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M/tree/main)

### Bloom

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle (s/epoch) | Torch Setup | Torch (s/epoch) | Delta |
|---------------|----------|----------|------------|--------------|------------------|-------------|-----------------|-------|
| Bloomz-7b1-mt | LoRA     | 1        | 4          | fp16 O2      | 456              | fp16        | 602             | -24%  |
| Bloomz-7b1-mt | LoRA     | 1        | 8          | fp16 O2      | 493              | fp16        | 596             | -18%  |
| Bloomz-7b1-mt | Finetune | 4        | 8          | fp16 O2 MP 4 | 410              | fp16 ZeRO 3 | 709             | -42%  |
| Bloomz-7b1-mt | Finetune | 4        | 16         | fp16 O2 MP 4 | 363              | fp16 ZeRO 3 | 421             | -14%  |
| Bloomz-7b1-mt | Finetune | 4        | 32         | fp16 O2 MP 4 | OOM              | fp16 ZeRO 3 | 324             | N/A   |

* transformers默认的half_precision_backend是`torch.cuda.amp`, 不使用`fp16_opt_level`参数
