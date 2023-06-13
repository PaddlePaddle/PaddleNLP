# Benchmark Results

### Bloom

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle (s/epoch) | Torch Setup | Torch (s/epoch) | Delta |
|---------------|----------|----------|------------|--------------|------------------|-------------|-----------------|-------|
| Bloomz-3b     | Finetune | 1        | 4          | fp16 O2      | 234              | fp16        | 343             | -32%  |
| Bloomz-3b     | LoRA     | 1        | 4          | fp16 O2      | 113              | fp16        | 126             | -10%  |
| Bloomz-7b1-mt | Finetune | 4        | 8          | fp16 O2 MP 4 | 133              | fp16 ZeRO 3 | 313             | -58%  |
| Bloomz-7b1-mt | Finetune | 4        | 16         | fp16 O2 MP 4 | 106              | fp16 ZeRO 3 | 170             | -38%  |
| Bloomz-7b1-mt | Finetune | 4        | 32         | fp16 O2 MP 4 | 85               | fp16 ZeRO 3 | 107             | -21%  |

* transformers默认的half_precision_backend是`torch.cuda.amp`, 不是`apex`. 所以不存在fp16_opt_level参数
