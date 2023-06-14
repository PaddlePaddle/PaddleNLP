# Benchmark Results

### 硬件与环境

- 硬件: A100-80G with NVLink, 具体卡数见表
- Torch环境：见torch/requirements.txt

### Bloom

| Model         | Method   | Num GPUs | Batch Size | Paddle Setup | Paddle (s/epoch) | Torch Setup | Torch (s/epoch) | Delta |
|---------------|----------|----------|------------|--------------|------------------|-------------|-----------------|-------|
| Bloomz-7b1-mt | LoRA     | 1        | 4          | fp16 O2      | 179              | fp16        | 219             | -18%  |
| Bloomz-7b1-mt | LoRA     | 1        | 8          | FP16 O2      | 171              | fp16        | 197             | -13%  |
| Bloomz-7b1-mt | Finetune | 4        | 8          | fp16 O2 MP 4 | 133              | fp16 ZeRO 3 | 288             | -54%  |
| Bloomz-7b1-mt | Finetune | 4        | 16         | fp16 O2 MP 4 | 106              | fp16 ZeRO 3 | 150             | -29%  |
| Bloomz-7b1-mt | Finetune | 4        | 32         | fp16 O2 MP 4 | 85               | fp16 ZeRO 3 | 94              | -10%  |

* transformers默认的half_precision_backend是`torch.cuda.amp`, 不使用`fp16_opt_level`参数
