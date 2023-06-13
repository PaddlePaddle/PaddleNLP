# Benchmark Results

### Bloom

| Model     | Method   | Num GPUs | Setup         | Paddle (s/epoch) | Torch (s/epoch) | Delta |
|-----------|----------|----------|---------------|------------------|-----------------|-------|
| Bloomz-3b | Finetune | 1        | bsz 4 fp16 O1 | 299              | 343*            | -13%  |
| Bloomz-3b | Finetune | 1        | bsz 4 fp16 O2 | 234              | 343*            | -32%  |
| Bloomz-3b | LoRA     | 1        | bsz 4 fp16 O1 | 155              | 126*            | +23%  |
| Bloomz-3b | LoRA     | 1        | bsz 4 fp16 O2 | 113              | 126*            | -10%  |

* transformers默认的half_precision_backend是`torch.cuda.amp`, 不是`apex`. 所以不存在fp16_opt_level参数
