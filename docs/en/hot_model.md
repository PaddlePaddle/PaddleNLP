# Introduction to Popular Models

## DeepSeek Series and Leading Reasoning Models

PaddleNLP fully supports DeepSeek V3/R1/R1-Distill, QwQ-32B and other leading reasoning models.

**Inference**: The complete DeepSeek V3/R1 supports FP8, INT8, 4-bit quantized inference, and MTP speculative decoding. Achieves over 1000 tokens/s output with single-machine FP8 inference, and exceeds 2100 tokens/s with 4-bit quantized inference! Released new inference deployment images for one-click deployment of popular models. The inference deployment documentation has been fully updated with comprehensive experience improvements!

**Training**: Leveraging data parallelism, data group sharding parallelism, model parallelism, pipeline parallelism, and expert parallelism, combined with Paddle framework's unique column-sparse attention masking technique - FlashMask method, the DeepSeek-R1 series achieves significant memory reduction during training while demonstrating outstanding training performance improvements.

| Model Series | Model Names |
|--------------|-------------|
| DeepSeek-R1  | deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-R1-Zero, deepseek-ai/DeepSeek-R1-Distill-Llama-70B, deepseek-ai/DeepSeek-R1-Distill-Llama-8B, deepseek-ai/DeepSeek-R1-Distill-Qwen-14B, deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |
| QwQ          | Qwen/QwQ-32B, Qwen/QwQ-32B-Preview |

## PP-UIE Series
PP-UIE, the next-generation general information extraction model, is officially released. Enhanced with:
- Strong zero-shot learning capabilities, supporting efficient cold-start and transfer learning with minimal or even no labeled data
- Long-text processing capability supporting 8192-token documents
- Complete customizable training and inference pipeline
- 1.8x training efficiency improvement compared to LLama-Factory

| Model Series | Model Names |
|--------------|-------------|
| PP-UIE       | paddlenlp/PP-UIE-0.5B, paddlenlp/PP-UIE-1.5B, paddlenlp/PP-UIE-7B, paddlenlp/PP-UIE-14B |
