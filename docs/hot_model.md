# 热门模型介绍

## DeepSeek 系列及热门思考模型

PaddleNLP 全面支持 DeepSeek V3/R1/R1-Distill, 及 QwQ-32B 等热门思考模型。

推理方面： DeepSeek V3/R1完整版支持 FP8、INT8、4-bit 量化推理，MTP 投机解码。单机 FP8推理输出超1000 tokens/s; 4-bit 推理输出超2100 tokens/s! 发布新版推理部署镜像，热门模型一键部署。推理部署使用文档全面更新，体验全面提升！

训练方面：凭借数据并行、数据分组切分并行、模型并行、流水线并行以及专家并行等一系列先进的分布式训练能力，结合 Paddle 框架独有的列稀疏注意力掩码表示技术——FlashMask 方法，DeepSeek-R1系列模型在训练过程中显著降低了显存消耗，同时取得了卓越的训练性能提升。

| 模型系列       | 模型名称                                                                                   |
|----------------|--------------------------------------------------------------------------------------------|
| DeepSeek-R1    | deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-R1-Zero, deepseek-ai/DeepSeek-R1-Distill-Llama-70B, deepseek-ai/DeepSeek-R1-Distill-Llama-8B, deepseek-ai/DeepSeek-R1-Distill-Qwen-14B, deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |
| QwQ            | Qwen/QwQ-32B, Qwen/QwQ-32B-Preview                                                         |

## PP-UIE 系列
PP-UIE 自研下一代通用信息抽取模型全新发布。强化零样本学习能力，支持极少甚至零标注数据实现高效冷启动与迁移学习，显著降低数据标注成本；具备处理长文本能力，支持 8192 个 Token 长度文档信息抽取，实现跨段落识别关键信息，形成完整理解；提供完整可定制化的训练和推理全流程，训练效率相较于 LLama-Factory 实现了1.8倍的提升。

| 模型系列       | 模型名称                                                                                   |
|----------------|--------------------------------------------------------------------------------------------|
| PP-UIE         | paddlenlp/PP-UIE-0.5B,paddlenlp/PP-UIE-1.5B, paddlenlp/PP-UIE-7B, paddlenlp/PP-UIE-14B     |
