# 投机解码教程

投机解码是一个通过投机性地一次性猜测多个 token 然后进行验证和接收的算法，通过投机解码可以极大地减小推理时延。PaddleNLP 提供了简单、高效的投机解码推理流程。下面提供 PaddleNLP 中各种投机解码算法的使用说明。

## Inference with reference

该算法通过 n-gram 窗口从 prompt 中匹配 draft tokens，适合输入和输出有很大 overlap 的场景如代码编辑、文档查询等，更多信息查看查看[论文地址](https://arxiv.org/pdf/2304.04487)。

### 使用命令

```shell
# 动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --speculate_method inference_with_reference --speculate_max_draft_token_num 5 --speculate_max_ngram_size 2
```

**Note:**

1. 该算法目前只支持 llama 系列模型
2. 投机解码同时支持量化推理，具体命令参考[推理示例](./inference.md)，将 speculate_method 等投机解码参数加上即可。

## TopP + Window verify

在投机解码中一个影响加速比重要的点是 verify 后的接受率，区别于简单的贪心搜索，TopP 和 TopK 采样，我们采取 TopP + Window verify 的方法来提高接受率(默认开启，若要切换到 Top1 verify 请指定环境变量 export SPECULATE_VERIFY_USE_TOPK=1)，下面详细介绍 TopP + Window verify 策略的原理。

在推理 draft tokens 得到 verify tokens 的 logits 后，我们先通过 TopP 采样得到 verify tokens，如果 TopP 个 verify tokens 的数目不足 speculate_max_candidate_len 个时 padding 到 speculate_max_candidate_len 个 verify tokens，然后对于每一个 draft token 判断是否位于 verify tokens 中的 TopP 个 token 中，是则 TopP verify 接收此 draft token，否则判断后面两个 draft token 的 Top1 verify 是否接收，只有当后面两个 draft token 都被 Top1 verify 接收时，才同时接收这三个 draft tokens 否则都不接收(window verify)。