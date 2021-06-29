# Day11 开放域对话作业辅导

本教程旨在辅导同学如何完成 AI Studio课程课后作业——[必修 | 开放域对话系统
](https://aistudio.baidu.com/aistudio/education/objective/25753)。

## 1. 对于多轮闲聊对话，以下哪种技术路线可以产生更连贯的回复？

    A. 检索式系统
    B. 生成式系统

    正确答案：B

    解析：检索式系统，是从大规模语料库中检索出一些比较相关的回复，但较难保持多轮对话的连贯自然

## 2. 对话生成模型常用的训练目标，是最小化以下哪种 loss ？

    A. Negative Log-likelihood (NLL) loss
    B. Bag-of-words (BOW) loss

    正确答案：A

    解析：一般是最小化 NLL loss

## 3. 如果想要模型生成更多样化的回复，应该采用以下哪种解码策略？

    A. Greedy decoding
    B. Sampling-based decoding

    正确答案：B

    解析：Sampling-based decoding 可以得到更多样化/随机的回复

## 4. 使用 top-p sampling 进行生成，如果减少 p 的取值，会产生怎样效果的回复？

    A. 更通用/安全的回复
    B. 更多样化/有风险的回复

    正确答案：A

    解析：减小 p 产生更通用/安全的结果

## 5. 以下哪个自动评估指标，常用来衡量生成的多样性？

    A. Distinct-1/2
    B. BLEU

    正确答案：A

    解析：Distinct-1/2 衡量生成的多样性
