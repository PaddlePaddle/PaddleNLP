# 文本匹配

**文本匹配一直是自然语言处理（NLP）领域一个基础且重要的方向，一般研究两段文本之间的关系。文本相似度计算、自然语言推理、问答系统、信息检索等，都可以看作针对不同数据和场景的文本匹配应用。这些自然语言处理任务在很大程度上都可以抽象成文本匹配问题，比如信息检索可以归结为搜索词和文档资源的匹配，问答系统可以归结为问题和候选答案的匹配，复述问题可以归结为两个同义句的匹配。**

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/1d24ea95d560465995515f8a3040202b092b07c6d03e4501b64a16dce01a1bbe" hspace='10'/> <br />
</p>


<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/ff58769b237444b89bde5fec9d7215e02825b7d1f2864269986f1daa01b9f497" hspace='10'/> <br />
</p>


文本匹配任务数据每一个样本通常由两个文本组成（query，title）。类别形式为0或1，0表示query与title不匹配； 1表示匹配。


该项目展示了使用传统的[SimNet](./simnet) 和 [SentenceBert](./sentence_bert)两种方法完成本匹配任务。

## SimNet

[SimNet](./simnet) 展示了如何使用CNN、LSTM、GRU等网络完成文本匹配任务。

## Sentence Transformers

[Sentence Transformers](./sentence_transformers) 展示了如何使用以ERNIE为代表的模型Fine-tune完成文本匹配任务。
