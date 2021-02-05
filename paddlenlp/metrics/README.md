# paddlenlp.metrics

目前paddlenlp提供以下评价指标：

| Metric                                                   | 简介                                                         | API                                                          |
| -------------------------------------------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| Perplexity                                               | 困惑度，常用来衡量语言模型优劣，也可用于机器翻译、文本生成等任务。 | `paddlenlp.metrics.Perplexity`                               |
| BLEU(bilingual evaluation understudy)                    | 机器翻译常用评价指标                                         | `paddlenlp.metrics.BLEU`                                     |
| Rouge(Recall-Oriented Understudy for Gisting Evaluation) | 评估自动文摘以及机器翻译的指标                               | `paddlenlp.metrics.RougeL`, `paddlenlp.metrics.RougeN`       |
| AccuracyAndF1                                            | 准确率及F1-score，可用于GLUE中的MRPC 和QQP任务               | `paddlenlp.metrics.AccuracyAndF1`                            |
| PearsonAndSpearman                                       | 皮尔森相关性系数和斯皮尔曼相关系数。可用于GLUE中的STS-B任务  | `paddlenlp.metrics.PearsonAndSpearman`                       |
| Mcc(Matthews correlation coefficient)                    | 马修斯相关系数，用以测量二分类的分类性能的指标。可用于GLUE中的CoLA任务 | `paddlenlp.metrics.Mcc`                                      |
| ChunkEvaluator                                           | 计算了块检测的精确率、召回率和F1-score。常用于序列标记任务，如命名实体识别（NER） | `paddlenlp.metrics.ChunkEvaluator`                           |
| Squad                                                    | 用于SQuAD和DuReader-robust的评价指标                         | `paddlenlp.metrics.compute_predictions`, `paddlenlp.metrics.squad_evaluate` |
