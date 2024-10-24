# Pointer Generator Network for Text Summarization

This code is the Paddle v2.0 implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*.
The code adapts and aligns with [a previous Pytorch implmentation](https://github.com/atulkum/pointer_summarizer).

To reach the state-of-the-art performance stated in the source paper, please use the default hyper-parameters listed in *config.py*.

## Model performance (with pointer generation and coverage loss enabled)
After training for 100k iterations with *batch_size=8*, the Paddle implementation achieves a ROUGE-1-f1 of 0.3980 (0.3907 by [a previous Pytorch implmentation](https://github.com/atulkum/pointer_summarizer) and 0.3953 by [the source paper](https://arxiv.org/abs/1704.04368)).

```
ROUGE-1:
rouge_1_f_score: 0.3980 with confidence interval (0.3959, 0.4002)
rouge_1_recall: 0.4639 with confidence interval (0.4613, 0.4667)
rouge_1_precision: 0.3707 with confidence interval (0.3683, 0.3732)

ROUGE-2:
rouge_2_f_score: 0.1726 with confidence interval (0.1704, 0.1749)
rouge_2_recall: 0.2008 with confidence interval (0.1984, 0.2034)
rouge_2_precision: 0.1615 with confidence interval (0.1593, 0.1638)

ROUGE-l:
rouge_l_f_score: 0.3617 with confidence interval (0.3597, 0.3640)
rouge_l_recall: 0.4214 with confidence interval (0.4188, 0.4242)
rouge_l_precision: 0.3371 with confidence interval (0.3348, 0.3396)

```


[detail in](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/examples/text_summarization/pointer_summarizer)
