# Paddle Pointer Summarizer

This code is the Paddle v2.0 implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*.
The code adapts and aligns with [a previous Pytorch implmentation](https://github.com/atulkum/pointer_summarizer).

To reach the state-of-the-art performance stated in the source paper, please use the default hyper-parameters listed in *utils/config.py*.  

## Train with pointer generation and coverage loss enabled
After training for 100k iterations with coverage loss enabled (batch size 8), the Paddle implementation achieves a ROUGE-1-f1 of 0.3980 (0.3907 by [a previous Pytorch implmentation](https://github.com/atulkum/pointer_summarizer) and 39.53 by [the source paper](https://arxiv.org/abs/1704.04368)).

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


## How to run training:
1) Follow data generation instruction from https://github.com/abisee/cnn-dailymail; place the folder *finished_files/* as a sister folder of *src/*, *utils/*
2) You might need to change some paths and parameters in *utils/config.py*
3) You need to setup [pyrouge](https://github.com/andersjo/pyrouge) to get the rouge score; also see [this tutorial](https://poojithansl7.wordpress.com/2018/08/04/setting-up-rouge/) to set up rouge and pyrouge.
4)
* To train the model from start:
```
cd src/ && python train.py
```
* To continue training using a previously trained model:
```
cd src/ && python train.py -m path/to/model/dir/
```
* To decode using a previously trained model:
```
cd src/ && python decode.py path/to/model/dir/
```
* If you already have the summaries generated using *src/decode.py* and only needs to run rouge evaluation:
```
cd src/ && python rouge_eval.py path/to/decoded/dir/
```


## Other information:
* The code is tested on Python 3.7.1 and Paddle 2.0.0
* Training takes around 1s/iter on a single Tesla V100 (\~28 hours to train 100k iters)
* Decoding the entire test set takes 2-3 hours

## Papers using this code:
1) TBD
