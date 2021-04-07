This code is the Paddle v2.0 implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*.
The code adapts and aligns with [a previous Pytorch implmentation](https://github.com/atulkum/pointer_summarizer).

To reach the state-of-the-art performance stated in the source paper, please use the default hyper-parameters listed in *utils/config.py*.  

## Train with pointer generation and coverage loss enabled 
After training for 100k iterations with coverage loss enabled (batch size 8)

```
ROUGE-1:
rouge_1_f_score: 0.3903 with confidence interval (0.3881, 0.3924)
rouge_1_recall: 0.4313 with confidence interval (0.4287, 0.4340)
rouge_1_precision: 0.3814 with confidence interval (0.3788, 0.3841)

ROUGE-2:
rouge_2_f_score: 0.1674 with confidence interval (0.1652, 0.1696)
rouge_2_recall: 0.1839 with confidence interval (0.1815, 0.1865)
rouge_2_precision: 0.1651 with confidence interval (0.1629, 0.1675)

ROUGE-l:
rouge_l_f_score: 0.3545 with confidence interval (0.3523, 0.3566)
rouge_l_recall: 0.3915 with confidence interval (0.3888, 0.3940)
rouge_l_precision: 0.3467 with confidence interval (0.3442, 0.3493)

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
