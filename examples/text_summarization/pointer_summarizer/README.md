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

## Prerequisites:
* The code is tested on Python 3.7.1 and Paddle 2.0.0
* Training takes around 1s/iter on a single Tesla V100 (\~28 hours to train 100k iters)
* Decoding the entire test set takes 2-3 hours

## Data Preprocessing:
1) Follow data generation instruction from https://github.com/abisee/cnn-dailymail **but place the *make_datafiles_json.py* script provided in this repo into https://github.com/abisee/cnn-dailymail and run *make_datafiles_json.py* instead of *make_datafiles.py* to minimize package dependencies.**
2) place the output folder *finished_files_json/* as a subfolder in this repo
3) You might need to change some paths and parameters in *config.py*


## How to run training:
* To train the model from start:
```
python train.py
```
* To continue training using a previously trained model:
```
python train.py -m path/to/model/dir/
```

## Set up ROUGE
* You need to setup [pyrouge](https://github.com/andersjo/pyrouge) to get the rouge score
* Also see [this tutorial](https://poojithansl7.wordpress.com/2018/08/04/setting-up-rouge/) to set up rouge and pyrouge.


## How to decode & evaluate:
* To decode using a previously trained model:
```
python decode.py path/to/model/dir/
```
* If you already have the summaries generated using *decode.py* and only needs to run rouge evaluation:
```
python rouge_eval.py path/to/decoded/dir/
```
