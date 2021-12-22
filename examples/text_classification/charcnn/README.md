# CharCNN
A PaddlePaddle implementation of CharCNN. 

> Merged from [Paddorch](https://github.com/paddorch/CharCNN.paddle).


## 1. Introduction

![](https://user-images.githubusercontent.com/27690278/145707824-82aff372-5ee8-4724-80e1-3b3b9e16776c.png)

Paper: [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626v3.pdf)

## 2. Results

|  Datasets          | Paper error rate <br> (large / small)| Our error rate <br> (large / small) | abs. improv. <br> (large / small) | epochs |
|--------------------|-----------------|-----------------|--------------|:---:|
| AG’s News          | 13.39 / 14.80   | 9.38 / 10.17    | 4.01 / 4.63  | 60  |
| Yahoo! Answers     | 28.80 / 29.84   | 27.73 / 28.69   | 1.07 / 1.15  | 15  |
| Amazon Review Full | 40.45 / 40.43   | 38.22 / 38.97   | 2.23 / 1.46  | 7   |

> Note: the `large` model has not yet converged, and the accuracy can be improved by continuing training.

## 3. Dataset

![](https://user-images.githubusercontent.com/27690278/145707817-4a20d611-bb84-4eae-9c33-03f2a05a4ed0.png)

Format:
```
"class idx","sentence or text to be classified"  
```

Samples are separated by newline.

Example:
```shell
"3","Fears for T N pension after talks, Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
"4","The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com)","SPACE.com - TORONTO, Canada -- A second\team of rocketeers competing for the  #36;10 million Ansari X Prize, a contest for\privately funded suborbital space flight, has officially announced the first\launch date for its manned rocket."
```

## 4. Requirement

- PaddlePaddle >= 2.0.0
- see `requirements.txt`

## 5. Usage

### Train
1. download [AG News dataset](https://github.com/paddorch/CharCNN.paddle/tree/main/data/ag_news_csv) to folder `./data/ag_news`，and then split the training set into `train` and `dev` part:

```shell
bash ./scripts/split_data.sh data/ag_news/train.csv
```

2. start train
```shell
bash ./scripts/train_ag_news.sh
```

### Test
```shell
bash ./scripts/eval_ag_news.sh
```

## 6. Implementation Details
### Data Augumentation
We use [nlpaug](https://github.com/makcedward/nlpaug) to augment data, specifically, we substitute similar word according to `WordNet`.

there's two implementation: `SynonymAug` and [`GeometricSynonymAug`](https://github.com/paddorch/CharCNN.paddle/blob/main/utils/augmenter.py#L6), `GeometricSynonymAug` is our adapted version of `SynonymAug`, which leverages geometric distribution in substitution as described in the CharCNN paper.

Augumentation demos:
```
==================== GeometricSynonymAug
The straightaway brown dodger rise complete the lazy domestic dog
The quick john brown fox jumps over the lazy dog
The quick brown slyboots jumps over the lazy dog
The straightaway brownness fox start all over the lazy canis familiaris
The quick brown fox jumps over the indolent canis familiaris
The straightaway brown charles james fox jumps terminated the lazy domestic dog
The quick brown george fox jumps over the lazy domestic dog
The quick brown fox jumps over the indolent dog
The immediate brownness fox jumps ended the slothful dog
The quick brown fox jumps over the lazy canis familiaris
--- 2.56608247756958 seconds ---

==================== SynonymAug
The quick brown fox leap over the lazy frank
The ready brown charles james fox jumps over the lazy dog
The quick brown fox jump over the lazy frank
The speedy brown university fox jumps over the lazy dog
The ready brown fox jump off over the lazy dog
The quick robert brown fox jump over the lazy dog
The quick brown fox jumps concluded the lazy hound
The quick brown university fox jumps over the lazy click
The quick brown fox jumps over the slothful andiron
The quick brown fox parachute over the lazy domestic dog
--- 0.011068582534790039 seconds ---
```

We experimented GeometricSynonymAug on `AG’s News` with `small` model, the accuracy dropped by about `0.4` (error rate: 10.59).

## References
```bibtex
@article{zhang2015character,
  title={Character-level convolutional networks for text classification},
  author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  journal={Advances in neural information processing systems},
  volume={28},
  pages={649--657},
  year={2015}
}
```

- https://github.com/srviest/char-cnn-text-classification-pytorch