# DuReader<sub>retrieval</sub> Dataset
**Passage retrieval** requires systems to find relevant passages from a large passage collection, which is an important task in the fields of natural language processing and information retrieval. The traditional retrieval systems use term-based sparse representations (e.g. BM25) to match the query and the candidate passages. But such methods cannot handle term mismatch, i.e. semantically relevant but with few overlapped terms. Recent studies have shown that dense retrieval based on pre-trained language models can effectively deal with such problem via semantic dense representations of query and passages. The method has better performance in many applications, like question answering. 

To promote the research in dense retrieval, we present **DuReader<sub>retrieval</sub>**, a large-scale Chinese dataset for passage retrieval. The dataset contains over **\*90K\*** questions and **\*8M\*** passages from real users, and covers many challenges in real-world applications. For more details about the dataset, please refer to this [paper](https://arxiv.org/abs/2203.10232).

We provide two types of dataset:

- **Orginal dataset** in json format, containing queries, their corresponding positive passages, and passage collection.
- **Pre-processed dataset** in tsv format, used for the baseline system, containing extra hard negative samples that selected using our retrieval model. 

# DuReader<sub>retrieval</sub> Baseline System
In this repository, we release a baseline system for DuReader<sub>retrieval</sub> dataset. The baseline system is based on [RocketQA](https://arxiv.org/pdf/2010.08191.pdf) and [ERNIE 1.0](https://arxiv.org/abs/1904.09223), and is implemented with [PaddlePaddle](https://www.paddlepaddle.org.cn/) framework. To run the baseline system, please follow the instructions below.

## Environment Requirements
The baseline system has been tested on

 - CentOS 6.3
 - PaddlePaddle 2.2 
 - Python 3.7.0
 - Faiss 1.7.1
 - Cuda 10.1
 - CuDnn 7.6
 
To install PaddlePaddle, please see the [PaddlePaddle Homepage](http://paddlepaddle.org/) for more information.


## Download
Before run the baseline system, please download the pre-processed dataset and the pretrained and fine-tuned model parameters (ERNIE 1.0 base):

```
sh script/download.sh
```
The dataset will be saved into `dureader-retrieval-baseline-dataset/`, the pretrained and fine-tuned model parameters will be saved into `pretrained-models/` and `finetuned-models/`, respectively.  The descriptions of the data structure can be found in `dureader-retrieval-baseline-dataset/readme.md`. 

**Note**: in addition to the positive samples from the origianl dataset, we also provide hard negative samples that produced by our dense retrieval model in the training data. Users may use their own strategies for hard negative sample selection. 


## Run Baseline
The baseline system contatins two steps: 

- Step 1: a dual-encoder for passage retrieval; 
- Step 2: a cross-encoder for passage re-ranking.

For more details about the model structure, please refer to [RocketQA](https://arxiv.org/pdf/2010.08191.pdf) (Qu et al., 2021). 

### Step 1 - Dual-encoder (for retrieval)
#### Training
To fine-tune a retrieval model, please run the following command:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
TRAIN_SET="dureader-retrieval-baseline-dataset/train/dual.train.demo.tsv"
MODEL_PATH="pretrained-models/ernie_base_1.0_twin_CN/params"
sh script/run_dual_encoder_train.sh $TRAIN_SET $MODEL_PATH 10 4 
```
This will train on the demo data for 10 epochs with 4 gpu cars. The training log will be saved into `log/`. At the end of training, model parameters will be saved into `output/`. To start the training on the full dataset, please set `TRAIN_SET=dureader-retrieval-baseline-dataset/train/dual.train.tsv`.

**Note**: We strongly recommend to use more gpus for training. The performance increases with the effective batch size, which is related to the number of gpus. For single-gpu training, please turn off the option `use_cross_batch` in `script/run_dual_encoder_train.sh`. 


#### Prediction
To predict with fine-tuned parameters, (e.g. on the devlopment set), please run the following command:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
TEST_SET="dureader-retrieval-baseline-dataset/dev/dev.q.format"
MODEL_PATH="finetuned-models/dual_params/"
DATA_PATH="dureader-retrieval-baseline-dataset/passage-collection"
TOP_K=50
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K
```
The fine-tuned parameters under `MODEL_PATH ` will be loaded for prediction. The prediction on the development set will take a few hours on 4*V100 cards. The predicted results will be saved into `output/`. 

We provide a script to convert the model output to the standard json format for evaluation. To preform the conversion:

```
QUERY2ID="dureader-retrieval-baseline-dataset/dev/q2qid.dev.json"
PARA2ID="dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json"
MODEL_OUTPUT="output/dev.res.top50"
python metric/convert_recall_res_to_json.py $QUERY2ID $PARA2ID $MODEL_OUTPUT
```
Where `MODEL_OUTPUT` represents the output file from the dual-encoder, `QUERY2ID `, `PARA2ID ` are the mapping files which maps the query and passages to their original IDs. The output json file will be saved in `output/dual_res.json`.


**Note**: We divide the passage collection into 4 parts for data parallel. For users who use different number of GPUs, please update the data files (i.e. `dureader-retrieval-baseline-dataset/passage-collection/part-0x`) and the corresponding configurations.

### Step 2 - Cross-encoder (for re-ranking)
#### Training
To fine-tune a re-ranking model, please run the following command:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
TRAIN_SET=dureader-retrieval-baseline-dataset/train/cross.train.demo.tsv
MODEL_PATH=pretrained-models/ernie_base_1.0_CN/params
sh script/run_cross_encoder_train.sh $TRAIN_SET $MODEL_PATH 3 4
```
This will train on the demo data for 3 epochs with 4 gpu cars (a few minutes on 4*V100). The training log will be saved into `log/`. The model parameters will be saved into `output/`. To start the training on the full dataset, please set `TRAIN_SET=dureader-retrieval-baseline-dataset/train/cross.train.tsv`


#### Prediction
To predict with fine-tuned parameters, (e.g. on the devlopment set), please run the following command:

```
export CUDA_VISIBLE_DEVICES=0
TEST_SET=dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv
MODEL_PATH=finetuned-models/cross_params/ 
sh script/run_cross_encoder_inference.sh $TEST_SET $MODEL_PATH
```
Where `TEST_SET` is the top-50 retrieved passages for each query from step 1, `MODEL_PATH` is the path to fined-tuned model parameters. The predicted answers will be saved into `output/`. 

We provide a script to convert the model output to the standard json format for evaluation. To preform the conversion:

```
MODEL_OUTPUT="output/dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv.score.0.0"
ID_MAP="dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.id_map.tsv"
python metric/convert_rerank_res_to_json.py $MODEL_OUTPUT $ID_MAP 
```
Where `MODEL_OUTPUT` represents the output file from the cross-encoder, `ID_MAP` is the mapping file which maps the query and passages to their original IDs. The output json file will be saved in `output/cross_res.json`.

## Evaluation
`MRR@10`, `Recall@1` and `Recall@50` are used as evaluation metrics. Here we provide a script `evaluation.py` for evaluation.

To evluate, run

```
REFERENCE_FIEL="dureader-retrieval-baseline-dataset/dev/dev.json"
PREDICTION_FILE="output/cross_res.json"
python metric/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE
```
Where `REFERENCE_FIEL` is the origianl dataset file, and `PREDICTION_FILE ` is the model prediction that should be a valid JSON file of `(qid, [list-of -top50-pid])` pairs, for example:

```
{
   "edb58f525bd14724d6f490722fa8a657":[
      "5bc347ff17d488f1704e2893c9e8ecfa",
      "6e67389d07da8ce02ed97167d23baf9d",
      "06031941b9613d2fde5cb309bbefaf88",
      ...
      "58c00697311c9ad6eb384b6fca7bd12d",
      "e06eb2750f5ed163eb85b6ef02b7c608",
      "aa425d7dcb409592527a22ce5eccd4d5"
   ],
   "a451acd1e9836b04b16664e9f0c290e5":[
      "71c8004cc562f2a75181b2d3d370a45a",
      "ef3d34ea63b3de9db612bd7b7ffd143a",
      "2c839510f35d5495251c6b3c057bd300",
      ...
      "6b9777840bb537a433add0b9f553fd42",
      "4203161e38b9b5e67ff16fc777f614be",
      "ae651b80efbb10f786380a6afdc1dcbe",
   ]
}
```

After runing the evaluation script, you will get the evaluation results with the following format:

```
{"MRR@10": 0.7284081349206347, "QueriesRanked": 2000, "recall@1": 0.641, "recall@50": 0.9175}
```

## Baseline Performance
The performance of our baseline model on the development set are shown below:

| Model |  MRR@10 | recall@1 | recall@50 |
| --- | --- | --- | --- |
| dual-encoder (retrieval) | 60.45 | 49.75 | 91.75|
| cross-encoder (re-ranking) | 72.84 | 64.10 | 91.75|

# Copyright and License
This repository is provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/RocketQA/blob/main/LICENSE).
