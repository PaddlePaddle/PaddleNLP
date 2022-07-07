English | [简体中文](./README.md)

**Table of contents**

- [Background introduction](https://github-com.translate.goog/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/ranking/ernie_matching?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#背景介绍)
- ERNIE-Gram
  - [1. Technical solutions and evaluation indicators](https://github-com.translate.goog/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/ranking/ernie_matching?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#技术方案)
  - [2. Environment Dependency](https://github-com.translate.goog/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/ranking/ernie_matching?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#环境依赖)
  - [3. Code structure](https://github-com.translate.goog/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/ranking/ernie_matching?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#代码结构)
  - [4. Data Preparation](https://github-com.translate.goog/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/ranking/ernie_matching?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#数据准备)
  - [5. Model training](https://github-com.translate.goog/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/ranking/ernie_matching?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#模型训练)
  - [6. Evaluation](https://github-com.translate.goog/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/ranking/ernie_matching?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#开始评估)
  - [7. Forecast](https://github-com.translate.goog/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/ranking/ernie_matching?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#预测)
  - [8. Deployment](https://github-com.translate.goog/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/ranking/ernie_matching?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp#部署)



# Background introduction

Pair-wise model is trained based on ERNIE-Gram. The pair-wise matching model is suitable for the application scenario where the text pair similarity is input as one of the features to the upper-level sorting module for sorting.



# ERNIE-Gram



## 1. Technical solutions and evaluation indicators

### Technical solutions

Double tower model, use ERNIE-Gram pre-training model, use margin_ranking_loss to train the model.

### Evaluation Metrics

(1) The AUC index is used to evaluate the ranking effect of the ranking model.

**effect evaluation**

| Model      | AUC   |
| ---------- | ----- |
| ERNIE-Gram | 0.801 |



## 2. Environment dependencies and installation instructions

**environment dependent**

- python >= 3.x
- paddlepaddle >= 2.1.3
- paddlenlp >= 2.2
- pandas >= 0.25.1
- scipy >= 1.3.1



## 3. Code structure

The following is the main code structure and description of this project:

```
ernie_matching/
├── deply # deployment
    ├── cpp
        ├── rpc_client.py # bash script for RPC client
        ├── http_client.py # bash file for http client
        └── start_server.sh # Script to start c++ service
    └── python
        ├── deploy.sh # bash script for predict deployment
        ├── config_nlp.yml # Pipeline configuration file
        ├── web_service.py # Pipeline server script
        ├── rpc_client.py # Script for Pipeline RPC client
        └── predict.py # python prediction deployment example
|—— scripts
    ├── export_model.sh # bash file to transform Dynamic graph parameters to static graph parameters
    ├── export_to_serving.sh # Export bash file in Paddle Serving model format
    ├── train_pairwise.sh # Pair-wise single tower matching model training bash file
    ├── evaluate.sh # bash script of evaluating the performance
    ├── predict_pairwise.sh # Pair-wise single tower matching model prediction script bash file
├── export_model.py # bash file to transform Dynamic graph parameters to static graph parameters
├── export_to_serving.py # Script to export Paddle Serving model format
├── model.py #  Pair-wise matching model networking
├── data.py #  Pair-wise transformation logic for training samples, Pair-wise logic for generating random negative examples
├── train_pairwise.py # Pair-wise single tower matching model training script
├── evaluate.py # evaluation and verification file
├── predict_pairwise.py # Pair-wise single tower matching model prediction script, the output text pair is the similarity
```



## 4. Data Preparation

### Dataset Description

The sample data is as follows:

```
个人所得税税务筹划      基于新个税视角下的个人所得税纳税筹划分析新个税;个人所得税;纳税筹划      个人所得税工资薪金税务筹划研究个人所得税,工资薪金,税务筹划
液压支架底座受力分析    ZY4000/09/19D型液压支架的有限元分析液压支架,有限元分析,两端加载,偏载,扭转       基于ANSYS的液压支架多工况受力分析液压支架,四种工况,仿真分析,ANSYS,应力集中,优化
迟发性血管痉挛  西洛他唑治疗动脉瘤性蛛网膜下腔出血后脑血管痉挛的Meta分析西洛他唑,蛛网膜下腔出血,脑血管痉挛,Meta分析     西洛他唑治疗动脉瘤性蛛网膜下腔出血后脑血管痉挛的Meta分析西洛他唑,蛛网膜下腔出血,脑血管痉挛,Meta分析
氧化亚硅        复合溶胶-凝胶一锅法制备锂离子电池氧化亚硅/碳复合负极材料氧化亚硅,溶胶-凝胶法,纳米颗粒,负极,锂离子电池   负载型聚酰亚胺-二氧化硅-银杂化膜的制备和表征聚酰亚胺,二氧化硅,银,杂化膜,促进传输
```

### Dataset download

- [literature_search_data](https://translate.google.com/website?sl=zh-CN&tl=en&hl=zh-CN&client=webapp&u=https://bj.bcebos.com/v1/paddlenlp/data/literature_search_data.zip)

```
├── milvus # milvus建库数据集
    ├── milvus_data.csv.  # 构建召回库的数据
├── recall  # 召回（语义索引）数据集
    ├── corpus.csv # 用于测试的召回库
    ├── dev.csv  # 召回验证集
    ├── test.csv # 召回测试集
    ├── train.csv  # 召回训练集
    ├── train_unsupervised.csv # 无监督训练集
├── sort # 排序数据集
    ├── test_pairwise.csv   # 排序测试集
    ├── dev_pairwise.csv    # 排序验证集
    └── train_pairwise.csv  # 排序训练集
```



## 5. Model training

**Sorting model download link:**

| Model                                                        | Training parameter configuration     | hardware         | MD5                              |
| ------------------------------------------------------------ | ------------------------------------ | ---------------- | -------------------------------- |
| [ERNIE-Gram-Sort](https://translate.google.com/website?sl=zh-CN&tl=en&hl=zh-CN&client=webapp&u=https://bj.bcebos.com/v1/paddlenlp/models/ernie_gram_sort.zip) | epoch: 3 lr: 5E-5 bs: 64 max_len: 64 | 4 cards v100-16g | d24ece68b7c3626ce6a24baa58dd297d |

### Description of the training environment

- NVIDIA Driver Version: 440.64.00
- Ubuntu 16.04.6 LTS (Docker)
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

### Single-machine single-card training/single-machine multi-card training

Here, the single-machine multi-card mode is used for training. The following commands are used to specify GPU 0, 1, 2, and 3 cards. The training model is based on ERNIE-Gram. The amount of data is relatively large, and it takes about 20 hours and 10 minutes. If you use a single machine and a single card for training, you only need to `--gpu`set the parameter to the card number of the single card.

The training command is as follows:

```
python -u -m paddle.distributed.launch --gpus "0,2,3,4" train_pairwise.py \
        --device gpu \
        --save_dir ./checkpoints \
        --batch_size 32 \
        --learning_rate 2E-5 \
        --margin 0.1 \
        --eval_step 100 \
        --train_file data/train_pairwise.csv \
        --test_file data/dev_pairwise.csv
```

It is also possible to run a bash script:

```
sh scripts/train_pairwise.sh
```



## 6. Evaluation

```
unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0" evaluate.py \
        --device gpu \
        --batch_size 32 \
        --learning_rate 2E-5 \
        --init_from_ckpt "./checkpoints/model_30000/model_state.pdparams" \
        --test_file data/dev_pairwise.csv
```

It is also possible to run a bash script:

```
sh scripts/evaluate.sh
```

After a successful run, the following metrics will be output:

```
eval_dev auc:0.796
```



## 7. Forecast

### Prepare forecast data

The data to be predicted is a tab-delimited tsv file, each row has a text pair, and the semantic index similarity of the text pair, some examples are as follows:

```
中西方语言与文化的差异  第二语言习得的一大障碍就是文化差异。    0.5160342454910278
中西方语言与文化的差异  跨文化视角下中国文化对外传播路径琐谈跨文化,中国文化,传播,翻译   0.5145505666732788
中西方语言与文化的差异  从中西方民族文化心理的差异看英汉翻译语言,文化,民族文化心理,思维方式,翻译        0.5141439437866211
中西方语言与文化的差异  中英文化差异对翻译的影响中英文化,差异,翻译的影响        0.5138794183731079
中西方语言与文化的差异  浅谈文化与语言习得文化,语言,文化与语言的关系,文化与语言习得意识,跨文化交际      0.5131710171699524
```

### start forecasting

Taking the above demo data as an example, run the following command to start calculating the semantic similarity of text Pair based on our open source ERNIE-Gram model:

```
python -u -m paddle.distributed.launch --gpus "0" \
        predict_pairwise.py \
        --device gpu \
        --params_path "./checkpoints/model_30000/model_state.pdparams"\
        --batch_size 128 \
        --max_seq_length 64 \
        --input_file 'sort/test_pairwise.csv'
```

You can also execute the following commands directly:

```
sh scripts/predict_pairwise.sh
```

The following outputs are obtained, which are query, title and corresponding predicted probability:

```
{'query': '中西方语言与文化的差异', 'title': '第二语言习得的一大障碍就是文化差异。', 'pred_prob': 0.85112214}
{'query': '中西方语言与文化的差异', 'title': '跨文化视角下中国文化对外传播路径琐谈跨文化,中国文化,传播,翻译', 'pred_prob': 0.78629625}
{'query': '中西方语言与文化的差异', 'title': '从中西方民族文化心理的差异看英汉翻译语言,文化,民族文化心理,思维方式,翻译', 'pred_prob': 0.91767526}
{'query': '中西方语言与文化的差异', 'title': '中英文化差异对翻译的影响中英文化,差异,翻译的影响', 'pred_prob': 0.8601749}
{'query': '中西方语言与文化的差异', 'title': '浅谈文化与语言习得文化,语言,文化与语言的关系,文化与语言习得意识,跨文化交际', 'pred_prob': 0.8944413}
```



## 8. Deployment

### Dynamic to static export

First convert the dynamic graph model to a static graph:

```
python export_model.py --params_path checkpoints/model_30000/model_state.pdparams --output_path=./output
```

You can also run the following bash script:

```
sh scripts/export_model.sh
```

### Paddle Inference

Modify the prediction file path:

```
input_file='../../sort/test_pairwise.csv'
```

Then use PaddleInference

```
python predict.py --model_dir=../../output
```

You can also run the following bash script:

```
sh deploy.sh
```

The following output is obtained, which outputs the query, title and corresponding probability of the sample:

```
Data: {'query': '中西方语言与文化的差异', 'title': '第二语言习得的一大障碍就是文化差异。'}       prob: [0.8511221]
Data: {'query': '中西方语言与文化的差异', 'title': '跨文化视角下中国文化对外传播路径琐谈跨文化,中国文化,传播,翻译'}      prob: [0.7862964]
Data: {'query': '中西方语言与文化的差异', 'title': '从中西方民族文化心理的差异看英汉翻译语言,文化,民族文化心理,思维方式,翻译'}   prob: [0.91767514]
Data: {'query': '中西方语言与文化的差异', 'title': '中英文化差异对翻译的影响中英文化,差异,翻译的影响'}   prob: [0.8601747]
Data: {'query': '中西方语言与文化的差异', 'title': '浅谈文化与语言习得文化,语言,文化与语言的关系,文化与语言习得意识,跨文化交际'}     prob: [0.8944413]
```

### Paddle Serving Deployment

For detailed documentation of Paddle Serving, please refer to [Pipeline_Design](https://github-com.translate.goog/PaddlePaddle/Serving/blob/v0.7.0/doc/Python_Pipeline/Pipeline_Design_CN.md?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp) and [Serving_Design](https://github-com.translate.goog/PaddlePaddle/Serving/blob/v0.7.0/doc/Serving_Design_CN.md?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp) . First, convert the static graph model into Serving format:

```
python export_to_serving.py \
    --dirname "output" \
    --model_filename "inference.predict.pdmodel" \
    --params_filename "inference.predict.pdiparams" \
    --server_path "serving_server" \
    --client_path "serving_client" \
    --fetch_alias_names "predict"
```

Parameter meaning description

- `dirname`: The storage path of the model file to be converted. The program structure file and parameter file are saved in this directory.
- `model_filename`: The name of the file where the Inference Program structure of the model that needs to be converted is stored. If set to None, use `__model__`as default filename
- `params_filename`: The name of the file where all parameters of the model to be converted are stored. It needs to be specified if and only if all model parameters are stored in a single binary file. If the model parameters are stored in separate files, set it to None
- `server_path`: The storage path of the converted model files and configuration files. Default is serving_server
- `client_path`: The converted client configuration file storage path. Default is serving_client
- `fetch_alias_names`: Alias settings for model output, such as input input_ids, etc., can be specified as other names, which are not specified by default
- `feed_alias_names`: Alias settings for model input, such as output pooled_out, etc., can be re-specified to other models, and are not specified by default

You can also run the following bash script:

```
sh scripts/export_to_serving.sh
```

There are two ways to deploy Paddle Serving. The first way is the Pipeline way, and the second way is the C++ way. The following describes the usage of these two ways:

#### Pipeline method

Start Pipeline Server:

```
python web_service.py
```

Start the client to call Server.

First modify the samples that need to be predicted in rpc_client.py:

```
list_data = [{"query":"中西方语言与文化的差异","title":"第二语言习得的一大障碍就是文化差异。"}]`
```

Then run:

```
python rpc_client.py
```

The output of the model is:

```
PipelineClient::predict pack_data time:1656912047.5986433
PipelineClient::predict before time:1656912047.599081
time to cost :0.012039899826049805 seconds
(1, 1)
[[0.85112208]]
```

It can be seen that the client sent 1 text, and the probability value of this text is similar.

#### C++ way

Start Serving of C++:

```
python -m paddle_serving_server.serve --model serving_server --port 8600 --gpu_id 0 --thread 5 --ir_optim True
```

A script can also be used:

```
sh deploy/cpp/start_server.sh
```

Client can use http or rpc two ways, the rpc way is:

```
python deploy/cpp/rpc_client.py
```

The output of the run is:

```
I0704 05:19:00.443437  1987 general_model.cpp:490] [client]logid=0,client_cost=8.477ms,server_cost=6.458ms.
time to cost :0.008707761764526367 seconds
{'predict': array([[0.8511221]], dtype=float32)}
```

You can see that the server returns the similarity result

Or use http client access mode:

```
python deploy/cpp/http_client.py
```

The output of the run is:

```
time to cost :0.006819009780883789 seconds
[0.8511220812797546]
```

You can see that the server returns the similarity result

You can also use curl to send Http requests:

```
curl -XPOST http://0.0.0.0:8600/GeneralModelService/inference -d  ' {"tensor":[{"int64_data":[    1,    12,   213,    58,   405,   545,    54,    68,    73,
            5,   859,   712,     2,   131,   177,   405,   545,   489,
          116,     5,     7,    19,   843,  1767,   113,    10,    68,
           73,   859,   712, 12043,     2],"elem_type":0,"name":"input_ids","alias_name":"input_ids","shape":[1,32]},
    {"int64_data":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1],"elem_type":0,"name":"token_type_ids","alias_name":"token_type_ids","shape":[1,32]}
        ],
"fetch_var_names":["sigmoid_2.tmp_0"],
"log_id":0
}'
```

## Reference

[1] Xiao, Dongling, Yu-Kun Li, Han Zhang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. “ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding.” ArXiv:2010.12148 [Cs].