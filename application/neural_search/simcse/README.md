# SimCSE

SimCSE 模型适合缺乏监督数据，但是又有大量无监督数据的匹配和检索场景。


## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```
simcse/
├── model.py # SimCSE 模型组网代码
├── data.py # 无监督语义匹配训练数据、测试数据的读取逻辑
├── predict.py # 基于训练好的无监督语义匹配模型计算文本 Pair 相似度
└── train.py # SimCSE 模型训练、评估逻辑
```


### 模型训练
训练数据使用的是万方的数据集:

无监督预训练的数据的数据格式为：

```
煤矸石-污泥基活性炭介导强化污水厌氧消化
睡眠障碍与常见神经系统疾病的关系睡眠觉醒障碍,神经系统疾病,睡眠,快速眼运动,细胞增殖,阿尔茨海默病
......
```


训练的命令如下：

```shell
$ unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus '0,1,2,3' \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 3 \
	--save_steps 2000 \
	--eval_steps 100 \
	--max_seq_length 64 \
	--infer_with_fc_pooler \
	--dropout 0.2 \
    --output_emb_size 256 \
	--train_set_file "./data/train_unsupervised.csv" \
	--test_set_file "./data/test.csv" 
```
也可以使用bash脚本：

```
sh train.sh
```

可支持配置的参数：

* `infer_with_fc_pooler`：可选，在预测阶段计算文本 embedding 表示的时候网络前向是否会过训练阶段最后一层的 fc;  建议打开模型效果最好。
* `scale`：可选，在计算 cross_entropy loss 之前对 cosine 相似度进行缩放的因子；默认为 20。
* `dropout`：可选，SimCSE 网络前向使用的 dropout 取值；默认 0.1。
* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `max_seq_length`：可选，ERNIE-Gram 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `epochs`: 训练轮次，默认为1。
* `warmup_proption`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── model_100
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:**
* 如需恢复模型训练，则可以设置`init_from_ckpt`， 如`init_from_ckpt=checkpoints/model_100/model_state.pdparams`。

### 构建索引

```
root_dir="checkpoints" 

python -u -m paddle.distributed.launch --gpus "3" --log_dir "recall_log/" \
        recall.py \
        --device gpu \
        --recall_result_dir "recall_result_dir" \
        --recall_result_file "recall_result.txt" \
        --params_path "${root_dir}/model_10000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 64 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "data/test.csv" \
        --corpus_file "data/corpus.csv" 


```
也可以使用bash脚本：

```
sh run_build_index.sh
```

### 召回评估

```
python -u evaluate.py \
    --similar_text_pair "data/test.csv" \
    --recall_result_file "./recall_result_dir/recall_result.txt" \
    --recall_num 50
```
可以使用bash脚本：

```
sh evaluate.sh
```

### 基于动态图模型预测
 
测试数据示例如下，：
```text
煤矸石-污泥基活性炭介导强化污水厌氧消化 煤矸石-污泥基活性炭介导强化污水厌氧消化煤矸石,污泥,复合基活性炭,厌氧消化,直接种间电子传递
. 睡眠障碍与常见神经系统疾病的关系      睡眠障碍与常见神经系统疾病的关系睡眠觉醒障碍,神经系统疾病,睡眠,快速眼运动,细胞增殖,阿尔茨海默病
城市道路交通流中观仿真研究      城市道路交通流中观仿真研究智能运输系统;城市交通管理;计算机仿真;城市道路;交通流;路径选择
网络健康可信性研究      网络健康可信性研究网络健康信息;可信性;评估模式
脑瘫患儿家庭复原力的影响因素及干预模式雏形 研究 脑瘫患儿家庭复原力的影响因素及干预模式雏形研究脑瘫患儿;家庭功能;干预模式
地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较      地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较
```

执行如下命令开始预测：
```shell
python -u -m paddle.distributed.launch --gpus "0" \
        predict.py \
        --device gpu \
        --params_path "./checkpoints/model_4400/model_state.pdparams"\
        --batch_size 64 \
        --max_seq_length 64 \
        --text_pair_file 'test.tsv'
```




## Reference
[1] Gao, Tianyu, Xingcheng Yao, and Danqi Chen. “SimCSE: Simple Contrastive Learning of Sentence Embeddings.” ArXiv:2104.08821 [Cs], April 18, 2021. http://arxiv.org/abs/2104.08821.
