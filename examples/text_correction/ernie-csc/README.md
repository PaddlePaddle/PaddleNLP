# ERNIE for Chinese Spelling Correction

## 简介

中文文本纠错任务是一项NLP基础任务，其输入是一个可能含有语法错误的中文句子，输出是一个正确的中文句子。语法错误类型很多，有多字、少字、错别字等，目前最常见的错误类型是`错别字`。大部分研究工作围绕错别字这一类型进行研究。百度NLP部门在最新的`ACL 2021`上提出以ERNIE为基础，结合中文字语义特征、拼音特征的Softmask策略的中文错别字纠错模型。PaddleNLP将基于该纠错模型提供中文错别字纠错能力。模型结构如下：

![image](https://user-images.githubusercontent.com/10826371/131974040-fc84ec04-566f-4310-9839-862bfb27172e.png)

以下是本项目的简要目录结构及说明：

```text
.
├── README.md                   # 文档
├── download.py                 # 下载Sighan测试集
├── pinyin_vocab.txt            # 拼音字表
├── predict.py                  # 预测标准输入的句子
├── predict_sighan.py           # 生成Sighan测试集的预测结果
├── model.py                    # 纠错模型实现
├── requirements.txt            # 本项目的Python依赖项
├── run_sighan_predict.sh       # 生成模型在Sighan测试集的预测结果并输出预测效果
├── sighan_evaluate.py          # 评估模型在Sighan测试集上预测效果
├── train.py                    # 训练脚本
└── utils.py                    # 通用函数工具
```

## 安装依赖项
```
pip install -r requirements.txt
```

## 模型训练

### 参数
- `model_name_or_path` 指示了Fine-tuning使用的具体预训练模型以及预训练时使用的tokenizer，目前支持的预训练模型有："ernie-1.0"。预训练模型需要与模型类型对应。若模型相关内容保存在本地，这里也可以提供相应目录地址，例如："./checkpoint/model_xx/"。
- `max_seq_length` 表示最大句子长度，超过该长度的部分将被切分成下一个样本。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔步数。
- `save_steps` 表示模型保存及评估间隔步数。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `seed` 表示随机数种子。
- `weight_decay` 表示AdamW的权重衰减系数。
- `warmup_proportion` 表示学习率warmup系数。
- `detection_prob` 表示检测loss的比例。
- `pinyin_vocab_file_path` 拼音字表路径。默认为当前目录下的`pinyin_vocab.txt`文件。

### 单卡训练

```python
python train.py --batch_size 32 --logging_steps 100 --epochs 10 --learning_rate 5e-5 --model_name_or_path ernie-1.0 --output_dir checkpoints5e-5
```

### 多卡训练

```python
python -m paddle.distributed.launch --gpus "0,1"  train.py --batch_size 32 --logging_steps 100 --epochs 10 --learning_rate 5e-5 --model_name_or_path ernie-1.0 --output_dir checkpoints5e-5
```

## 模型预测

### 预测Sighan测试集

Sighan13，Sighan 14，Sighan15是目前中文错别字纠错任务常用的benchmark数据。由于Sighan官方提供的是繁体字数据集，PaddleNLP将提供简体版本的Sighan测试数据。以下运行Sighan预测脚本：

```shell
sh run_sighan_predict.sh
```

该脚本会下载Sighan数据集，加载checkpoint的模型参数运行模型，输出Sighan测试集的预测结果到predict文件，并输出预测效果。

### 预测部署

#### 模型导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见export_model.py。静态图参数保存在`output_path`指定路径中。

运行方式：

```shell
python export_model.py --params_path checkpoints5e-5/best_model.pdparams
```

其中`checkpoints5e-5/best_model.pdparams`是训练过程中保存的参数文件，请更换为实际得到的训练保存路径。

* `params_path`是指动态图训练保存的参数路径
* `output_path`是指静态图参数导出路径。

#### 预测

导出模型之后，可以用于部署，predict.py文件提供了python部署预测示例。运行方式：

```python
python predict.py --model_file infer_model/static_graph_params.pdmodel --params_file infer_model/static_graph_params.pdiparams
```

输出如下图：
![image](https://user-images.githubusercontent.com/10826371/132180831-03b35b03-9eff-4abc-80c3-b43233edfc02.png)


## 参考文献
* Ruiqing Zhang, Chao Pang et al. "Correcting Chinese Spelling Errors with Phonetic Pre-training", ACL, 2021
