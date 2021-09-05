# 中文文本纠错

## 简介

中文文本纠错任务是一项NLP基础任务，其输入是一个可能含有语法错误的中文句子，输出是一个正确的中文句子。语法错误类型很多，有多字、少字、错别字等，目前最常见的错误类型是`错别字`。大部分研究工作围绕错别字这一类型进行研究。百度NLP部门在最新的`ACL 2021`上提出结合拼音特征的Softmask策略的中文错别字纠错模型。PaddleNLP将基于该纠错模型提供中文错别字纠错能力。模型结构如下：

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

## 快速开始

### 安装依赖项
```
pip install -r requirements.txt
```

### 模型训练

#### 参数
- `model_type` 指示模型使用的预训练模型类型。目前支持的类型有："ernie", "ernie_gram","roberta"。
- `model_name_or_path` 指示了Fine-tuning使用的具体预训练模型以及预训练时使用的tokenizer，目前支持的预训练模型有："ernie-1.0", "ernie-gram-zh", "roberta-wwm-ext"。预训练模型需要与模型类型对应。若模型相关内容保存在本地，这里也可以提供相应目录地址，例如："./checkpoint/model_xx/"。
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

#### 单卡训练

```python
python train.py --batch_size 32 --logging_steps 100 --epochs 10 --learning_rate 5e-5 --model_type ernie --model_name_or_path ernie-1.0 --output_dir ernie_log/checkpoints5e-5
```

#### 多卡训练

```python
python -m paddle.distributed.launch --gpus "0,1"  train.py --batch_size 32 --logging_steps 100 --epochs 10 --learning_rate 5e-5 --model_type ernie --model_name_or_path ernie-1.0 --output_dir ernie_log/checkpoints5e-5
```

### 模型预测

#### 预测Sighan测试集

 Sighan13，Sighan 14，Sighan15是目前中文错别字纠错任务常用的benchmark数据。由于Sighan官方提供的是繁体字数据集，PaddleNLP将提供简体版本的Sighan测试数据。

 1. 下载数据

    ```python
    python download.py
    ```

 2. 运行Sighan预测脚本
    该脚本会加载checkpoint的模型参数运行模型，输出Sighan测试集的预测结果到predict文件，并输出预测效果。
    ```shell
    sh run_sighan_predict.sh
    ```

    运行预测脚本后应得到如下的输出：
    ![image](https://user-images.githubusercontent.com/10826371/131977272-35cfe428-77ad-4db3-b09f-839a4784823c.png)

#### 预测
可运行`predict.py`根据提示输入错误句子，得到纠正后的句子。

```python
python predict.py --init_checkpoint_path ernie_log/checkpoints5e-5/best_model.pdparams --model_type ernie --model_name_or_path ernie-1.0
```

## 参考文献
* Ruiqing Zhang, Chao Pang et al. "Correcting Chinese Spelling Errors with Phonetic Pre-training", ACL, 2021
