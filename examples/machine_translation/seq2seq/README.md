# Machine Translation using Seq2Seq with Attention

以下是本范例模型的简要目录结构及说明：

```
.
├── deploy                 # 预测部署目录
│ └── python
│   └── infer.py           # 用预测模型进行推理的程序
├── README.md              # 文档，本文件
├── args.py                # 训练、预测、导出模型以及模型参数配置程序
├── data.py                # 数据读入程序
├── train.py               # 训练主程序
├── predict.py             # 预测主程序
├── export_model.py        # 导出预测模型的程序
└── seq2seq_attn.py        # 带注意力机制的翻译模型程序
```

## 简介

Sequence to Sequence (Seq2Seq)，使用编码器-解码器（Encoder-Decoder）结构，用编码器将源序列编码成vector，再用解码器将该vector解码为目标序列。Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。

本目录包含Seq2Seq的一个经典样例：机器翻译，带Attention机制的翻译模型。Seq2Seq翻译模型，模拟了人类在进行翻译类任务时的行为：先解析源语言，理解其含义，再根据该含义来写出目标语言的语句。更多关于机器翻译的具体原理和数学表达式，我们推荐参考飞桨官网[机器翻译案例](https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/nlp_case/machine_translation/README.cn.html)。

## 模型概览

本模型中，在编码器方面，我们采用了基于LSTM的多层的RNN encoder；在解码器方面，我们使用了带注意力（Attention）机制的RNN decoder，在预测时我们使用柱搜索（beam search）算法来生成翻译的目标语句。

## 数据介绍

本教程使用[IWSLT'15 English-Vietnamese data ](https://nlp.stanford.edu/projects/nmt/)数据集中的英语到越南语的数据作为训练语料，tst2012的数据作为开发集，tst2013的数据作为测试集。

### 数据获取
如果用户在初始化数据集时没有提供路径，数据集会自动下载到`paddlenlp.utils.env.DATA_HOME`的`IWSLT15/`路径下，例如在linux系统下，默认存储路径是`~/.paddlenlp/datasets/IWSLT15`。

## 模型训练

执行以下命令即可训练带有注意力机制的Seq2Seq机器翻译模型：

```sh
python train.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --device gpu \
    --model_path ./attention_models
```

各参数的具体说明请参阅 `args.py` 。训练程序会在每个epoch训练结束之后，save一次模型。

**NOTE:** 如需恢复模型训练，则`init_from_ckpt`只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=attention_models/5`即可，程序会自动加载模型参数`attention_models/5.pdparams`，也会自动加载优化器状态`attention_models/5.pdopt`。

## 模型预测

训练完成之后，可以使用保存的模型（由 `--init_from_ckpt` 指定）对测试集的数据集进行beam search解码。生成的翻译结果位于`--infer_output_file`指定的路径，预测命令如下：

```sh
python predict.py \
     --num_layers 2 \
     --hidden_size 512 \
     --batch_size 128 \
     --dropout 0.2 \
     --init_scale  0.1 \
     --max_grad_norm 5.0 \
     --init_from_ckpt attention_models/9 \
     --infer_output_file infer_output.txt \
     --beam_size 10 \
     --device gpu
```

各参数的具体说明请参阅 `args.py` ，注意预测时所用模型超参数需和训练时一致。

## 预测效果评价
取第10个epoch的结果，用取beam_size为10的beam search解码，`predict.py`脚本在生成翻译结果之后，会调用`paddlenlp.metrics.BLEU`计算翻译结果的BLEU指标，最终计算出的BLEU分数为0.24329954822714048

## 保存预测模型
这里指定的参数`export_path` 表示导出预测模型文件的前缀。保存时会添加后缀（`pdiparams`，`pdiparams.info`，`pdmodel`）。
```shell
python export_model.py \
     --num_layers 2 \
     --hidden_size 512 \
     --batch_size 128 \
     --dropout 0.2 \
     --init_scale  0.1 \
     --max_grad_norm 5.0 \
     --init_from_ckpt attention_models/9.pdparams \
     --beam_size 10 \
     --export_path ./infer_model/model
```

## 基于预测引擎推理
然后按照如下的方式对IWSLT15数据集中的测试集（有标注的）进行预测（基于Paddle的[Python预测API](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/python_infer_cn.html)）：

```shell
cd deploy/python
python infer.py \
    --export_path ../../infer_model/model \
    --device gpu \
    --batch_size 128 \
    --infer_output_file infer_output.txt
```
