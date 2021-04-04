# 阅读理解 DuReader

## 简介

在机器阅读理解(MRC)任务中，我们会给定一个问题(Q)以及一个或多个段落(P)/文档(D)，然后利用机器在给定的段落中寻找正确答案(A)，即Q + P or D => A. 机器阅读理解(MRC)是自然语言处理(NLP)中的关键任务之一，需要机器对语言有深刻的理解才能找到正确的答案。

本示例采用BERT和ERNIE模型通过Question Awnsering接口快速完成阅读理解任务的Finetuning。

## 数据集

**DuReader**是一个大规模、面向真实应用、由人类生成的中文阅读理解数据集。DuReader聚焦于真实世界中的不限定领域的问答任务。相较于其他阅读理解数据集，DuReader的优势包括:

* 问题来自于真实的搜索日志
* 文章内容来自于真实网页
* 答案由人类生成
* 面向真实应用场景
* 标注更加丰富细致

更多关于**DuReader**数据集的详细信息可参考[BROAD官网](https://ai.baidu.com//broad/subordinate?dataset=dureader)。

## 快速开始

### 数据准备

为了方便开发者快速测试，PaddleNLP Dataset API内置了DuReader数据集，用户也可以通过`--data_path`传入自定义的阅读理解数据集的目录，数据集需保证与DuReader数据集格式一致。

### Fine-tune

按如下方式启动 Fine-tuning:

```shell
python -u ./run_du.py \
    --model_type bert \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 512 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/dureader/ \
    --n_gpu 2 \
 ```

* `model_type`: 预训练模型的种类。如bert，ernie，roberta等。
* `model_name_or_path`: 预训练模型的具体名称。如bert-base-uncased，bert-large-cased等。或者是模型文件的本地路径。
* `output_dir`: 保存模型checkpoint的路径。

训练结束后会在模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "ROUGE-L": ,
  "BLEU-4":
}
```

**NOTE:** 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如`--model_name_or_path=./tmp/dureader/model_19000/`，程序会自动加载模型参数`/model_state.pdparams`，也会自动加载词表，模型config和tokenizer的config。

### 2. 目录结构

```text
.
├── README.md           # 文档
├── run_du.py           # 训练代码  
├── args.py             # 参数读取
```

## Reference

- [DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications](https://arxiv.org/abs/1711.05073)