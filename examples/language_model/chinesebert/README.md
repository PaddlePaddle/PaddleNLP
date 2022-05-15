# ChineseBert with PaddleNLP

[ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://arxiv.org/pdf/2106.16038.pdf)

**摘要：**
最近的汉语预训练模型忽略了汉语特有的两个重要方面：字形和拼音，它们对语言理解具有重要的语法和语义信息。在本研究中，我们提出了汉语预训练，它将汉字的字形和拼音信息纳入语言模型预训练中。字形嵌入是基于汉字的不同字体获得的，能够从视觉特征中捕捉汉字语义，拼音嵌入代表汉字的发音，处理汉语中高度流行的异义现象（同一汉字具有不同的发音和不同的含义）。在大规模的未标记中文语料库上进行预训练后，所提出的ChineseBERT模型在训练步骤较少的基线模型上产生了显著的性能提高。该模型在广泛的中国自然语言处理任务上实现了新的SOTA性能，包括机器阅读理解、自然语言推理、文本分类、句子对匹配和命名实体识别方面的竞争性能。

本项目是 ChineseBert 在 Paddle 2.x上的开源实现。

## **数据准备**
涉及到的ChnSentiCorp，crmc2018，XNLI数据
部分Paddle已提供，其他可参考https://github.com/27182812/ChineseBERT_paddle,
在data目录下。


## **模型预训练**
模型预训练过程可参考[Electra的README](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/electra/README.md)

## **Fine-tuning**

### 运行Fine-tuning

#### **使用Paddle提供的预训练模型运行 Fine-tuning**

#### 1、ChnSentiCorp
以ChnSentiCorp数据集为例

#### （1）模型微调：
```shell
# 运行训练
python train_chn.py \
--data_path './data/ChnSentiCorp' \
--device 'gpu' \
--epochs 10 \
--max_seq_length 512 \
--batch_size 8 \
--learning_rate 2e-5 \
--weight_decay 0.0001 \
--warmup_proportion 0.1 \
--seed 2333 \
--save_dir 'outputs/chn' | tee outputs/train_chn.log
```
其中参数释义如下：
- `data_path` 表示微调数据路径
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。
- `epochs` 表示训练轮数。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示优化器中使用的weight_decay的系数。
- `warmup_steps` 表示动态学习率热启动的step数。
- `seed` 指定随机种子。
- `save_dir` 表示模型保存路径。

#### (2) 评估

在dev和test数据集上acc分别为95.8和96.08，达到论文精度要求。

#### 2、XNLI

#### （1）训练

```bash
python train_xnli.py \
--data_path './data/XNLI' \
--device 'gpu' \
--epochs 5 \
--max_seq_len 256 \
--batch_size 16 \
--learning_rate 1.3e-5 \
--weight_decay 0.001 \
--warmup_proportion 0.1 \
--seed 2333 \
--save_dir outputs/xnli | tee outputs/train_xnli.log
```
其中参数释义如下：
- `data_path` 表示微调数据路径
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。
- `epochs` 表示训练轮数。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示优化器中使用的weight_decay的系数。
- `warmup_steps` 表示动态学习率热启动的step数。
- `seed` 指定随机种子。
- `save_dir` 表示模型保存路径。

#### （2）评估

test数据集 acc最好结果为81.657,达到论文精度要求。

#### 3、cmrc2018

#### (1) 训练

```shell
# 开始训练
python train_cmrc2018.py \
    --data_dir "data/cmrc2018" \
    --model_name_or_path ChineseBERT-large \
    --max_seq_length 512 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --eval_batch_size 16 \
    --learning_rate 4e-5 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --logging_steps 2 \
    --save_steps 20 \
    --warmup_radio 0.1 \
    --weight_decay 0.01 \
    --output_dir outputs/cmrc2018 \
    --seed 1111 \
    --num_workers 0 \
    --use_amp
```
其中参数释义如下：
- `data_path` 表示微调数据路径。
- `model_name_or_path` 模型名称或者路径，支持ChineseBERT-base、ChineseBERT-large两种种规格。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `train_batch_size` 表示训练过程中每次迭代**每张卡**上的样本数目。
- `gradient_accumulation_steps` 梯度累加步数。
- `eval_batch_size` 表示验证过程中每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `max_grad_norm` 梯度裁剪。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `warmup_radio` 表示动态学习率热启动的比例。
- `weight_decay` 表示优化器中使用的weight_decay的系数。
- `output_dir` 表示模型保存路径。
- `seed` 指定随机种子。
- `num_workers` 表示同时工作进程。
- `use_amp` 表示是否使用混合精度。

训练过程中模型会在dev数据集进行评估，其中最好的结果如下所示：

```python

{
    AVERAGE = 82.791
    F1 = 91.055
    EM = 74.526
    TOTAL = 3219
    SKIP = 0
}

```

#### （2）运行eval_cmrc.py，生成test数据集预测答案

```bash
python eval_cmrc.py --model_name_or_path outputs/step-340 --n_best_size 35 --max_answer_length 65
```

其中，model_name_or_path为模型路径

#### （3）提交CLUE

test数据集 EM为78.55，达到论文精度要求


## Reference

```bibtex
@article{sun2021chinesebert,
  title={ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information},
  author={Sun, Zijun and Li, Xiaoya and Sun, Xiaofei and Meng, Yuxian and Ao, Xiang and He, Qing and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:2106.16038},
  year={2021}
}

```
