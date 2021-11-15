# T5
[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683v3.pdf)

## 摘要
迁移学习在自然语言处理(NLP)中已经成为一种强大的技术。迁移学习首先在数据丰富的任务上进行预训练，然后在下游任务上进行调整。迁移学习的有效性引起了不同的方法、方法和实践。在本文中，我们通过引入一个统一的框架，将所有基于文本的语言问题转换为文本到文本的格式，来探索自然语言处理的迁移学习技术。我们的系统研究比较了数十项语言理解任务的训练前目标、架构、未标记数据集、迁移方法和其他因素。通过将我们的探索与规模和我们的新"Colossal Clean Crawled Corpus"数据集相结合，我们在摘要、问答、文本分类等许多基准测试中取得了最先进的结果。为了促进NLP迁移学习的未来工作，我们发布了我们的数据集、预训练模型和代码。

本项目是T5在 Paddle 2.x上的开源实现，包含了`模型权重`转换代码和`GLUE任务`的微调代码。

## 快速开始

### GLUE任务

### 执行Fine-tunning

启动rte分类任务的Fine-tuning的方式如下：

```shell
python run_glue.py \
    --model_name_or_path t5-base \
    --task_name rte \
    --max_seq_length 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 10 \
    --logging_steps 100 \
    --save_steps 100 \
    --seed 42 \
    --scheduler_type linear \
    --output_dir outputs/rte/
```
其中参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录的地址。
- `task_name` GLUE任务名称，可从选["cola","sst-2","mrpc","sts-b","qqp","mnli", "rte", "qnli"]选择。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `train_batch_size` 表示训练时的样本数目。
- `eval_batch_size` 表示验证时的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `warmup_radio` warmup比率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机种子。
- `scheduler_type` scheduler类型，可选linear和cosine，默认linear。
- `output_dir` 表示模型保存路径。

###### t5-base模型在GLUE开发集上的结果：
| Model                          | cola  | sst-2  | mrpc        | sts-b             | qqp         | mnli       | qnli | rte   | mean |
|--------------------------------|-------|-------|-------------|------------------|-------------|-------------|------|-------|-------|
|                                | mcc   | acc   | acc      | pearson | acc      | acc      | acc  | acc   |         |
| T5-base-Paddle | 61.74 | 95.18 | 90.44 | 90.09   | 91.60 | 87.18 | 93.56 | 81.95 | 86.4675 |


### GLUE Demo测试

```sh
python glue_demo.py
```

```
input text: sst2 sentence: contains no wit , only labored gags
label: negative
==================================================
input text: sst2 sentence: that loves its characters and communicates something rather beautiful about human nature
label: positive
==================================================
input text: cola sentence: Mickey looked it up.
label: acceptable
==================================================
input text: sst2 sentence: remains utterly satisfied to remain the same throughout
label: positive
==================================================
input text: sst2 sentence: a well-made and often lovely depiction of the mysteries of friendship
label: positive
==================================================
```

# Reference

```bibtex
@article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html}
}
```
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
