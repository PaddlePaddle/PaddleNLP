# CANINE

论文链接：[CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://paperswithcode.com/paper/canine-pre-training-an-efficient-tokenization)

世界上存在海量的语言与词汇，在处理多语言场景时，传统预训练模型采用的 Vocab 和 Tokenization 方案难免会遇到 out of vocabulary 和 unkonw token 的情况。 Canine 提供了 tokenization-free 的预训练模型方案，提高了模型在多语言任务下的能力。Canine 在多语言阅读理解数据集 TydiQA 上实现了 Selection Passage Task 66% F1及 Minimum Answer Span Task 58% F1 的精度，比 TydiQA 基线（mBERT）高出约 2%。

## 快速开始

Canine 论文中的指标是基于 TydiQA 数据集测试的，考虑到 TydiQA 数据处理、测试流程多且繁琐，而本仓库主要目的是展示 Canine 模型的使用方式。因此以下模型使用案例将以 SQuAD 任务展开。

关于 Canine 在 TydiQA 上的复现效果，请参考 [canine paddle复现仓库](https://github.com/kevinng77/canine_paddle)。

#### SQuAD v1.1 微调

```shell
python -m paddle.distributed.launch --gpus "0"  run_squad.py \
    --model_type canine \
    --model_name_or_path canine-s \
    --max_seq_length 2048 \
    --doc_stride 512 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 1000 \
    --save_steps 5000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device gpu \
    --do_train \
    --do_predict \
    --fp16
```

训练过程中模型会自动对结果进行评估，如下所示：

```shell
{
  "exact": 73.66130558183538,
  "f1": 82.9771680528166,
  "total": 10570,
  "HasAns_exact": 73.66130558183538,
  "HasAns_f1": 82.9771680528166,
  "HasAns_total": 10570
}
```

Canine 采用了字符编码的方式，提高了其他语种（如日语、韩语）学习能力的同时，牺牲了模型在英文任务上的效果。此外 Canine 的预训练语料也以多语言为准，预训练仅采用了NSP + MLM，因此 Canine 在 TydiQA 上指标比 mBert高，但在 SQuAD 上的指标比Bert 差是符合预期的。Canine 采用了字符编码的方式，提高了其他语种（如日语、韩语）学习能力的同时，牺牲了模型在英文任务上的效果，因此 Canine 在 TydiQA 上指标比 mBert高，但在 SQuAD 上的指标比Bert 差是符合预期的

#### SQuAD v2.0 微调

```shell
python -m example.run_squad \
    --model_type canine \
    --model_name_or_path canine-s \
    --max_seq_length 2048 \
    --doc_stride 512 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 5000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device gpu \
    --do_train \
    --do_predict \
    --version_2_with_negative
```

训练过程中模型会自动对结果进行评估，如下所示：

```shell
{
  "exact": 67.25343215699486,
  "f1": 71.00323746224537,
  "total": 11873,
  "HasAns_exact": 63.37719298245614,
  "HasAns_f1": 70.88755708320467,
  "HasAns_total": 5928,
  "NoAns_exact": 71.11858704793944,
  "NoAns_f1": 71.11858704793944,
  "NoAns_total": 5945,
  "best_exact": 68.55049271456245,
  "best_exact_thresh": -1.6865906715393066,
  "best_f1": 71.76821697826219,
  "best_f1_thresh": -1.2721405029296875
}
```


## 参考

```
@misc{canine,
  title={{CANINE}: Pre-training an Efficient Tokenization-Free Encoder for Language Representation},
  author={Jonathan H. Clark and Dan Garrette and Iulia Turc and John Wieting},
  year={2021},
  eprint={2103.06874},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
