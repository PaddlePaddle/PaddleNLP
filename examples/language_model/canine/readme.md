# CANINE 使用示例

论文链接：[CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://paperswithcode.com/paper/canine-pre-training-an-efficient-tokenization)

AIstudio CANINE 复现 项目链接 [基于 Paddle 实现 Canine 模型](https://aistudio.baidu.com/aistudio/projectdetail/4063353?contributionType=1&shared=1)

世界上存在海量的语言与词汇，在处理多语言场景时，传统预训练模型采用的 Vocab 和 Tokenization 方案难免会遇到 out of vocabulary 和 unkonw token 的情况。 Canine 提供了 tokenization-free 的预训练方案，提高了模型在多语言任务下的能力，并在 TydiQA 任务上取得了高出 TydiQA 基线 3% 的成绩。

本教程中使用 SQuAD 任务进行 Canine 模型使用演示，该文件中的大部分注解文档引用于 [阅读理解 SQuAD 说明文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_reading_comprehension/SQuAD/README.md)


## 快速开始

以下命令请在 `Paddlenlp/examples/language_model/canine` 文件夹下执行。

#### SQuAD v1.1 微调

```shell
python run_squad.py \
    --model_type canine \
    --model_name_or_path canine-s \
    --max_seq_length 2048 \
    --doc_stride 512 \
    --batch_size 4 \
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

* `model_type`: 预训练模型的种类。此处仅支持 `canine`。
* `model_name_or_path`: 预训练模型的具体名称。如 `canine-s` 。或者是模型文件的本地路径。
* `doc_stride`: 在机械阅读任务中，对长文章进行截断采样的 `stride` 间隔值。
* `output_dir`: 保存模型checkpoint的路径。
* `do_train`: 是否进行训练。
* `do_predict`: 是否进行预测。
* `device`:  训练使用的设备，输入 `cpu` 或 `gpu`。
* `fp16`：是否进行混合精度训练。若不提供该参数，则采用全 FP32 训练。

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

#### SQuAD v2.0 微调

```shell
python run_squad.py \
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

+ `version_2_with_negative`: 使用 squad2.0 数据集和评价指标的标志。
+ 关于其他参数解释，请参考上文 [SQuAD v1.1 微调]() 小节。

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

其中会输出 `best_f1_thresh` 是最佳阈值，可以使用这个阈值重新训练，或者从 `all_nbest_json`变量中获取最终 `prediction`。 训练方法与前面大体相同，只需要设定 `--null_score_diff_threshold` 参数的值为测评时输出的 `best_f1_thresh` ，通常这个值在 -1.0 到 -5.0 之间。

**NOTE:** 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如 `--model_name_or_path=./tmp/squad/model_19000/`，程序会自动加载模型参数 `/model_state.pdparams`，也会自动加载词表，模型config和tokenizer的config。

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
