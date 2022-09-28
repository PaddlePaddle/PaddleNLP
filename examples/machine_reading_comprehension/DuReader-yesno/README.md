# 阅读理解 DuReader-yesno

## 简介

### 任务说明
机器阅读理解评测中常用的F1、EM等指标虽然能够很好的衡量抽取式模型所预测的答案和真实答案的匹配程度，但在处理观点类问题时，该类指标难以衡量模型是否真正理解答案所代表的含义，例如答案中包含的观点极性。DuReader-yesno是一个以观点极性判断为目标任务的数据集，通过引入该数据集，可以弥补抽取类数据集的不足，从而更好地评价模型的自然语言理解能力。


### 数据集

该数据集的任务定义如下：
对于一个给定的问题q、一系列相关文档D=d1, d2, …, dn，以及人工抽取答案段落摘要a，要求参评系统自动对问题q、候选文档D以及答案段落摘要a进行分析，输出每个答案段落摘要所表述的是非观点极性。其中，极性分为三类 {Yes, No, Depends}。其中：

* Yes：肯定观点，肯定观点指的是答案给出了较为明确的肯定态度。有客观事实的从客观事实的角度出发，主观态度类的从答案的整体态度来判断。
* No：否定观点，否定观点通常指的是答案较为明确的给出了与问题相反的态度。
* Depends：无法确定/分情况，主要指的是事情本身存在多种情况，不同情况下对应的观点不一致；或者答案本身对问题表示不确定，要具体具体情况才能判断。

例如：
```text
{
    "documents":[
        {
            "title":"香蕉能放冰箱吗 香蕉剥皮冷冻保存_健康贴士_保健_99健康网",
            "paragraphs":[
                "本文导读:............."
            ]
        }
    ],
    "yesno_answer":"No",
    "question":"香蕉能放冰箱吗",
    "answer":"香蕉不能放冰箱，香蕉如果放冰箱里，会更容易变坏，会发黑腐烂。",
    "id":293
}
```

## 快速开始

### Fine-tune

按如下方式启动 Fine-tuning:

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_du.py \
    --model_type ernie_gram \
    --model_name_or_path ernie-gram-zh \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 200 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/dureader-yesno/ \
    --device gpu \
 ```

* `model_type`: 预训练模型的种类。如bert，ernie，roberta等。
* `model_name_or_path`: 预训练模型的具体名称。如bert-base-uncased，bert-large-cased等。或者是模型文件的本地路径。
* `output_dir`: 保存模型checkpoint的路径。

训练结束后模型会自动对结果进行评估，得到类似如下的输出：

```text
accu: 0.874954
```
评估结束后模型会自动对测试集进行预测，并将可提交的结果生成在`prediction.json`中。

**NOTE:** 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如`--model_name_or_path=./tmp/dureader-yesno/model_19000/`，程序会自动加载模型参数`/model_state.pdparams`，也会自动加载词表，模型config和tokenizer的config。
