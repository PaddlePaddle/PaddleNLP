# C-Eval 评测脚本

此 C-Eval 评测脚本修改自[ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)项目。

## 数据准备

从 C-Eval 官方指定路径下载评测数据集，并解压至 data 文件夹：

```
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip -d data
```
将 data 文件夹放置于本项目的 scripts/ceval 目录下。

## 运行预测脚本

运行以下脚本：

```
cd scripts/ceval
python eval.py \
    --model_name_or_path /path/to/your/model \
    --cot False \
    --few_shot False \
    --with_prompt True \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --ntrain 5 \
    --do_save_csv False \
    --do_test False \
    --output_dir ${output_path} \
```

参数说明

- model_path：待评测模型所在目录（合并 LoRA 后的 HF 格式模型）
- cot：是否使用 chain-of-thought
- few_shot：是否使用 few-shot
- ntrain：few_shot=True 时，指定 few-shot 实例的数量（5-shot：ntrain=5）；few_shot=False 时该项不起作用
- with_prompt：模型输入是否包含针对 Alpaca 模型的指令模板
- constrained_decoding：由于 C-Eval 评测的标准答案格式为选项'A'/'B'/'C'/'D'，所以我们提供了两种从模型生成内容中抽取答案的方案：
    - 当 constrained_decoding=True，计算模型生成的第一个 token 分别为'A', 'B', 'C', 'D'的概率，选择其中概率最大的一个作为答案
    - 当 constrained_decoding=False，用正则表达式从模型生成内容中提取答案
- temperature：模型解码时的温度
- n_times：指定评测的重复次数，将在 output_dir 下生成指定次数的文件夹
- do_save_csv：是否将模型生成结果、提取的答案等内容保存在 csv 文件中
- output_dir：指定评测结果的输出路径
- do_test：在 valid 或 test 集上测试：当 do_test=False，在 valid 集上测试；当 do_test=True，在 test 集上测试

## 评测输出
模型预测完成后，生成目录`outputs/take*`，其中*代表数字，范围为0至`n_times-1`，分别储存了`n_times`次解码的结果。

`outputs/take*`下包含`submission.json`和`summary.json`两个 json 文件。若`do_save_csv=True`，还将包含52个保存的模型生成结果、提取的答案等内容的 csv 文件。

`submission.json`为依据官方提交规范生成的存储模型评测答案的文件，形式如：

```
{
    "computer_network": {
        "0": "A",
        "1": "B",
        ...
    },
      "marxism": {
        "0": "B",
        "1": "A",
        ...
      },
      ...
}
```

summary.json 包含模型在52个主题下、4个大类下和总体平均的评测结果。例如，json 文件最后的 All 字段中会显示总体平均效果：

```
  "All": {
    "score": 0.36701337295690933,
    "num": 1346,
  "correct": 494.0
}
```

其中 score 为准确率，num 为测试的总样本条数，correct 为正确的数量。
