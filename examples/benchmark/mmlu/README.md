# MMLU 英文评测数据集
MMLU ([Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300v3.pdf))用于衡量文本模型在多种任务上的准确性，是目前主流的 LLM 英文评测数据集。该数据集涵盖了57个任务，包括基础数学、美国历史、计算机科学、法律等等。

此 MMLU 评测脚本修改自[hendrycks/test](https://github.com/hendrycks/test)项目。

## 数据准备

从指定路径下载评测数据集：

```
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar xf data.tar
```

## 运行评测脚本

在当前目录下运行以下脚本：

- 单卡运行
```
export CUDA_VISIBLE_DEVICES=0
python eval.py \
    --model_name_or_path /path/to/your/model \
    --temperature 0.2 \
    --ntrain 5 \
    --output_dir ${output_path} \
    --dtype 'float16'
```
- 多卡运行
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.fleet.launch eval.py \
    --model_name_or_path /path/to/your/model \
    --temperature 0.2 \
    --ntrain 5 \
    --output_dir ${output_path} \
    --dtype 'float16' \
    --tensor_parallel_degree 4
```

参数说明

- model_name_or_path：待评测模型所在目录
- ntrain：指定few-shot实例的数量（5-shot：ntrain=5）
- with_prompt：模型输入是否包含针对Alpaca模型的指令模板
- temperature：模型解码时的温度
- output_dir：指定评测结果的输出路径

## 评测输出
模型预测完成后，将在输出路径下用csv文件保存57个任务下模型的答题结果，其中 `sumaray.json` 包含模型在22个主题下和总体平均的评测结果。例如，json文件最后的All字段中会显示总体平均效果：

```
  "All": {
    "score": 0.36701337295690933,
    "num": 1346,
  "correct": 494.0
}
```

其中score为准确率，num为测试的总样本条数，correct为正确的数量。
