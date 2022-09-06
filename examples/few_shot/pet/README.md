# [PET](https://arxiv.org/abs/2001.07676)

[PET](https://arxiv.org/abs/2001.07676) (Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference)  提出将输入示例转换为完形填空式短语，以帮助语言模型理解给定的任务

## 代码结构及说明
```
|—— pet.py # PET 策略的训练、评估主脚本
|—— dataset.py # PET 策略针对 FewCLUE 9 个数据集的任务转换逻辑，以及明文 -> 训练数据的转换
|—— model.py # PET 的网络结构
|—— evaluate.py # 针对 FewCLUE 9 个数据集的评估函数
|—— predict.py # 针对 FewCLUE 9 个数据集进行预测
```


## 基于 FewCLUE 进行 PET 实验
PaddleNLP 内置了 FewCLUE 数据集，可以直接用来进行 PET 策略训练、评估、预测，并生成 FewCLUE 榜单的提交结果，参与 FewCLUE 竞赛。

###  数据准备
基于 FewCLUE 数据集进行实验只需要  1 行代码，这部分代码在 `pet.py` 脚本中

```
from paddlenlp.datasets import load_dataset

# 通过指定 "fewclue" 和数据集名字 name="tnews" 即可一键加载 FewCLUE 中的 tnews 数据集
train_ds, dev_ds, public_test_ds = load_dataset("fewclue", name="tnews", splits=("train_0", "dev_0", "test_public"))
````
### 模型训练&评估
通过如下命令，指定 GPU 0 卡,  在 FewCLUE 的 `tnews` 数据集上进行训练&评估
```
python -u -m paddle.distributed.launch --gpus "0" \
    pet.py \
	--task_name "tnews" \
	--device gpu \
    --pattern_id 0 \
	--save_dir ./tnews \
	--index 0 \
	--batch_size 16 \
	--learning_rate 1E-4 \
	--epochs 10 \
	--max_seq_length 512 \
	--language_model "ernie-3.0-medium-zh" \
    --rdrop_coef 0 \
```
参数含义说明
- `task_name`: FewCLUE 中的数据集名字
- `device`: 使用 cpu/gpu 进行训练
- `pattern_id` 完形填空的模式
- `save_dir`: 模型存储路径
- `max_seq_length`: 文本的最大截断长度
- `rdrop_coef`: R-Drop 策略 Loss 的权重系数，默认为 0， 若为 0 则未使用 R-Drop 策略

模型每训练 1 个 epoch,  会在验证集上进行评估

### 模型预测
通过如下命令，指定 GPU 0 卡， 在 `FewCLUE` 的 `tnews` 数据集上进行预测
```
python -u -m paddle.distributed.launch --gpus "0" predict.py \
        --task_name "tnews" \
        --device gpu \
        --init_from_ckpt "./tnews/model_120/model_state.pdparams" \
        --output_dir "./tnews/output" \
        --batch_size 32 \
        --max_seq_length 512
```

### 模型导出
在动态图模型训练完毕之后，可以将动态图模型导出成静态图模型用于预测部署。

```python
python export_model.py --params_path=./tnews/model_120/model_state.pdparams --output_path=./export
```
**注意params_path需要填写训练完毕之后真实路径**

## References
[1] Schick, Timo, and Hinrich Schütze. “Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference.” ArXiv:2001.07676 [Cs], January 25, 2021. http://arxiv.org/abs/2001.07676.
