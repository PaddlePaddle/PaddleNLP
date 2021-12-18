# 基于ofa架构在msra实体识别任务的模型压缩
## 直接面向工业化落地场景，训练过程保存最优模型
### 一、教师模型正常训练
```shell
export CUDA_VISIBLE_DEVICES=0

python -u ./run_msra_ner.py \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ../tmp/msra_ner/ \
    --device gpu
```

其中参数释义如下：
- `model_name_or_path`: 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer，支持[PaddleNLP Transformer API](../../../docs/model_zoo/transformers.rst)中除ernie-gen以外的所有模型。若使用非BERT系列模型，需修改脚本导入相应的Task和Tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_seq_length`: 表示最大句子长度，超过该长度将被截断。
- `batch_size`: 表示每次迭代**每张卡**上的样本数目。
- `learning_rate`: 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs`: 表示训练轮数。
- `logging_steps`: 表示日志打印间隔。
- `save_steps`: 表示模型保存及评估间隔。
- `output_dir`: 表示模型保存路径。
- `device`: 训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。


### 二、OFA接口，模型压缩训练

单卡训练
```shell
export TASK_NAME=msra
export task_pretrained_model_dir=../tmp/msra_ner/model_best/

python -u ./run_msra_ner_ofa.py --model_type bert \
          --model_name_or_path ${task_pretrained_model_dir} \
          --task_name $TASK_NAME \
          --max_seq_length 128     \
          --batch_size 32       \
          --learning_rate 2e-5     \
          --num_train_epochs 30     \
          --logging_steps 1     \
          --save_steps 100     \
          --output_dir ../tmp/msra_ner_ofa/ \
          --device gpu  \
          --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5
```

其中参数释义如下：
- `model_type` 指示了模型类型，当前仅支持BERT模型。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `task_name` 表示 Fine-tuning 的任务。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `width_mult_list` 表示压缩训练过程中，对每层Transformer Block的宽度选择的范围。


### 三、将训练完的超网络模型导出成子模型
根据传入的config导出相应的子模型并转为静态图模型。
启动命令：

```shell
python -u ./export_model_msra_ner_ofa.py --model_type bert \
                --model_name_or_path ../tmp/msra_ner_ofa/model_0.5 \
                --max_seq_length 128     \
                --sub_model_output_dir ../tmp/msra_ner_ofa/model_0.5/dynamic_model \
                --static_sub_model ../tmp/msra_ner_ofa/model_0.5/static_model \
                --n_gpu 1 \
                --width_mult  0.5 
```

其中参数释义如下：
- `model_type` 指示了模型类型，当前仅支持BERT模型。
- `model_name_or_path` 指示了某种特定配置的经过OFA训练后保存的模型，对应有其预训练模型和预训练时使用的tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。默认：128.
- `sub_model_output_dir` 指示了导出子模型动态图参数的目录。
- `static_sub_model` 指示了导出子模型静态图模型及参数的目录，设置为None，则表示不导出静态图模型。默认：None。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `width_mult` 表示导出子模型的宽度。默认：1.0.


### 四、项目在BML CodeLab运行的链接，
替换成自己的训练集后，可以一键启动训练，应用在实际的工业化落地场景进行模型部署，在基本无精度损失的情况下，提速相当明显，预测速度提升100%  

| Task  | Metric                       | Result            | Result with PaddleSlim | 
|:-----:|:----------------------------:|:-----------------:|:----------------------:|
| MSRA  | ChunkEvaluator               |      0.94738      |       0.93754          |

| Task  | Latency(s)                   | Result            | Result with PaddleSlim | 
|:-----:|:----------------------------:|:-----------------:|:----------------------:|
| MSRA  | 10451个中文字符(i7-10510U CPU) |      18s-19s      |       8-10s            |

地址：https://aistudio.baidu.com/aistudio/projectdetail/2876713?contributionType=1&shared=1