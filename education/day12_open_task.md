# Day12 预训练模型小型化与部署 - 【开放性题目】辅导

本教程旨在辅导同学如何完成产业实践开放性题目（参照ERNIE-1.0的压缩部署流程，改用ERNIE-Gram的Teacher模型进行蒸馏）——[『NLP打卡营』实践课12：预训练模型小型化与部署实战](https://aistudio.baidu.com/aistudio/projectdetail/2114383)

必修作业是跑通原项目即可噢。

## 1. 对ERNIE-Gram进行fine-tuning得到教师模型
由于我们的蒸馏是在中文情感分析ChnSentiCorp任务上，因此我们需要对PaddleNLP提供的ERNIE-Gram在我们的任务上进行Fine-tuning。下面是详细的步骤：

在[PaddleNLP Transformer API](../docs/model_zoo/transformers.rst)查询PaddleNLP所支持的Transformer预训练模型。我们可以在这里找到ERNIE-Gram中的**ernie-gram-zh**。
参考AI studio教程中在中文情感分类ChnSentiCorp数据集下对**ERNIE-1.0**进行fine-tuning的方法，即对[PaddleNLP的run_glue脚本](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/benchmark/glue)进行适当修改，使其支持ERNIE-Gram在中文情感分类数据上的fine-tuning。

我们需要导入ERNIE-Gram模型所依赖的相关模块：

```python
from paddlenlp.transformers import ErnieGramForSequenceClassification, ErnieGramTokenizer
```

### a.对现有的run_glue.py脚本修改使其支持ChnSentiCorp任务

增加对ChnSentiCorp数据集的评估指标的配置

```python
METRIC_CLASSES = {
    "chnsenticorp": Accuracy,
}
```

增加对使用的预训练模型的配置

```python
MODEL_CLASSES = {
    "ernie-gram": (ErnieGramForSequenceClassification, ErnieGramTokenizer),
}
```

获取数据集的调用需要更新为：

```python
train_ds = load_dataset('chnsenticorp', splits='train')
dev_ds = load_dataset('chnsenticorp', splits='dev')
```

接着需要更新`convert_example`函数：

```python
def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['label']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    example = tokenizer(example['text'], max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']
```

### b.在ChnSentiCorp上对ERNIE-Gram进行fine-tuning

现在，我们就可以对ERNIE-Gram模型在ChnSentiCorp数据集上进行finetuning了~

可以使用下面的命令对ERNIE的预训练模型进行finetuning：

```shell

export TASK_NAME=ChnSentiCorp

python -u ./run_glue.py \
    --model_type ernie-gram \
    --model_name_or_path ernie-gram-zh \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 24   \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 10 \
    --output_dir ./tmp/$TASK_NAME/ \
    --device gpu # or cpu

```

## 2.用Fine-tuned ERNIE-Gram对Bi-LSTM进行蒸馏

由于本节的作业只需要替换蒸馏时使用的教师模型，因此我们只需要在蒸馏前重新导入第一步微调得到的教师模型即可。

假设我们对ERNIE-Gram Fine-tuning得到的最好的模型位于./tmp/ChnSentiCorp/best_model，那么下面的脚本可以导入教师模型：
```python
from paddlenlp.transformers import ErnieGramForSequenceClassification, ErnieGramTokenizer
teacher = ErnieGramForSequenceClassification.from_pretrained("./tmp/ChnSentiCorp/best_model")
```

蒸馏的过程同AI studio教程，这里就不再赘述啦~同学们按着与教程相同的步骤进行即可。同时，本repo中也提供了一个[从BERT到Bi-LSTM蒸馏](../examples/model_compression/distill_lstm)的例子可供参考。
