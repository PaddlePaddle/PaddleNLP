# PaddleNLP Transformer API

随着深度学习的发展，NLP领域涌现了一大批高质量的Transformer类预训练模型，多次刷新各种NLP任务SOTA。PaddleNLP为用户提供了常用的BERT、ERNIE、RoBERTa、XLNet经典结构预训练模型，让开发者能够方便快捷应用各类Transformer预训练模型及其下游任务。


## Transformer预训练模型汇总

下表汇总了目前PaddleNLP支持的各类预训练模型。用户可以使用PaddleNLP提供的模型，完成问答、文本分类、序列标注、文本生成等任务。同时我们提供了29种预训练的参数权重供用户使用，其中包含了12种中文语言模型的预训练权重。

| Model | Tokenizer | Supported Task | Pretrained Weight|
|---|---|---|---|
| [BERT](https://arxiv.org/abs/1810.04805) | BertTokenizer|BertModel<br> BertForQuestionAnswering<br> BertForSequenceClassification<br>BertForTokenClassification| `bert-base-uncased`<br> `bert-large-uncased` <br>`bert-base-multilingual-uncased` <br>`bert-base-cased`<br> `bert-base-chinese`<br> `bert-base-multilingual-cased`<br> `bert-large-cased`<br> `bert-wwm-chinese`<br> `bert-wwm-ext-chinese` |
|[ERNIE](https://arxiv.org/abs/1904.09223)|ErnieTokenizer<br>ErnieTinyTokenizer|ErnieModel<br> ErnieForQuestionAnswering<br> ErnieForSequenceClassification<br> ErnieForTokenClassification | `ernie-1.0`<br> `ernie-tiny`<br> `ernie-2.0-en`<br> `ernie-2.0-large-en`|
|[ERNIE-GEN](https://arxiv.org/abs/2001.11314)|ErnieTokenizer| ErnieForGeneration|`ernie-gen-base-en`<br>`ernie-gen-large-en`<br>`ernie-gen-large-en-430g`|
|[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)| GPT2Tokenizer<br> GPT2ChineseTokenizer| GPT2ForGreedyGeneration| `gpt2-base-cn` <br> `gpt2-medium-en`|
|[RoBERTa](https://arxiv.org/abs/1907.11692)|RobertaTokenizer| RobertaModel<br>RobertaForQuestionAnswering<br>RobertaForSequenceClassification<br>RobertaForTokenClassification| `roberta-wwm-ext`<br> `roberta-wwm-ext-large`<br> `rbt3`<br> `rbtl3`|
|[ELECTRA](https://arxiv.org/abs/2003.10555) | ElectraTokenizer| ElectraModel<br>ElectraForSequenceClassification<br>ElectraForTokenClassification<br>|`electra-small`<br> `electra-base`<br> `electra-large`<br> `chinese-electra-small`<br> `chinese-electra-base`<br>|
|[XLNet](https://arxiv.org/abs/1906.08237)| XLNetTokenizer| XLNetModel<br> XLNetForSequenceClassification<br> XLNetForTokenClassification |`xlnet-base-cased`<br> `xlnet-large-cased`|
|[Transformer](https://arxiv.org/abs/1706.03762) |- | TransformerModel | - |

**NOTE**：其中中文的预训练模型有`bert-base-chinese, bert-wwm-chinese, bert-wwm-ext-chinese, ernie-1.0, ernie-tiny, gpt2-base-cn, roberta-wwm-ext, roberta-wwm-ext-large, rbt3, rbtl3, chinese-electra-base, chinese-electra-small`。

## 预训练模型使用方法

PaddleNLP Transformer API在提丰富预训练模型的同时，也降低了用户的使用门槛。只需十几行代码，用户即可完成模型加载和下游任务Fine-tuning。

```python
import paddle
from paddlenlp.datasets import ChnSentiCorp
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer

train_ds, dev_ds, test_ds = ChnSentiCorp.get_datasets(['train', 'dev', 'test'])

model = BertForSequenceClassification.from_pretrained("bert-wwm-chinese", num_classes=len(train_ds.get_labels()))

tokenizer = BertTokenizer.from_pretrained("bert-wwm-chinese")

# Define the dataloader from dataset and tokenizer here

optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())

criterion = paddle.nn.loss.CrossEntropyLoss()

for input_ids, token_type_ids, labels in train_dataloader:
    logits = model(input_ids, token_type_ids)
    loss = criterion(logits, labels)
    probs = paddle.nn.functional.softmax(logits, axis=1)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
```

上面的代码给出使用预训练模型的简要示例，更完整详细的示例代码，可以参考[使用预训练模型Fine-tune完成中文文本分类任务](../examples/text_classification/pretrained_models)。

1. 加载数据集：PaddleNLP内置了多种数据集，用户可以一键导入所需的数据集。
2. 加载预训练模型：PaddleNLP的预训练模型可以很容易地通过`from_pretrained()`方法加载。第一个参数是汇总表中对应的 `Pretrained Weight`，可加载对应的预训练权重。`BertForSequenceClassification`初始化`__init__`所需的其他参数，如`num_classes`等，也是通过`from_pretrained()`传入。`Tokenizer`使用同样的`from_pretrained`方法加载。
3. 使用tokenier将dataset处理成模型的输入。此部分可以参考前述的详细示例代码。
4. 定义训练所需的优化器，loss函数等，就可以开始进行模型fine-tune任务。


## 预训练模型适用任务汇总

本小节按照模型适用的不同任务类型，对上表[Transformer预训练模型汇总](#Transformer预训练模型汇总)的Task进行分类汇总。主要包括文本分类、序列标注、问答任务、文本生成、机器翻译等。

|任务|模型|应用场景|预训练权重|
|---|---|---|---|
|文本分类<br>SequenceClassification |BertForSequenceClassification <br> ErnieForSequenceClassification <br> RobertaForSequenceClassification <br> ElectraForSequenceClassification <br> XLNetForSequenceClassification | [文本分类](../examples/text_classification/pretrained_models/)、[阅读理解](../examples/machine_reading_comprehension/DuReader-yesno/)等| [见上表](#Transformer预训练模型汇总)|
|序列标注<br>TokenClassification|BertForTokenClassification <br> ErnieForTokenClassification <br> RobertaForTokenClassification <br> ElectraForTokenClassification <br> XLNetForTokenClassification | [命名实体标注](../examples/named_entity_recognition/)等|[见上表](#Transformer预训练模型汇总)|
|问答任务<br>QuestionAnswering|BertForQuestionAnswering <br> ErnieForQuestionAnswering <br> RobertaForQuestionAnswering| [阅读理解](../examples/machine_reading_comprehension/SQuAD/)等|[见上表](#Transformer预训练模型汇总)|
|文本生成<br>TextGeneration | 	ErnieForGeneration <br> GPT2ForGreedyGeneration |[文本生成](../examples/text_generation/ernie-gen)等|[见上表](#Transformer预训练模型汇总)|
|机器翻译<br>MachineTranslation| TransformerModel | [机器翻译](../examples/machine_translation/transformer/)|[见上表](#Transformer预训练模型汇总)|

用户可以切换表格中的不同模型，来处理相同类型的任务。如对于[预训练模型使用方法](#预训练模型使用方法)中的文本分类任务，用户可以将`BertForSequenceClassification`换成`ErnieForSequenceClassification`, 来寻找更适合的预训练模型。

## 参考资料：
- 部分中文预训练模型来自：https://github.com/ymcui/Chinese-BERT-wwm
- Sun, Yu, et al. "Ernie: Enhanced representation through knowledge integration." arXiv preprint arXiv:1904.09223 (2019).
- Cui, Yiming, et al. "Pre-training with whole word masking for chinese bert." arXiv preprint arXiv:1906.08101 (2019).
