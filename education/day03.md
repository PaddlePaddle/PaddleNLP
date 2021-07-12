# Day03 词法分析作业辅导

本教程旨在辅导同学如何完成 AI Studio课程——[『NLP打卡营』实践课3：使用预训练模型实现快递单信息抽取
](https://aistudio.baidu.com/aistudio/projectdetail/1329361)课后作业。

## 1. 更换预训练模型

在[PaddleNLP Transformer API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/transformers.rst#transformer%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E6%B1%87%E6%80%BB)查询PaddleNLP所支持的Transformer预训练模型。选择其中一个模型，如**bert-base-chinese**，只需将代码中的

```python
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification

model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
```

修改为

```python
from paddlenlp.transformers import BertTokenizer, BertForTokenClassification

model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_classes=len(label_vocab))
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
```

即可将预训练模型从**ernie-1.0**切换至**bert-base-chinese**。

## 2. 更换数据集

PaddleNLP集成了一系列[序列标注数据集](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/datasets.md#%E5%BA%8F%E5%88%97%E6%A0%87%E6%B3%A8)，用户可以一键调用相应API快速下载调用相关数据集，我们在这里选择其中**MSRA_NER**数据集，将

```python
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

# Create dataset, tokenizer and dataloader.
train_ds, dev_ds, test_ds = load_dataset(datafiles=(
        './data/train.txt', './data/dev.txt', './data/test.txt'))
```

修改为

```python
from paddlenlp.datasets import load_dataset

# 由于MSRA_NER数据集没有dev dataset，我们这里重复加载test dataset作为dev_ds
train_ds, dev_ds, test_ds = load_dataset(
        'msra_ner', splits=('train', 'test', 'test'), lazy=False)

# 注意删除 label_vocab = load_dict('./data/tag.dic')
label_vocab = {label:label_id for label_id, label in enumerate(train_ds.label_list)}
```

### 2.1 适配数据集预处理

为了适配该数据集，我们还需要修改数据预处理代码，修改`utils.py`中的`convert_example`函数为：

```python
def convert_example(example, tokenizer, label_vocab, max_seq_len=128):
    labels = example['labels']
    example = example['tokens']
    no_entity_id = label_vocab['O']
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len'], tokenized_input['labels']
```


### 2.2  适配数据集后处理

不同于快递单数据集，`MSRA_NER`数据集的标注采用的是'BIO'在前的标注方式，因此还需要修改`utils.py`中的`parse_decodes`函数为：


```python
def parse_decodes(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx]['tokens'][:end]
        tags = [id_label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.startswith('B-') or t == 'O':
                if len(words):
                    sent_out.append(words)
                if t.startswith('B-'):
                    tags_out.append(t.split('-')[1])
                else:
                    tags_out.append(t)
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs
```
