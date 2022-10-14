# Data Augmentation API

PaddleNLP提供了Data Augmentation数据增强API，可用于训练数据数据增强

**目录**
* [词级别数据增强策略](#词级别数据增强策略)
    * [词替换](#词替换)
    * [词插入](#词插入)
    * [词删除](#词删除)
    * [词交换](#词交换)

## 词级别数据增强策略

### 词替换
词替换数据增强策略也即将句子中的词随机替换为其他单词进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.WordSubstitute`进行词级别替换的数据增强。

```text
WordSubstitute 参数介绍：

    aug_type(str or list(str))：
        词替换增强策略类别。可以选择"synonym"、"homonym"、"custom"、"random"、"mlm"或者
        前三种词替换增强策略组合。

    custom_file_path (str，*可选*）：
        本地数据增强词表路径。如果词替换增强策略选择"custom"，本地数据增强词表路径不能为None。默认为None。

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被替换词数量。默认为None

    aug_percent（int）：
        数据增强句子中被替换词数量占全句词比例。如果aug_n不为None，则被替换词数量为aug_n。默认为0.02。

    aug_min (int)：
        数据增强句子中被替换词数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被替换词数量最大值。默认为10。

    tf_idf (bool)：
        使用TF-IDF分数确定哪些词进行增强。默认为False。

    tf_idf_file (str，*可选*)：
        用于计算TF-IDF分数的文件。如果tf_idf为True，本地数据增强词表路径不能为None。默认为None。
```

我们接下来将以下面的例子介绍词级别替换的使用：

``` python
from paddlenlp.dataaug import WordSubstitute
s1 = "人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。"
s2 = "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"
```

**同义词替换**

根据同义词词表将句子中的词替换为同义词：
``` python
aug = WordSubstitute('synonym', create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的意义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`：

``` python
aug = WordSubstitute('synonym', create_n=3, aug_n=1)
augmenteds = aug.augment(s1)
print("origin:", s1)
for i, augmented in enumerate(augmenteds):
    print("augmented {} :".format(i), augmented)
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 0 : 人类语言是空空如也的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 1 : 人类语言是抽象的信息符号，其中蕴含着丰富的疑义信息，人类可以很轻松地理解其中的含义。
# augmented 2 : 人类语言是干瘪瘪的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
```
可以根据的实际需求，修改句子中被替换的词数量 `aug_n`：
``` python
aug = WordSubstitute('synonym', create_n=1, aug_n=3)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信符号，其中蕴含着丰富的语义消息，人类可以很轻松地懂其中的含义。
```

可以以列表的形式同时输入多个句子：
``` python
aug = WordSubstitute('synonym', create_n=1, aug_n=1)
sentences = [s1,s2]
augmenteds = aug.augment(sentences)
for sentence, augmented in zip(sentences, augmenteds):
    print("origin:", sentence)
    for i, aug in enumerate(augmented):
        print("augmented {} :".format(i), aug)
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 0 : 人类语言是肤浅的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# origin: 而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。
# augmented 0 : 而计算机只能处理数值化的信息，无法直接明人类语言，所以需要将人类语言进行数值化转换。
```
**同音词替换**

根据同音词词表将句子中的词替换为同音词：

``` python
aug = WordSubstitute('homonym', create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的语义新喜，人类可以很轻松地理解其中的含义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`和句子中被替换的词数量 `aug_n`。

**本地词表替换**

只需要传入本地词表文件路径`custom_file_path`，即可使用自定义的词表进行替换。本地词表文件为固定格式的`json`文件，字典关键字(key)为词，字典键值(item)为列表形式的替换词。例如自定义本地词表`"custom.json"`如下：
```
{"人类":["人", "人种"], "抽象":["abstract","具象"]}
```

使用自定义的本地词表进行句子中词替换:
``` python
custom_file_path = "custom.json"
aug = WordSubstitute('custom', custom_file_path=custom_file_path, create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人种语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`和句子中被替换的词数量 `aug_n`。

**组合替换**

还可以选择将同义词、同音词、本地词表进行随机组合,例如组合同义词词表核本地词表进行词替换：
``` python
custom_file_path = "custom.json"
aug = WordSubstitute(['custom','synonym'], custom_file_path=custom_file_path, create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地知晓其中的含义。
```

可以根据的实际需求，修改数据增强生成句子数 `create_n`和句子中被替换的词数量 `aug_n`。

**随机词替换**

使用随机词进行句子中词替换:
``` python
aug = WordSubstitute('random', create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的接防。
```

可以根据的实际需求，修改数据增强生成句子数 `create_n`和句子中被替换的词数量 `aug_n`。

**上下文替换**

上下文替换是随机将句子中单词进行掩码，利用中文预训练模型ERNIE 1.0，根据句子中的上下文预测被掩码的单词。相比于根据词表进行词替换，上下文替换预测出的单词更匹配句子内容，数据增强所需的时间也更长。

使用模型根据上下文预测单词进行句子中词替换:
``` python
import paddle
# 在GPU环境下运行
paddle.set_device("gpu")
# 在CPU下环境运行
# paddle.set_device("cpu")
aug = WordSubstitute('mlm', create_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息载体，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`，句子中被替换的词数量 `aug_n` **默认为1**。

**基于TF-IDF的词替换**

TF-IDF算法认为如果一个词在同一个句子中出现的次数多，词对句子的重要性就会增加；如果它在语料库中出现频率越高，它的重要性将被降低。我们将计算每个词的TF-IDF分数，**低的TF-IDF得分将有很高的概率被替换**。

我们可以在上面所有词替换策略中使用TF-IDF计算词被替换的概率，我们首先需要将`tf_idf`设为True，并传入语料库文件(包含所有训练的数据) `tf_idf_file` 用于计算单词的TF-IDF分数。语料库文件为固定 `txt` 格式，每一行为一条句子。以语料库文件`"data.txt"`做同义词替换为例，语料库文件格式如下：
``` text
人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。
...
```

``` python
tf_idf_file = "data.txt"
aug = WordSubstitute('synonym', tf_idf=True, tf_idf_file=tf_idf_file, create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的意思。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`和句子中被替换的词数量 `aug_n`。

### 词插入
词插入数据增强策略也即将句子中的词随机插入其他单词进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.WordInsert`进行词级别插入的数据增强。

```text
WordInsert 参数介绍：

    aug_type(str or list(str))：
        词插入增强策略类别。可以选择"synonym"、"homonym"、"custom"、"random"、"mlm"或者
        前三种词插入增强策略组合。

    custom_file_path (str，*可选*）：
        本地数据增强词表路径。如果词插入增强策略选择"custom"，本地数据增强词表路径不能为None。默认为None。

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被插入词数量。默认为None

    aug_percent（int）：
        数据增强句子中被插入词数量占全句词比例。如果aug_n不为None，则被插入词数量为aug_n。默认为0.02。

    aug_min (int)：
        数据增强句子中被插入词数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被插入词数量最大值。默认为10。
```

我们接下来将以下面的例子介绍词级别插入的使用：

``` python
from paddlenlp.dataaug import WordInsert
s1 = "人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。"
s2 = "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"
```

**同义词插入**

根据同义词词表将句子中的词插入为同义词：
``` python
aug = WordInsert('synonym', create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地明亮理解其中的含义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`：

``` python
aug = WordInsert('synonym', create_n=3, aug_n=1)
augmenteds = aug.augment(s1)
print("origin:", s1)
for i, augmented in enumerate(augmenteds):
    print("augmented {} :".format(i), augmented)
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 0 : 人类语言是干瘪瘪抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 1 : 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地明了理解其中的含义。
# augmented 2 : 人类语言是抽象的信息符号，其中蕴含着丰富的语义音信信息，人类可以很轻松地理解其中的含义。
```
可以根据的实际需求，修改句子中被插入的词数量 `aug_n`：
``` python
aug = WordInsert('synonym', create_n=1, aug_n=3)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的音信息符号，其中蕴含着丰富的贬义语义信息，人类可以很轻松地知理解其中的含义。
```

可以以列表的形式同时输入多个句子：
``` python
aug = WordInsert('synonym', create_n=1, aug_n=1)
sentences = [s1,s2]
augmenteds = aug.augment(sentences)
for sentence, augmented in zip(sentences, augmenteds):
    print("origin:", sentence)
    for i, aug in enumerate(augmented):
        print("augmented {} :".format(i), aug)
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 0 : 人类语言是抽象的信息符号号子，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# origin: 而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。
# augmented 0 : 而计算机只能处理数值化的音息信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。
```
**同音词插入**

根据同音词词表将句子中的词插入为同音词：

``` python
aug = WordInsert('homonym', create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息心喜符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`和句子中被插入的词数量 `aug_n`。

**本地词表插入**

只需要传入本地词表文件路径`custom_file_path`，即可使用自定义的词表进行插入。本地词表文件为固定格式的`json`文件，字典关键字(key)为词，字典键值(item)为列表形式的插入词。例如自定义本地词表`"custom.json"`如下：
```
{"人类":["人累", "扔雷"], "抽象":["丑相"]}
```

使用自定义的本地词表进行句子中词插入:
``` python
custom_file_path = "custom.json"
aug = WordInsert('custom', custom_file_path=custom_file_path, create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类人累语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`和句子中被插入的词数量 `aug_n`。

**组合插入**

还可以选择将同义词、同音词、本地词表进行随机组合,例如组合同义词词表核本地词表进行词插入：
``` python
custom_file_path = "custom.json"
aug = WordInsert(['custom','synonym'], custom_file_path=custom_file_path, create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类扔雷可以很轻松地理解其中的含义。
```

可以根据的实际需求，修改数据增强生成句子数 `create_n`和句子中被插入的词数量 `aug_n`。


**随机词插入**

使用随机词进行句子中词插入:
``` python
aug = WordInsert('random', create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类崇新语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
```

可以根据的实际需求，修改数据增强生成句子数 `create_n`和句子中被插入的词数量 `aug_n`。

**上下文插入**

上下文插入是随机将句子中单词进行掩码，利用中文预训练模型ERNIE 1.0，根据句子中的上下文预测被掩码的单词。相比于根据词表进行词插入，上下文插入预测出的单词更匹配句子内容，数据增强所需的时间也更长。

使用模型根据上下文预测单词进行句子中词插入:
``` python
import paddle
# 在GPU环境下运行
paddle.set_device("gpu")
# 在CPU下环境运行
# paddle.set_device("cpu")
aug = WordInsert('mlm', create_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻轻松松地理解其中的含义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`，句子中被插入的词数量 `aug_n` **默认为1**。

### 词删除

词删除数据增强策略也即将句子中的词随机删除进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.WordDelete`进行词级别删除的数据增强。

```text
WordDelete 参数介绍：

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被删除词数量。默认为None

    aug_percent（int）：
        数据增强句子中被删除词数量占全句词比例。如果aug_n不为None，则被删除词数量为aug_n。默认为0.02。

    aug_min (int)：
        数据增强句子中被删除词数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被删除词数量最大值。默认为10。
```

我们接下来将以下面的例子介绍词级别删除的使用：

``` python
from paddlenlp.dataaug import WordDelete
s1 = "人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。"
s2 = "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"
```

将随机删除句子中的词：
``` python
aug = WordDelete(create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以松地理解其中的含义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`：

``` python
aug = WordDelete(create_n=3, aug_n=1)
augmenteds = aug.augment(s1)
print("origin:", s1)
for i, augmented in enumerate(augmenteds):
    print("augmented {} :".format(i), augmented)
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 0 : 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，可以很轻松地理解其中的含义。
# augmented 1 : 人类是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 2 : 人类语言是抽象的符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
```

可以根据的实际需求，修改句子中被删除的词数量 `aug_n`：
``` python
aug = WordDelete(create_n=1, aug_n=3)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以松地理解其中的。
```

可以以列表的形式同时输入多个句子：
``` python
aug = WordDelete(create_n=1, aug_n=1)
sentences = [s1,s2]
augmenteds = aug.augment(sentences)
for sentence, augmented in zip(sentences, augmenteds):
    print("origin:", sentence)
    for i, aug in enumerate(augmented):
        print("augmented {} :".format(i), aug)
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 0 : 人类语言是抽象的信息符号，其中丰富的语义信息，人类可以很轻松地理解其中的含义。
# origin: 而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。
# augmented 0 : 而计算机只能处理化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。
```

### 词交换

词交换数据增强策略也即将句子中的词的位置随机交换进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.WordSwap`进行词级别交换的数据增强。

```text
WordSwap 参数介绍：

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被交换词数量。默认为None

    aug_percent（int）：
        数据增强句子中被交换词数量占全句词比例。如果aug_n不为None，则被交换词数量为aug_n。默认为0.02。

    aug_min (int)：
        数据增强句子中被交换词数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被交换词数量最大值。默认为10。
```

我们接下来将以下面的例子介绍词级别交换的使用：

``` python
from paddlenlp.dataaug import WordSwap
s1 = "人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。"
s2 = "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"
```

将随机交换句子中的词：
``` python
aug = WordSwap(create_n=1, aug_n=1)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 语言人类是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
```
可以根据的实际需求，修改数据增强生成句子数 `create_n`：

``` python
aug = WordSwap(create_n=3, aug_n=1)
augmenteds = aug.augment(s1)
print("origin:", s1)
for i, augmented in enumerate(augmenteds):
    print("augmented {} :".format(i), augmented)
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 0 : 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻理解松地其中的含义。
# augmented 1 : 人类语言是抽象的信息符号，其中蕴含着丰富的信息语义，人类可以很轻松地理解其中的含义。
# augmented 2 : 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以松地很轻理解其中的含义。
```

可以根据的实际需求，修改句子中被交换的词数量 `aug_n`：
``` python
aug = WordSwap(create_n=1, aug_n=3)
augmented = aug.augment(s1)
print("origin:", s1)
print("augmented:", augmented[0])
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented: 人类语言是抽象的信息符号，其中蕴含着丰富的信息语义，人类可以很轻松地理解其中的含义。
```

可以以列表的形式同时输入多个句子：
``` python
aug = WordSwap(create_n=1, aug_n=1)
sentences = [s1,s2]
augmenteds = aug.augment(sentences)
for sentence, augmented in zip(sentences, augmenteds):
    print("origin:", sentence)
    for i, aug in enumerate(augmented):
        print("augmented {} :".format(i), aug)
# origin: 人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# augmented 0 : 人类语言是抽象的符号信息，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。
# origin: 而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。
# augmented 0 : 而只能计算机处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。
```
