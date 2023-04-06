# Data Augmentation API

PaddleNLP提供了Data Augmentation数据增强API，可用于训练数据数据增强

**目录**
* [1. 词级别数据增强策略](#词级别数据增强策略)
    * [1.1 词替换](#词替换)
    * [1.2 词插入](#词插入)
    * [1.3 词删除](#词删除)
    * [1.4 词交换](#词交换)
* [2. 句子级别数据增强策略](#句子级别数据增强策略)
    * [2.1 同义句生成](#同义句生成)
    * [2.2 句子回译](#句子回译)
    * [2.3 句子摘要](#句子摘要)
    * [2.4 句子续写](#句子续写)
* [3. 字级别数据增强策略](#字级别数据增强策略)
    * [3.1 字替换](#字替换)
    * [3.2 字插入](#字插入)
    * [3.3 字删除](#字删除)
    * [3.4 字交换](#字交换)
* [4. 文档一键增强](#文档一键增强)


<a name="词级别数据增强策略"></a>

## 1.词级别数据增强策略

<a name="词替换"></a>

### 1.1 词替换
词替换数据增强策略也即将句子中的词随机替换为其他单词进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.WordSubstitute`进行词级别替换的数据增强。

```text
WordSubstitute 参数介绍：

    aug_type(str or list(str))：
        词替换增强策略类别。可以选择"antonym"、"embedding"、"synonym"、"homonym"、"custom"、"random"、"mlm"或者
        前四种词替换增强策略组合。

    custom_file_path (str，*可选*）：
        本地数据增强词表路径。如果词替换增强策略选择"custom"，本地数据增强词表路径不能为None。默认为None。

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被替换词数量。默认为None

    aug_percent（int）：
        数据增强句子中被替换词数量占全句词比例。如果aug_n不为None，则被替换词数量为aug_n。默认为0.1。

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
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

**同义词替换**

根据同义词词表将句子中的词替换为同义词，可以根据实际需要，设置被替换词数量占全句词比例`aug_percent`和生成增强句子数量`create_n`。`synonym`基于[中文同义词词表](https://github.com/guotong1988/chinese_dictionary)实现，`embedding`则是基于词向量（word embedding）之间的词距离构建的同义词词表确定，可以根据实际效果选择合适的词表。

``` python
aug = WordSubstitute('synonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的音符号，其中蕴含着丰富的语义信，生人可以很轻松地理解其中的含义。', '全人类语言是泛泛的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的意思。']]
augmented = aug.augment(s)
print(augmented)
# [['全人类言语是抽象的信符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。', '全人类语言是抽象的信息标记，其中蕴含着丰富的语义信息，人类可以很轻松地略知一二其中的含义。'], ['而计算机不得不处理数值化的信息，无法直接理解人类言语，所以需要将人类语言进行数值化更换。', '而计算机只能处理数值化的信息，无法直接理解人类言语，所以需要将生人语言进行数值化变换。']]
```

可以根据的实际需求，直接设置句子中被替换的词数量 `aug_n`：
``` python
aug = WordSubstitute('synonym', create_n=1, aug_n=3)
augmented = aug.augment(s[0])
print(augmented)
# [['全人类语言是空泛的信息符号，其中蕴含着丰富的涵义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的消息符号，其中蕴含着丰富的疑义信息，人类可以很轻松地理解其中的意义。'], ['而计算机唯其如此处理实测值化的信息，无法直接理解人类语言，所以需要将人类语言进行实测值化转换。']]
```

``` python
aug = WordSubstitute('embedding', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的音符号，其中蕴含着丰富的语义信，生人可以很轻松地理解其中的含义。', '全人类语言是泛泛的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的意思。']]
augmented = aug.augment(s)
print(augmented)
# [['全人类言语是抽象的信符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。', '全人类语言是抽象的信息标记，其中蕴含着丰富的语义信息，人类可以很轻松地略知一二其中的含义。'], ['而计算机不得不处理数值化的信息，无法直接理解人类言语，所以需要将人类语言进行数值化更换。', '而计算机只能处理数值化的信息，无法直接理解人类言语，所以需要将生人语言进行数值化变换。']]
```

**同音词替换**

根据同音词词表将句子中的词替换为同音词：

``` python
aug = WordSubstitute('homonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是臭香的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松德力竭其中的含义。', '任雷语言是抽象的信息富豪，其中蕴含着丰富的语义信息，任蕾可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是臭香的新潟符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含一。', '任雷语言是抽象的新潟符号，其中蕴含着丰富的语义信息，人类可以很庆松地理解其中的含义。'], ['而计算机只能处理数值化的新戏，无法直接丽姐人类语言，所以需要将人类语言进行书之化转换。', '而计算机只能处理数值化的心系，无法直接李杰人类玉烟，所以需要将人类语言进行数值化转换。']]
```

**反义词替换**

根据反义词词表将句子中的词替换为反义词：

``` python
aug = WordSubstitute('antonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是具体的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地糊涂其中的含义。', '人类语言是具体的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地懵懂其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是具体的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地糊涂其中的含义。', '人类语言是具体的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地困惑其中的含义。'], ['而计算机只能处理数值冻的信息，无法直接困惑人类语言，所以需要将人类语言进行数值冻转换。', '而计算机只能处理数值冻的信息，无法直接懵懂人类语言，所以需要将人类语言进行数值冻转换。']]
```

**本地词表替换**

只需要传入本地词表文件路径`custom_file_path`，即可使用自定义的词表进行替换。本地词表文件为固定格式的`json`文件，字典关键字(key)为词，字典键值(item)为列表形式的替换词。例如自定义本地词表`custom.json`如下：
```
{"人类":["人", "人种","全人类"], "抽象":["abstract","具象"], "轻松":["简单","容易"]}
```

使用自定义的本地词表进行句子中词替换:
``` python
custom_file_path = "custom.json"
aug = WordSubstitute('custom', custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人语言是abstract的信息符号，其中蕴含着丰富的语义信息，全人类可以很轻松地理解其中的含义。', '全人类语言是具象的信息符号，其中蕴含着丰富的语义信息，人可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人语言是abstract的信息符号，其中蕴含着丰富的语义信息，人种可以很轻松地理解其中的含义。', '人语言是具象的信息符号，其中蕴含着丰富的语义信息，人种可以很轻松地理解其中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人语言，所以需要将全人类语言进行数值化转换。', '而计算机只能处理数值化的信息，无法直接理解全人类语言，所以需要将人语言进行数值化转换。']]
```

**组合替换**

还可以选择将同义词、同音词、本地词表进行随机组合,例如组合同义词词表核本地词表进行词替换：
``` python
custom_file_path = "custom.json"
aug = WordSubstitute(['custom','synonym'], custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义音信，生人可以很轻松地领悟其中的含义。', '人种语言是抽象的信息符号，其中蕴含着丰富的贬义信息，人可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信符号，其中蕴含着丰富的语义消息，生人可以很轻松地理解其中的含义。', '人语言是抽象的信息符号，其中蕴含着丰富的语义消息，人类可以很轻松地亮堂其中的含义。'], ['而计算机只能处理数值变成的信息，无法直接理解人类语言，所以需要将生人语言进行数值变为转换。', '而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类言语进行标注值变为转换。']]
```

**随机词替换**

使用随机词进行句子中词替换:
``` python
aug = WordSubstitute('random', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类塘屿是抽象的黄芪酒符号，其中蕴含着丰富的语义信息，人类可以很轻单官理解其中的含义。', '人类语言是抽象的亞符号，其中蕴含着丰富的语义镇咳药，人类可以いていた松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类共进退是抽象的信息符号，其中蕴含着丰富的梦界信息，人类可以很轻大凤理解其中的含义。', '人类语言是4490的信息符号，其中蕴含着丰富的语义信息，科摩可以很轻松地崔磊其中的含义。'], ['而库山乡只能处理数值化的信息，无法直接理解街亭失守MicrosoftWorks，所以需要将人类语言进行数值化转换。', '而0.57万只能处理数值化的信息，无法直接理解人类语言，所以需要将穆哥叶楚进行数值化转换。']]
```

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
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的信义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的语字符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。'], ['而计算机只能处理数值化的信言，无法直接理解人类语言，所以需要将人类语言进行数值化转换。']]
```
句子中被替换的词数量目前只支持 `aug_n` 为1。

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
aug = WordSubstitute('synonym', tf_idf=True, tf_idf_file=tf_idf_file, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的消息符号，其中蕴含着丰富的语义音信，人类可以很轻松地敞亮其中的含义。', '生人语言是抽象的消息符号，其中蕴含着丰富的语义信息，全人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类言语是抽象的信息符号，其中蕴含着丰富的语义信息，生人可以很轻松地分晓其中的含义。', '人类言语是抽象的音问符号，其中蕴含着丰富的语义信息，全人类可以很轻松地理解其中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人类言语，所以需要将全人类言语进行数值化转换。', '而计算机只能处理数值化的信息，无法直接理解生人语言，所以需要将全人类语言进行数值化变换。']]
```


### 词插入
词插入数据增强策略也即将句子中的词随机插入其他单词进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.WordInsert`进行词级别插入的数据增强。

```text
WordInsert 参数介绍：

    aug_type(str or list(str))：
        词插入增强策略类别。可以选择"antonym"、"embedding"、"synonym"、"homonym"、"custom"、"random"、"mlm"或者
        前三种词插入增强策略组合。

    custom_file_path (str，*可选*）：
        本地数据增强词表路径。如果词插入增强策略选择"custom"，本地数据增强词表路径不能为None。默认为None。

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被插入词数量。默认为None

    aug_percent（int）：
        数据增强句子中被插入词数量占全句词比例。如果aug_n不为None，则被插入词数量为aug_n。默认为0.1。

    aug_min (int)：
        数据增强句子中被插入词数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被插入词数量最大值。默认为10。
```

我们接下来将以下面的例子介绍词级别插入的使用：

``` python
from paddlenlp.dataaug import WordInsert
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

**同义词插入**
根据同义词词表将句子中的词前/后插入同义词，可以根据实际需要，设置插入词数量占全句词比例`aug_percent`和生成增强句子数量`create_n`。`synonym`基于[中文同义词词表](https://github.com/guotong1988/chinese_dictionary)实现，`embedding`则是基于词向量（word embedding）之间的词距离构建的同义词词表确定，可以根据实际效果选择合适的词表。

``` python
aug = WordInsert('synonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['全人类人类语言是华而不实抽象的信息符号，其中蕴含着丰富的语义消息信息，人类可以很轻松地理解其中的含义。', '人类语言是抽象的音信信息符号，其中蕴含着丰富的语义消息信息，生人人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言言语是抽象的信息符号，其中蕴含着丰富的语义褒义信息音问，人类可以很轻松地理解其中的含义。', '人类语言是抽象言之无物的信息符号记号，其中蕴含着丰富的语义信息，人类可以很轻松地理解清楚其中的含义。'], ['而计算机只能只得处理数值化变为的信息，无法直接理解人类生人语言，所以需要将人类语言进行数值化转换。', '而计算机只能处理数值分值化化为的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换变换。']]
```

可以根据的实际需求，直接设置句子中被替换的词数量 `aug_n`：
``` python
aug = WordInsert('synonym', create_n=1, aug_n=3)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言自然语言是抽象的信息符号，其中蕴含着蕴含丰富的语义信息数据，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象具象的信息符号，其中蕴含着丰富的语义演算信息，人类人类文明可以很轻松地理解其中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类全人类语言进行数值最大值化转换切换。']]
```

``` python
aug = WordInsert('embedding', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的音符号，其中蕴含着丰富的语义信，生人可以很轻松地理解其中的含义。', '全人类语言是泛泛的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的意思。']]
augmented = aug.augment(s)
print(augmented)
# [['全人类言语是抽象的信符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。', '全人类语言是抽象的信息标记，其中蕴含着丰富的语义信息，人类可以很轻松地略知一二其中的含义。'], ['而计算机不得不处理数值化的信息，无法直接理解人类言语，所以需要将人类语言进行数值化更换。', '而计算机只能处理数值化的信息，无法直接理解人类言语，所以需要将生人语言进行数值化变换。']]
```

**同音词插入**

根据同音词词表将句子中的词插入为同音词：

``` python
aug = WordInsert('homonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言雨燕是抽象的信息符号，其中蕴含着丰富的语义信息，人类任雷可以很轻松地理解其中的含义寒意。', '人泪人类语言是丑像抽象的心细信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象筹饷的信息符号，其中蕴含着丰富的语义信息，人类可以很轻恨情松地理解力竭其中的含义。', '人类语言是抽象臭香的信息新戏符号，其中蕴含着丰富的语义信息，人类可以很轻很庆松地理解其中的含义。'], ['而计算机只能纸能处理数值化的信息新西，无法直接理解李杰人类语言，所以需要将人类语言进行数值化转换。', '而计算机只能处理数值化的信息，无法直接理解人类语言语嫣，所以需要将人类语言语嫣进行数值书之化转换。']]
```

**反义词插入**

根据反义词词表将句子中的词前/后插入反义词：

``` python
aug = WordInsert('antonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象具体的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解懵懂其中的含义。', '人类语言是具体抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地懵懂理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象具体的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解懵懂其中的含义。', '人类语言是具体抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地困惑理解其中的含义。'], ['而计算机只能处理数值化凝的信息，无法直接理解困惑人类语言，所以需要将人类语言进行数值化冻转换。', '而计算机只能处理数值化凝的信息，无法直接理解懵懂人类语言，所以需要将人类语言进行数值化冻转换。']]
```

**本地词表插入**

只需要传入本地词表文件路径`custom_file_path`，即可使用自定义的词表进行插入。本地词表文件为固定格式的`json`文件，字典关键字(key)为词，字典键值(item)为列表形式的插入词。例如自定义本地词表`custom.json`如下：
```
{"人类":["人累", "扔雷"], "抽象":["丑相"], "符号":["富豪","负号","付豪"]}
```

使用自定义的本地词表进行句子中词插入:
``` python
custom_file_path = "custom.json"
aug = WordInsert('custom', custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类扔雷语言是抽象的信息符号富豪，其中蕴含着丰富的语义信息，人类扔雷可以很轻松地理解其中的含义。', '人类扔雷语言是抽象丑相的信息符号负号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['扔雷人类语言是丑相抽象的信息付豪符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。', '人类扔雷语言是抽象丑相的信息符号，其中蕴含着丰富的语义信息，人类人累可以很轻松地理解其中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人类人累语言，所以需要将人类扔雷语言进行数值化转换。', '而计算机只能处理数值化的信息，无法直接理解人类扔雷语言，所以需要将人类人累语言进行数值化转换。']]
```


**组合插入**

还可以选择将同义词、同音词、本地词表进行随机组合,例如组合同义词词表核本地词表进行词插入：
``` python
custom_file_path = "custom.json"
aug = WordInsert(['custom','synonym'], custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言词汇是抽象的信息数据符号，其中蕴含着蕴含丰富的语义信息，人类可以很轻松地理解其中的含义。', '人类语言是丑相抽象的信息符号，其中蕴含蕴含着丰富的语义信息，人类可以很轻松地理解其中的含意含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含蕴含着丰富的语义数据信息，人类可以很轻松地理解其中的涵义含义。', '人类人累语言语法是抽象的信息符号，其中蕴含着丰富的语义信息数据，人类可以很轻松地理解其中的含义。'], ['而计算机计算机系统只能处理数值值化的信息，无法直接理解人类人累语言，所以需要将人类语言进行数值化转换。', '而计算机只能处理数值计算结果化的信息，无法直接理解人类语言，所以需要将人类人类文明语言进行数值化转换变换。']]
```

**随机词插入**

使用随机词进行句子中词插入:
``` python
aug = WordInsert('random', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['郎氏人类语言是抽象的魏略信息符号，其中晓畅蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。', 'seeddestiny人类语言是抽象的那一双信息符号，其中九王坟蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类文胸语言是抽象解放日报的信息符号鸭池，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。', '堤平人类语言是文学作家抽象的信息中越关系符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。'], ['而勤业计算机只能处理数值HIStory化的信息，无法直接理解人类语言，所以需要将唐本佑人类语言进行数值化转换。', '而计算机刀弓只能处理数值化苏雨琪的信息，无法直接理解人类语言，所以需要将人类平达语言进行数值化转换。']]
```


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
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义语化信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息符号系统，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。'], ['而计算机只能直接处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。']]
```
句子中插入的词数量目前只支持 `aug_n` 为1。

### 词删除

词删除数据增强策略也即将句子中的词随机删除进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.WordDelete`进行词级别删除的数据增强。

```text
WordDelete 参数介绍：

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被删除词数量。默认为None

    aug_percent（int）：
        数据增强句子中被删除词数量占全句词比例。如果aug_n不为None，则被删除词数量为aug_n。默认为0.1。

    aug_min (int)：
        数据增强句子中被删除词数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被删除词数量最大值。默认为10。
```

我们接下来将以下面的例子介绍词级别删除的使用：

``` python
from paddlenlp.dataaug import WordDelete
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

将随机删除句子中的词：
``` python
aug = WordDelete(create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的。', '人类语言是抽象的信息符号，其中蕴含着丰富的语义，人类可以松地其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是的信息符号，其中丰富的语义，人类可以很轻松地理解其中的含义。', '人类语言是的信息，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的。'], ['而计算机只能处理数值化的信息，无法直接理解语言，所以需要将人类语言进行转换。', '而计算机处理数值化的信息，无法直接人类语言，所以需要将人类语言进行数值化。']]
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
        数据增强句子中被交换词数量占全句词比例。如果aug_n不为None，则被交换词数量为aug_n。默认为0.1。

    aug_min (int)：
        数据增强句子中被交换词数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被交换词数量最大值。默认为10。
```

我们接下来将以下面的例子介绍词级别交换的使用：

``` python
from paddlenlp.dataaug import WordSwap
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

将随机交换句子中的词：
``` python
aug = WordSwap(create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的符号信息，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以松地很轻理解其中的含义。'], ['而计算机只能处理化数值的信息，无法直接理解人类语言，所以需要将人类语言进行数值转换化。']]
```

<a name="句子级别数据增强策略"></a>

## 2. 句子级别数据增强策略

<a name="同义句生成"></a>

### 2.1 同义句生成

同义句生成数据增强策略也即根据输入句子生成相似句，模型首先生成`generate_n`个句子，然后再利用模型筛选出最佳的`create_n`。这里我们将介绍如何使用`paddlenlp.dataaug.SentenceGenerate`进行同义句生成的数据增强。

```text
SentenceGenerate 参数介绍：

    model_name (str)：
        生成同义句模型名，可选"roformer-chinese-sim-char-ft-base"， "roformer-chinese-sim-char-base"，"roformer-chinese-sim-char-ft-small"，"roformer-chinese-sim-char-small"。默认为"roformer-chinese-sim-char-base"。

    create_n（int）：
        数据增强句子数量，从生成相似句中筛选最佳的句子数量。默认为1。

    generate_n（int）：
        模型生成相似句数量。默认为5。

    max_length（int）：
        模型生成相似句最长长度。默认为128。

    top_p (float)：
        “sampling”策略中top-p-filtering的累积概率。该值应满足：math:`0<=top_p<1`。默认为0.95
```

我们接下来将以下面的例子介绍同义句生成的使用：

``` python
from paddlenlp.dataaug import SentenceGenerate
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

``` python
import paddle
# 建议在在GPU环境下运行
paddle.set_device("gpu")
# 在CPU下环境运行
# paddle.set_device("cpu")
aug = SentenceGenerate(create_n=2, generate_n=5, max_length=128, top_p=0.95)
augmented = aug.augment(s[0])
print(augmented)
# ['人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义', '人类语言是一个抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。', '人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义答。'], ['而计算机只能处理数值化的信息，无法直接理解人类语言，故需要将人类语言进行数值化转换。', '2、计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。']]
```

<a name="句子回译"></a>

### 2.2 句子回译

句子回译数据增强策略也即将输入的句子翻译为另一种语言，然后再翻译回来，生成语义相同表达方式不同的句子，用于数据增强。这里我们将介绍如何使用基于百度翻译API`paddlenlp.dataaug.SentenceBackTranslateAPI`进行句子回译的数据增强和基于模型的`paddlenlp.dataaug.SentenceBackTranslate`。


```text
SentenceBackTranslateAPI 参数介绍：

    src_lang (str)：
        输入句子的语言。默认为"zh"。

    tgt_lang（str）：
        目标句子的语言，增强策略将会把句子翻译为目标句子语言，再翻译回输入句子语言。默认为"en"。

    appid（str）：
        百度通用翻译API的APPID（如果你使用自己的百度翻译API服务appid/secretKey）。默认为None。

    secretKey (str)：
        百度通用翻译API的密钥（如果你使用自己的百度翻译API服务appid/secretKey）。默认为1。

    qps (int)：
        百度通用翻译API的QPS（如果你使用自己的百度翻译API服务appid/secretKey）。 默认为1。
```

我们接下来将以下面的例子介绍基于百度翻译API的句子回译的使用：

使用SentenceBackTranslateAPI需要安装PaddleHub
```shell
pip install paddlehub==2.3.1
```

``` python
from paddlenlp.dataaug import SentenceBackTranslateAPI
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

``` python
aug = SentenceBackTranslateAPI(src_lang='zh', tgt_lang='en')
augmented = aug.augment(s[0])
print(augmented)
# ['人类语言是一种抽象的信息符号，蕴含着丰富的语义信息。人类很容易理解它的含义。']
augmented = aug.augment(s)
print(augmented)
# ['人类语言是一种抽象的信息符号，蕴含着丰富的语义信息。人类很容易理解它的含义。', '然而，计算机只能处理数字信息，不能直接理解人类语言，因此有必要将人类语言转换为数字信息。']
```
**Note**
1. 默认使用PaddleHub提供的百度翻译API服务，也可以选择注册自己的百度翻译API服务账号获取相应的AppID和密钥，账号注册流程请参见[百度翻译API文档](https://fanyi-api.baidu.com/doc/21)，使用自己AppID和密钥则无需安装PaddleHub。
2. `src_lang`和`tgt_lang`支持的语言和服务异常报错详见[百度翻译API文档](https://fanyi-api.baidu.com/doc/21)中完整语种列表和错误码列表。

```text
SentenceBackTranslate 参数介绍：

    src_lang (str)：
        输入句子的语言。默认为"zh"。可选语言:'ar', 'cs', 'de', 'en', 'es', 'et', 'fi', 'fr', 'gu', 'hi', 'it', 'ja', 'kk', 'ko', 'lt', 'lv', 'my', 'ne', 'nl', 'ro', 'ru', 'si', 'tr', 'vi', 'zh', 'af', 'az', 'bn', 'fa', 'he', 'hr', 'id', 'ka', 'km', 'mk', 'ml', 'mn', 'mr', 'pl', 'ps', 'pt', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'uk', 'ur', 'xh', 'gl', 'sl'。

    tgt_lang（str）：
        目标句子的语言，增强策略将会把句子翻译为目标句子语言，再翻译回输入句子语言。默认为"en"。可选语言:'ar', 'cs', 'de', 'en', 'es', 'et', 'fi', 'fr', 'gu', 'hi', 'it', 'ja', 'kk', 'ko', 'lt', 'lv', 'my', 'ne', 'nl', 'ro', 'ru', 'si', 'tr', 'vi', 'zh', 'af', 'az', 'bn', 'fa', 'he', 'hr', 'id', 'ka', 'km', 'mk', 'ml', 'mn', 'mr', 'pl', 'ps', 'pt', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'uk', 'ur', 'xh', 'gl', 'sl'。

    max_length（int）：
        模型生成相似句最长长度。默认为128。

    batch_size (int)：
        批大小，如果显存不足，适当调小该值。默认为1。

    num_beams (int)：
        “beam_search”策略中的beam值。 默认为 4。

    use_faster (bool)：
        是否使用FasterGeneration进行加速。默认为False。

    decode_strategy (str)：
        生成中的解码策略。 目前支持三种解码策略：“greedy_search”、“sampling”和“beam_search”。 默认为“beam_search”。

```

我们接下来将以下面的例子介绍基于模型的句子回译的使用：

``` python
from paddlenlp.dataaug import SentenceBackTranslate
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

``` python
import paddle
# 建议在在GPU环境下运行
paddle.set_device("gpu")
# 在CPU下环境运行
# paddle.set_device("cpu")
aug = SentenceBackTranslate(src_lang='zh', tgt_lang='en', batch_size=1, max_length=128)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象信息符号, 它包含丰富的语义信息, 可以容易理解.']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象信息符号, 它包含丰富的语义信息, 可以容易理解.'], ['计算机只能处理数字化信息,不能直接理解人类语言,因此有必要进行数字化。']]
```
**Note**
1. 如果`use_faster`设为True，第一次执行PaddleNLP会启动即时编译（JIT Compile）自动编译高性能解码算子。编译过程通常会花费几分钟的时间编译只会进行一次，之后再次使用高性能解码就不需要重新编译了，编译完成后会继续运行。

<a name="句子摘要"></a>

### 2.3 句子摘要

句子摘要数据增强策略也即对输入句子生成摘要句子，这里我们将介绍如何使用`paddlenlp.dataaug.SentenceSummarize`进行句子摘要的数据增强。

```text
SentenceSummarize 参数介绍：

    create_n（int）：
        数据增强句子数量，从生成相似句中筛选最佳的句子数量。默认为1。

    max_length（int）：
        模型生成相似句最长长度。默认为128。

    batch_size (int)：
        批大小，如果显存不足，适当调小该值。默认为1。

    top_k (int)：
        “sampling”策略中top-k-filtering的最高概率token的数量, 0表示没有影响。默认为5。

    top_p (float)：
        “sampling”策略中top-p-filtering的累积概率。该值应满足：math:`0<=top_p<1`。默认为1.0，表示没有影响。

    temperature (float)：
        “sampling”策略中对下一个token概率进行建模的值。 默认为 1.0，表示没有影响。

    use_fp16_decoding (bool)：
        是否使用fp16进行加速。默认为False。
```

我们接下来将以下面的例子介绍句子摘要的使用：

``` python
from paddlenlp.dataaug import SentenceSummarize
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

``` python
import paddle
# 建议在在GPU环境下运行
paddle.set_device("gpu")
# 在CPU下环境运行
# paddle.set_device("cpu")
aug = SentenceSummarize(create_n=2, batch_size=1, max_length=128)
augmented = aug.augment(s[0])
print(augmented)
# [['什么是人类语言？', '为什么说人类语言是抽象的信息符号？']]
augmented = aug.augment(s)
print(augmented)
# [['什么是人类语言？', '为什么说人类语言是抽象的信息符号？'], ['计算机只能处理数值化的信息(图)', '计算机只能处理数值化的信息']]
```

<a name="句子续写"></a>

### 2.4 句子续写

句子续写数据增强策略也即对输入句子进行句子续写，这里我们将介绍如何使用`paddlenlp.dataaug.SentenceContinue`进行句子续写的数据增强。

```text
SentenceContinue 参数介绍：

    model_name (str)：
        生成同义句模型名，可选"gpt-cpm-large-cn", "gpt-cpm-small-cn-distill"。默认为"gpt-cpm-small-cn-distill"。

    max_length（int）：
        模型生成相似句最长长度。默认为128。

    decode_strategy (str)：
        生成中的解码策略。 目前支持三种解码策略：“greedy_search”、“sampling”和“beam_search”。 默认为“beam_search”。

    use_faster (bool)：
        是否使用FasterGeneration进行加速。默认为False。

    create_n（int）：
        数据增强句子数量，从生成相似句中筛选最佳的句子数量。默认为1。

    top_k (int)：
        “sampling”策略中top-k-filtering的最高概率token的数量, 0表示没有影响。默认为5。

    top_p (float)：
        “sampling”策略中top-p-filtering的累积概率。该值应满足：math:`0<=top_p<1`。默认为1.0，表示没有影响。

    temperature (float)：
        “sampling”策略中对下一个token概率进行建模的值。 默认为 1.0，表示没有影响。

    batch_size (int)：
        批大小，如果显存不足，适当调小该值。默认为1。
```

我们接下来将以下面的例子介绍同义句生成的使用：

``` python
from paddlenlp.dataaug import SentenceContinue
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

``` python
import paddle
# 建议在在GPU环境下运行
paddle.set_device("gpu")
# 在CPU下环境运行
# paddle.set_device("cpu")
aug = SentenceContinue(create_n=2, batch_size=1, max_length=64)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。然而语言本身的抽象不是简单的,语言的复杂性以及语言的抽象化则是人类认识世界的另一个重要途径。信息本身和人类的理解能力无关,人类理解世界的过程就是信息过程的不断丰富与不断', '人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。不过,这也是很不容易的。有一些事情是不可能实现的,对于一些人来说,不可能实现的事情只是遥不可及的梦,这也就是为什么在他们的思想中经常会']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。那么,为什么会出现这种现象呢?首先,我们知道人类拥有最简单的语言,但是我们无法通过语言去直接理解它,这就使得我们需要建立数学模型,使得理解过程比语言模型复杂得多', '人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。如果人类可以用语言解决语言问题,那么这个问题是不能回避的。这就是为什么计算机是一个语言的存在,因为它能够处理语言的逻辑关系。这就要求我们对语言的基本事实和各种各样的信息进行细致'], ['而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。因此,计算机在编程方面的功能就是将程序的数据进行算法处理,以便在特定情况下做出特定的功能。在这里可以看到,计算机编程的主要功能是处理文字的信息,而与文字的信息无关的', '而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。因此,“语言”这个词的含义,实际上可以由下面这个公式来表示:=\\alpha\\left(\\alpha-(\\alpha']]
```
<a name="字级别数据增强策略"></a>

## 3.字级别数据增强策略

<a name="字替换"></a>

### 3.1 字替换
字替换数据增强策略也即将句子中的字随机替换为其他单字进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.CharSubstitute`进行字级别替换的数据增强。

```text
CharSubstitute 参数介绍：

    aug_type(str or list(str))：
        字替换增强策略类别。可以选择"antonym"、"homonym"、"custom"、"random"、"mlm"或者
        前三种字替换增强策略组合。

    custom_file_path (str，*可选*）：
        本地数据增强字表路径。如果字替换增强策略选择"custom"，本地数据增强字表路径不能为None。默认为None。

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被替换字数量。默认为None

    aug_percent（int）：
        数据增强句子中被替换字数量占全句字比例。如果aug_n不为None，则被替换字数量为aug_n。默认为0.1。

    aug_min (int)：
        数据增强句子中被替换字数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被替换字数量最大值。默认为10。
```

我们接下来将以下面的例子介绍字级别替换的使用：

``` python
from paddlenlp.dataaug import CharSubstitute
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

**同音字替换**

根据同音字表将句子中的字替换为同音字，可以根据实际需要，设置被替换字数量占全句字比例`aug_percent`和生成增强句子数量`create_n`。

``` python
aug = CharSubstitute('homonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是筹象的信汐符号，其中蕴含着逢富的语义锌息，人类可以很轻诵地理解其中的含义。', '人类语嫣是抽象的信息符号，其中蕴含着丰富的语义信息，人垒可以很情松地理婕其种的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的辛息符豪，其中匀含着丰富的语义信息，人类可以很庆耸地理解其中的含义。', '人磊语晏是抽象的新息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理劫其种的含义。'], ['而叽算机只能处理数值化的信息，无法直接理解人蕾语堰，所以需要将人类语演进行数值化专换。', '而疾算机只能杵理数值华的信息，无法直接理捷人类语验，所以需要将人类语言进行数值化转换。']]
```

可以根据的实际需求，直接设置句子中被替换的字数量 `aug_n`：
``` python
aug = CharSubstitute('homonym', create_n=1, aug_n=3)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的裕义信息，人类可以很轻送地理解其中的含漪。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息符号，其中陨含着丰富的语义信息，人蕾可以很轻松地理解其种的含义。'], ['而计算机只能处理数值化的心息，无罚直接理解人类语言，所以需要将人类煜言进行数值化转换。']]
```

**反义字替换**

根据反义字字表将句子中的字替换为反义字：

``` python
aug = CharSubstitute('antonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰穷的语义信息，人类可以很轻紧地理结其西的露义。', '人类语言是抽象的疑息符号，其西蕴含着歉富的语义信息，人类可以很轻松地理结其中的露义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的疑作符号，其洋蕴含着丰贫的语义信息，人类可以很轻松地理解其中的露义。', '人类语言是抽象的信息符号，其洋蕴含着歉贫的语义信息，人类可以很轻紧地理系其中的含义。'], ['而计算机只能处理数值凝的疑作，无法曲接理扎人类语言，所以需要将人类语言进行数值化转换。', '而计算机只能处理数值化的信作，无法屈接理结人类语言，所以需要将人类语言退行数值凝转换。']]
```

**本地字表替换**

只需要传入本地字表文件路径`custom_file_path`，即可使用自定义的字表进行替换。本地字表文件为固定格式的`json`文件，字典关键字(key)为字，字典键值(item)为列表形式的替换字。例如自定义本地字表`custom.json`如下：
```
{"人":["任", "认","忍"], "抽":["丑","臭"], "轻":["亲","秦"],"数":["书","树"],"转":["赚","专"],"理":["里","例"]}
```

使用自定义的本地字表进行句子中字替换:
``` python
custom_file_path = "custom.json"
aug = CharSubstitute('custom', custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是丑象的信息符号，其中蕴含着丰富的语义信息，人类可以很秦松地理解其中的含义。', '人类语言是臭象的信息符号，其中蕴含着丰富的语义信息，人类可以很秦松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是丑象的信息符号，其中蕴含着丰富的语义信息，人类可以很秦松地例解其中的含义。', '人类语言是臭象的信息符号，其中蕴含着丰富的语义信息，人类可以很秦松地里解其中的含义。'], ['而计算机只能处例书值化的信息，无法直接里解人类语言，所以需要将人类语言进行书值化专换。', '而计算机只能处里书值化的信息，无法直接例解人类语言，所以需要将人类语言进行树值化赚换。']]
```

**组合替换**

还可以选择将同音字、本地字表进行随机组合,例如组合同音字表和本地字表进行字替换：
``` python
custom_file_path = "custom.json"
aug = CharSubstitute(['custom','homonym'], custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信囍符号，其中蕴含着丰斧的遇倚信息，人类可以很轻颂地理解其中的含义。', '人类语言是抽乡的信吸符好，其终蕴含着丰富的语义芯息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类於言是抽想的信息肤号，其中蕴含着丰腐的语义信息，人类可以很轻松地理解其中的含诣。', '人类语言是抽项的信息符号，其中蕴憨着丰富的娱义信息，人类可以很请怂地理解其中的含义。'], ['而计算机只能处理数值划的信羲，无法直接理解人类钰言，所以墟要将人类语闫进行数值化转换。', '而计算羁只能处理数值化的信熙，无法直介理解人类语岩，所以需要将人类语焰进行数值化转换。']]
```

**随机字替换**

使用随机字进行句子中字替换:
``` python
aug = CharSubstitute('random', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人开自言是抽象的信息符号，其中蕴正着丰富的语义信息，人类可以很拜松地理解其中的含侯。', '人类语言是抽象的许息符号，其世蕴银着丰B的语义莘息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类吧言是抽象的信息符号，其中蕴含着丰富的萎义桅小，人类可以很轻松地理解其中的后义。', '人类语言是河象的信夹符号，其中蕴含着丰刘的语义信息，人类可以很轻李地理解其中的含阿。'], ['而庙算机只能处葛数弘化的信息，无法直接理解人类语拉，所以需要将人吴语言进行数值化转换。', '而ｎ算机只能处理数值化的信息，无法直接理解人红语言，所以需要将人类语言进行林值查转P。']]
```

**上下文替换**

上下文替换是随机将句子中单字进行掩码，利用中文预训练模型ERNIE 3.0，根据句子中的上下文预测被掩码的单字。相比于根据字表进行字替换，上下文替换预测出的单字更匹配句子内容，数据增强所需的时间也更长。

使用模型根据上下文预测单字进行句子中字替换:
``` python
import paddle
# 在GPU环境下运行
paddle.set_device("gpu")
# 在CPU下环境运行
# paddle.set_device("cpu")
aug = CharSubstitute('mlm', create_n=1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息符号，其中包含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
```
句子中被替换的字数量目前只支持 `aug_n` 为1。



### 字插入
字插入数据增强策略也即将句子中的字随机插入其他单字进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.CharInsert`进行字级别插入的数据增强。

```text
CharInsert 参数介绍：

    aug_type(str or list(str))：
        字插入增强策略类别。可以选择"antonym"、"homonym"、"custom"、"random"、"mlm"或者
        前三种字插入增强策略组合。

    custom_file_path (str，*可选*）：
        本地数据增强字表路径。如果字插入增强策略选择"custom"，本地数据增强字表路径不能为None。默认为None。

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被插入字数量。默认为None

    aug_percent（int）：
        数据增强句子中被插入字数量占全句字比例。如果aug_n不为None，则被插入字数量为aug_n。默认为0.1。

    aug_min (int)：
        数据增强句子中被插入字数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被插入字数量最大值。默认为10。
```

我们接下来将以下面的例子介绍字级别插入的使用：

``` python
from paddlenlp.dataaug import CharInsert
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

**同音字插入**
根据同音字表将句子中的字前/后插入同音字，可以根据实际需要，设置插入字数量占全句字比例`aug_percent`和生成增强句子数量`create_n`。

``` python
aug = CharInsert('homonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语寓言咽是抽象的信息符复号，其中蕴韵含着丰富夫的语义信息，人类可以很轻松地理解其中的含义。', '人镭类语岩言是抽想象的信息符号，其忠中蕴含着疯丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类勒语言是抽象想的信息符号，其中蕴含着丰富的语誉义以信息，人类可以很轻卿松地理解其中的含义。', '人泪类语言是抽象的芯信息符号，其中蕴含着枫丰富的语疑义锌信息，人类可以很轻松地理解其中的含义。'], ['而计算机只能处理数植值化的新信息，无法直接狸理解人类语言，所以需要将人类峪语言进行书数值化转换。', '而计算机只能处理梳数值化的新信息，无法直接笠理解人类语言，所以需要将人类语衍言进行数值化赚转换。']]
```

可以根据的实际需求，直接设置句子中被替换的字数量 `aug_n`：
``` python
aug = CharInsert('homonym', create_n=1, aug_n=3)
augmented = aug.augment(s[0])
print(augmented)
# [['人类勒语言是抽象的信息符号，其中蕴含着丰缝富的语义信息，人类可以很轻松颂地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义新信息，人类可以很轻松地荔理解其终中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人类语言，所以序需要将人類类语言进刑行数值化转换。']]
```


**本地字表插入**

只需要传入本地字表文件路径`custom_file_path`，即可使用自定义的字表进行插入。本地字表文件为固定格式的`json`文件，字典关键字(key)为字，字典键值(item)为列表形式的插入字。例如自定义本地字表`custom.json`如下：
```
{"人":["任", "认","忍"], "抽":["丑","臭"], "轻":["亲","秦"],"数":["书","树"],"转":["赚","专"],"理":["里","例"]}
```

使用自定义的本地字表进行句子中字插入:
``` python
custom_file_path = "custom.json"
aug = CharInsert('custom', custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是臭抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很亲轻松地里理解其中的含义。', '人类语言是抽臭象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻秦松地理里解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是丑抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很秦轻松地例理解其中的含义。', '人类语言是丑抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很亲轻松地例理解其中的含义。'], ['而计算机只能处理例数树值化的信息，无法直接理例解人类语言，所以需要将人类语言进行数树值化转专换。', '而计算机只能处里理树数值化的信息，无法直接例理解人类语言，所以需要将人类语言进行书数值化赚转换。']]
```
**反义字插入**

根据反义字字表将句子中的字前/后插入反义字：

``` python
aug = CharInsert('antonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的疑信作息符号，其中蕴露含着丰富的语义信息，人类可以很轻紧松地理扎解其中的含义。', '人类语言是抽象的信疑息符号，其中洋蕴含着丰富穷的语义信息作，人类可以很轻松地理解其中的含露义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信作息符号，其中蕴含着丰富的语义信作息，人类可以很轻紧松地理系解其中的露含义。', '人类语言是抽象的信疑息符号，其中洋蕴含露着丰富的语义信息作，人类可以很轻松地理解扎其中的含义。'], ['而计算机只能处理数值凝化的信作息，无法屈直接理解人类语言，所以需要将人类语言进止行数值化停转换。', '而计算机只能处理数值化凝的信疑息，无法直接递理解系人类语言，所以需要将人类语言进行数值化凝转换。']]
```

**组合插入**

还可以选择将同音字、同音字、本地字表进行随机组合,例如组合同音字表核本地字表进行字插入：
``` python
custom_file_path = "custom.json"
aug = CharInsert(['custom','homonym'], custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类镭语言是抽象的信鑫息夕符号壕，其中蕴含着丰富的语义信息，人类可以很轻晴松地理解其中的含义。', '人类语咽言是抽翔象的信息覆符号，其中蕴含着丰腐富的语义信息，人类可以很轻松地離理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是稠抽象的芯信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理桀解其重中的含裔义。', '人类语言是抽象的信息囍符号壕，其中蕴含着丰富孵的语义信息奚，人类可以很轻卿松地理解其中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人类语言阎，所以需要将人类语言衍进金行数值化哗转专换。', '而计纪算机只能处岀理隶数值化的信息，无法直接理解人类语言，所以需要将人类雷语言进行数值芷化转换。']]
```

**随机字插入**

使用随机字进行句子中字插入:
``` python
aug = CharInsert('random', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类S语言是抽象的信息符号，其中蕴含着丰富的语义信息，人鞋类可以很轻J松地张理解其中的含陈义。', '人类谷语言是抽象的信息符号，其中蕴含着丰富的语义信息，人烘类可以很轻割松地灵理解其中的含异义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语创言是抽象的信好息符号，其中蕴王含着丰富的语义信如息，人类可以很轻松地理解其中的丹含义。', '人类语F言是抽象的信M息符号，其中蕴史含着丰富的语义信伊息，人类可以很轻松地理解其中的秀含义。'], ['而计算机只能处楚理数值化O的信息，无法直接理解人类语丁言，所以需P要将人类语言进行甲数值化转换。', '而计算机只能处漫理数值化翁的信息，无法直接理解人类语奚言，所以需中要将人类语言进行黄数值化转换。']]
```


**上下文插入**

上下文插入是随机将句子中单字进行掩码，利用中文预训练模型ERNIE 3.0，根据句子中的上下文预测被掩码的单字。相比于根据字表进行字插入，上下文插入预测出的单字更匹配句子内容，数据增强所需的时间也更长。

使用模型根据上下文预测单字进行句子中字插入:
``` python
import paddle
# 在GPU环境下运行
paddle.set_device("gpu")
# 在CPU下环境运行
# paddle.set_device("cpu")
aug = CharInsert('mlm', create_n=1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值转化转换。']]
```
句子中插入的字数量目前只支持 `aug_n` 为1。

### 字删除

字删除数据增强策略也即将句子中的字随机删除进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.CharDelete`进行字级别删除的数据增强。

```text
CharDelete 参数介绍：

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被删除字数量。默认为None

    aug_percent（int）：
        数据增强句子中被删除字数量占全句字比例。如果aug_n不为None，则被删除字数量为aug_n。默认为0.1。

    aug_min (int)：
        数据增强句子中被删除字数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被删除字数量最大值。默认为10。
```

我们接下来将以下面的例子介绍字级别删除的使用：

``` python
from paddlenlp.dataaug import CharDelete
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

将随机删除句子中的字：
``` python
aug = CharDelete(create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的。', '人类语言是抽象的信息符号，其中蕴含着丰富的语义，人类可以松地其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是的信息符号，其中丰富的语义，人类可以很轻松地理解其中的含义。', '人类语言是的信息，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的。'], ['而计算机只能处理数值化的信息，无法直接理解语言，所以需要将人类语言进行转换。', '而计算机处理数值化的信息，无法直接人类语言，所以需要将人类语言进行数值化。']]
```

### 字交换

字交换数据增强策略也即将句子中的字的位置随机交换进行数据增强，这里我们将介绍如何使用`paddlenlp.dataaug.CharSwap`进行字级别交换的数据增强。

```text
CharSwap 参数介绍：

    create_n（int）：
        数据增强句子数量。默认为1。

    aug_n（int）：
        数据增强句子中被交换字数量。默认为None

    aug_percent（int）：
        数据增强句子中被交换字数量占全句字比例。如果aug_n不为None，则被交换字数量为aug_n。默认为0.1。

    aug_min (int)：
        数据增强句子中被交换字数量最小值。默认为1。

    aug_max (int)：
        数据增强句子中被交换字数量最大值。默认为10。
```

我们接下来将以下面的例子介绍字级别交换的使用：

``` python
from paddlenlp.dataaug import CharSwap
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

将随机交换句子中的字：
``` python
aug = CharSwap(create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的符号信息，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以松地很轻理解其中的含义。'], ['而计算机只能处理化数值的信息，无法直接理解人类语言，所以需要将人类语言进行数值转换化。']]
```


<a name="文档一键增强"></a>

## 4. 文档一键增强

数据增强API也提供了文档一键增强功能，可以输入指定格式文件进行数据增强。
```text
FileAugment 初始化参数介绍：

    strategies(list)：
        输入应用的数据增强策略。
```

我们接下来将以下面的例子介绍文档一键增强的使用。

只需要传入固定格式的`txt`文件，如下自定义输入文件`data.txt`：

```text
25岁已经感觉脸部松弛了怎么办
小孩的眉毛剪了会长吗？
...
```

我们对文件`data.txt`应用词替换和词插入数据增强策略。

```python
from paddlenlp.dataaug import WordSubstitute, WordInsert, FileAugment
aug1 =  WordSubstitute('synonym', create_n=1, aug_percent=0.1)
aug2 = WordInsert('synonym', create_n=1, aug_percent=0.1)
aug = FileAugment([aug1,aug2])
aug.augment(input_file='data.txt', output_file="aug.txt")
```

数据增强结果保存在`aug.txt`中，如下：
```text
25岁已经感觉面松弛了怎么办
小朋友的眉毛剪了会长吗？
25岁已经感觉脸部松驰松弛了怎么办
幼儿小孩的眉毛剪了会长吗？
```

如果输入的文件中带有文本标签，如下自定义输入文件`data.txt`：

```text
25岁已经感觉脸部松弛了怎么办	治疗方案
小孩的眉毛剪了会长吗？	其他
```
我们可以通过定义`separator`和`separator_id`选择只对其中部分文本进行数据增强策略。
```python
aug.augment(input_file='data.txt', output_file="aug.txt", separator='\t', separator_id=0)
```

数据增强结果保存在`aug.txt`中，如下：

```text
25阴历年已经感觉脸部松弛了怎么办	治疗方案
小孩子的眉毛剪了会长吗？	其他
25岁已经感觉面庞脸部松弛了怎么办	治疗方案
小孩小朋友的眉毛剪了会长吗？	其他
```
