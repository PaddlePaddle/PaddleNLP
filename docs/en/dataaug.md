# Data Augmentation API

PaddleNLP provides a Data Augmentation API that can be used for training data enhancement.

**Table of Contents**
* [1. Word-level Data Augmentation Strategies](#Word-level-Data-Augmentation-Strategies)
    * [1.1 Word Substitution](#Word-Substitution)
    * [1.2 Word Insertion](#Word-Insertion)
    * [1.3 Word Deletion](#Word-Deletion)
    * [1.4 Word Swap](#Word-Swap)
* [2. Sentence-level Data Augmentation Strategies](#Sentence-level-Data-Augmentation-Strategies)
    * [2.1 Paraphrase Generation](#Paraphrase-Generation)
    * [2.2 Back Translation](#Back-Translation)
    * [2.3 Sentence Summarization](#Sentence-Summarization)
    * [2.4 Sentence Completion](#Sentence-Completion)
* [3. Character-level Data Augmentation Strategies](#Character-level-Data-Augmentation-Strategies)
    * [3.1 Character Substitution](#Character-Substitution)
    * [3.2 Character Insertion](#Character-Insertion)
    * [3.3 Character Deletion](#Character-Deletion)
    * [3.4 Character Swap](#Character-Swap)
* [4. One-click Document Augmentation](#One-click-Document-Augmentation)

<a name="Word-level-Data-Augmentation-Strategies"></a>

## 1. Word-level Data Augmentation Strategies

<a name="Word-Substitution"></a>

### 1.1 Word Substitution
Word substitution is a data augmentation strategy that randomly replaces words in a sentence with alternatives. Here we introduce how to use `paddlenlp.dataaug.WordSubstitute` for word-level substitution data augmentation.

```text
WordSubstitute Parameters:

    aug_type(str or list(str)):
        Word substitution strategy type. Can choose "antonym", "embedding", "synonym", "homonym", "custom", "random", "mlm" or
        combinations of the first four substitution strategies.

    custom_file_path (str, *optional*):
        Local vocabulary file path for data augmentation. Required when selecting "custom" strategy. Default: None.

    create_n(int):
        Number of augmented sentences to generate. Default: 1.

    aug_n(int):
        Number of words to be replaced in each sentence. Default: None

    aug_percent(int):
        Percentage of words to be replaced relative to total words in sentence. If aug_n is specified, this parameter is ignored. Default: 0.1.

    aug_min (int):
        Minimum number of words to be replaced. Default: 1.

    aug_max (int):
        Maximum number of words to be replaced. Default: 10.

    tf_idf (bool):
        Use TF-IDF scores to determine which words to augment. Default: False.

    tf_idf_file (str, *optional*):
        File path for TF-IDF calculation. Required when tf_idf is True. Default: None.
```

We will demonstrate the usage of word substitution with the following example:
``` python
from paddlenlp.dataaug import WordSubstitute
s = ["Human language is abstract information symbols that contain rich semantic meanings, which humans can easily comprehend.","Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations."]
```

**Synonym Substitution**

Replace words in sentences with synonyms based on a synonym dictionary. The percentage of words to be replaced per sentence `aug_percent` and the number of augmented sentences to generate `create_n` can be adjusted as needed. The `synonym` implementation is based on the [Chinese Synonym Dictionary](https://github.com/guotong1988/chinese_dictionary), while `embedding` uses synonym dictionaries constructed from word vector (word embedding) distances. Choose the appropriate dictionary based on actual performance.

``` python
aug = WordSubstitute('synonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols that contain rich semantic meanings, which humans can easily comprehend.', 'Human language is abstract information symbols that contain rich semantic meanings, which humans can easily comprehend.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols that contain rich semantic meanings, which humans can easily comprehend.', 'Human language is abstract information symbols that contain rich semantic meanings, which humans can easily comprehend.'], ['Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations.', 'Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations.']]
```

You can directly set the number of words to be replaced per sentence `aug_n` according to actual requirements:
``` python
aug = WordSubstitute('synonym', create_n=1, aug_n=3)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols that contain rich semantic meanings, which humans can easily comprehend.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols that contain rich semantic meanings, which humans can easily comprehend.'], ['Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations.']]
```
```python
aug = WordSubstitute('embedding', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract phonetic symbols containing rich semantic information, which humans can easily understand.', 'Human language is generalized information symbols containing abundant semantic meaning, people can readily comprehend the implications.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols containing rich semantic meaning, which people can effortlessly grasp.', 'Human language is abstract informational markers containing ample semantic content, humans can intuitively discern the implications.'], ['While computers must process numerical information and cannot directly comprehend human language, thus requiring conversion of human language into numerical representations.', 'Whereas computers can only handle numerical data and cannot directly understand human language, necessitating the transformation of human language into numerical formats.']]
```

**Homophone Substitution**

Replace words with homophones based on a homophone dictionary:

```python
aug = WordSubstitute('homonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is olfactory informational symbols containing rich semantic data, which humans can effortlessly comprehend.', 'Human language is abstract informational symbols containing abundant semantic information, which people can easily grasp.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract informational symbols containing rich semantic data, which humans can effortlessly apprehend.', 'Human language is abstract informational symbols containing ample semantic information, which people can readily understand.'], ['Whereas computers can only process numerical data and cannot directly interpret human language, thus requiring conversion of human language into numerical formats.', 'While computers can only handle numerical information and cannot directly comprehend human language, necessitating the transformation of human language into numerical representations.']]
```

**Antonym Substitution**

Replace words with antonyms based on an antonym dictionary:

```python
aug = WordSubstitute('antonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is concrete informational symbols containing rich semantic data, which humans can effortlessly misunderstand.', 'Human language is concrete informational symbols containing abundant semantic information, which people can readily confuse.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is concrete informational symbols containing rich semantic data, which humans can effortlessly misunderstand.', 'Human language is concrete informational symbols containing ample semantic information, which people can readily confuse.'], ['Whereas computers can only process numerical data and cannot directly comprehend human language, thus requiring conversion of human language into numerical formats.', 'While computers can only handle numerical information and cannot directly understand human language, necessitating the transformation of human language into numerical representations.']]
```

**Local Thesaurus Substitution**

Simply pass the local thesaurus file path via `custom_file_path`
You can use a custom vocabulary for substitution. The local vocabulary file is a fixed-format `json` file, where the dictionary key is the word and the dictionary value is a list of replacement words. For example, a custom local vocabulary `custom.json` looks like:
```
{"人类":["人", "人种","全人类"], "抽象":["abstract","具象"], "轻松":["简单","容易"]}
```

Perform word substitution in sentences using a custom local vocabulary:
```python
custom_file_path = "custom.json"
aug = WordSubstitute('custom', custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人语言是abstract的信息符号，其中蕴含着丰富的语义信息，全人类可以很轻松地理解其中的含义。', '全人类语言是具象的信息符号，其中蕴含着丰富的语义信息，人可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人语言是abstract的信息符号，其中蕴含着丰富的语义信息，人种可以很轻松地理解其中的含义。', '人语言是具象的信息符号，其中蕴含着丰富的语义信息，人种可以很轻松地理解其中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人语言，所以需要将全人类语言进行数值化转换。', '而计算机只能处理数值化的信息，无法直接理解全人类语言，所以需要将人语言进行数值化转换。']]
```

**Combination Substitution**

You can also randomly combine synonym, homophone, and local vocabulary substitutions. For example, combining a synonym vocabulary with a local vocabulary for word substitution:
```python
custom_file_path = "custom.json"
aug = WordSubstitute(['custom','synonym'], custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义音信，生人可以很轻松地领悟其中的含义。', '人种语言是抽象的信息符号，其中蕴含着丰富的贬义信息，人可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信符号，其中蕴含着丰富的语义消息，生人可以很轻松地理解其中的含义。', '人语言是抽象的信息符号，其中蕴含着丰富的语义消息，人类可以很轻松地亮堂其中的含义。'], ['而计算机只能处理数值变成的信息，无法直接理解人类语言，所以需要将生人语言进行数值变为转换。', '而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类言语进行标注值变为转换。']]
```

**Random Word Substitution**

Use random words for in-sentence word substitution:
``` python
aug = WordSubstitute('random', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily understand.', 'Human language is abstract information symbols containing rich semantic information, which humans can easily comprehend.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily understand.', 'Human language is abstract information symbols containing rich semantic information, which humans can easily comprehend.'], ['Computers can only process numerical information and cannot directly understand human language, thus requiring numerical conversion of human language.', 'Computers can only process numerical data and cannot directly comprehend human language, necessitating the numerical conversion of human language.']]

```

**Contextual Substitution**

Contextual substitution randomly masks words in sentences and uses the Chinese pre-trained model ERNIE 1.0 to predict masked words based on context. Compared to vocabulary-based word substitution, contextually predicted words better match sentence content, though requiring longer time for data augmentation.

Using the model to predict words based on context for in-sentence substitution:
``` python
import paddle
# Running in GPU environment
paddle.set_device("gpu")
# Running in CPU environment
# paddle.set_device("cpu")
aug = WordSubstitute('mlm', create_n=1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily understand.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract symbolic representations containing rich semantic information, which humans can easily comprehend.'], ['Computers can only process numerical data and cannot directly understand human language, thus requiring numerical conversion of human language.']]
```
The number of substituted words per sentence currently only supports `aug_n` being 1.

**TF-IDF Based Word Substitution**

The TF-IDF algorithm assumes that if a word appears frequently in a sentence, its importance increases; whereas if it appears frequently in the corpus, its importance decreases. We calculate TF-IDF scores for each word, **with lower TF-IDF scores having higher probability of being substituted**.

We can apply TF-IDF calculation for substitution probability in all the above word substitution strategies by setting `tf_idf` to True and passing the corpus file (containing all training data) `tf_idf_file` to compute word TF-IDF scores. The corpus file must be in fixed `txt` format with one sentence per line. Using corpus file "data.txt" for synonym substitution as example, the corpus file format is:
``` text
Human language is abstract information symbols containing rich semantic information, which humans can easily understand.
Computers can only process numerical information and cannot directly understand human language, thus requiring numerical conversion of human language.
...
```
```python
tf_idf_file = "data.txt"
aug = WordSubstitute('synonym', tf_idf=True, tf_idf_file=tf_idf_file, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, and humans can easily understand their meanings.', 'Human language is abstract message symbols containing abundant semantic information, and people can effortlessly comprehend their implications.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, and humans can easily understand their meanings.', 'Human language is abstract message symbols containing abundant semantic information, and people can effortlessly comprehend their implications.'], ['While computers can only process numerical information and cannot directly understand human language, thus requiring the conversion of human language into numerical representations.', 'Whereas computers can solely handle numerical data and are incapable of directly comprehending human language, necessitating the transformation of human language into numerical formats.']]
```

### Word Insertion
The word insertion data augmentation strategy involves randomly inserting other words into sentences. Here we introduce how to use `paddlenlp.dataaug.WordInsert` for word-level insertion data augmentation.

```text
WordInsert Parameters:

    aug_type(str or list(str)):
        Word insertion augmentation type. Can choose "antonym", "embedding", "synonym", "homonym", "custom", "random", "mlm" or combinations of the first three insertion strategies.

    custom_file_path (str, *optional*):
        Path to local custom lexicon for data augmentation. Required when aug_type is "custom". Default: None.

    create_n (int):
        Number of augmented sentences to generate. Default: 1.

    aug_n (int):
        Number of words to be inserted in augmented sentences. Default: None

    aug_percent (int):
        Percentage of words to be inserted relative to total words in sentence. If aug_n is specified, insertion count uses aug_n. Default: 0.1.

    aug_min (int):
        Minimum number of words to be inserted. Default: 1.

    aug_max (int):
        Maximum number of words to be inserted. Default: 10.
```

We will demonstrate word-level insertion using the following example:

```python
from paddlenlp.dataaug import WordInsert
s = ["Human language is abstract information symbols containing rich semantic information, and humans can easily understand their meanings.", "While computers can only process numerical information and cannot directly understand human language, thus requiring the conversion of human language into numerical representations."]
```

**Synonym Insertion**
Insert synonyms before/after words in sentences based on a synonym lexicon. You can adjust the insertion percentage `aug_percent` and number of augmented sentences `create_n` as needed. The `synonym`
Based on the [Chinese Synonym Dictionary](https://github.com/guotong1988/chinese_dictionary), the `embedding` is constructed using word distances between word vectors to determine synonyms. The appropriate synonym list can be selected based on actual performance.

```python
aug = WordInsert('synonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['全人类人类语言是华而不实抽象的信息符号，其中蕴含着丰富的语义消息信息，人类可以很轻松地理解其中的含义。', '人类语言是抽象的音信信息符号，其中蕴含着丰富的语义消息信息，生人人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言言语是抽象的信息符号，其中蕴含着丰富的语义褒义信息音问，人类可以很轻松地理解其中的含义。', '人类语言是抽象言之无物的信息符号记号，其中蕴含着丰富的语义信息，人类可以很轻松地理解清楚其中的含义。'], ['而计算机只能只得处理数值化变为的信息，无法直接理解人类生人语言，所以需要将人类语言进行数值化转换。', '而计算机只能处理数值分值化化为的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换变换。']]
```

The number of replaced words in sentences can be directly set according to actual requirements using `aug_n`:
```python
aug = WordInsert('synonym', create_n=1, aug_n=3)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言自然语言是抽象的信息符号，其中蕴含着蕴含丰富的语义信息数据，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象具象的信息符号，其中蕴含着丰富的语义演算信息，人类人类文明可以很轻松地理解其中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类全人类语言进行数值最大值化转换切换。']]
```

```python
aug = WordInsert('embedding', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的音符号，其中蕴含着丰富的语义信，生人可以很轻松地理解其中的含义。', '全人类语言是泛泛的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的意思。']]
augmented = aug.augment(s)
print(augmented)
# [['全人类言语是抽象的信符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。', '全人类语言是抽象的信息标记，其中蕴含着丰富的语义信息，人类可以很轻松地略知一二其中的含义。'], ['而计算机不得不处理数值化的信息，无法直接理解人类言语，所以需要将人类语言进行数值化更换。', '而计算机只能处理数值化的信息，无法直接理解人类言语，所以需要将生人语言进行数值化变换。']]
```

**Homophonic Insertion**

Insert homophonic words into sentences based on the homophone list:
``` python
aug = WordInsert('homonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言雨燕是抽象的信息符号，其中蕴含着丰富的语义信息，人类任雷可以很轻松地理解其中的含义寒意。', '人泪人类语言是丑像抽象的心细信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象筹饷的信息符号，其中蕴含着丰富的语义信息，人类可以很轻恨情松地理解力竭其中的含义。', '人类语言是抽象臭香的信息新戏符号，其中蕴含着丰富的语义信息，人类可以很轻很庆松地理解其中的含义。'], ['而计算机只能纸能处理数值化的信息新西，无法直接理解李杰人类语言，所以需要将人类语言进行数值化转换。', '而计算机只能处理数值化的信息，无法直接理解人类语言语嫣，所以需要将人类语言语嫣进行数值书之化转换。']]
```

**Antonym Insertion**

Insert antonyms before/after words in the sentence based on the antonym dictionary:

``` python
aug = WordInsert('antonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象具体的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解懵懂其中的含义。', '人类语言是具体抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地懵懂理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象具体的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解懵懂其中的含义。', '人类语言是具体抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地困惑理解其中的含义。'], ['而计算机只能处理数值化凝的信息，无法直接理解困惑人类语言，所以需要将人类语言进行数值化冻转换。', '而计算机只能处理数值化凝的信息，无法直接理解懵懂人类语言，所以需要将人类语言进行数值化冻转换。']]
```

**Local Thesaurus Insertion**

Simply pass the local thesaurus file path `custom_file_path` to use a custom thesaurus for insertion. The local thesaurus file should be a fixed-format `json` file, where the dictionary key is the word, and the dictionary value is a list of insertion words. For example, a custom local thesaurus `custom.json` would look like:
```
{"人类":["人累", "扔雷"], "抽象":["丑相"], "符号":["富豪","负号","付豪"]}
```

Using custom local thesaurus for word insertion in sentences:
```python
custom_file_path = "custom.json"
aug = WordInsert('custom', custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is an abstract information symbol that contains rich semantic information. Humans can easily understand its meaning.', 'Human language is an abstract information symbol that contains rich semantic information. Humans can easily understand its meaning.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is an abstract information symbol that contains rich semantic information. Humans can easily understand its meaning.', 'Human language is an abstract information symbol that contains rich semantic information. Humans can easily understand its meaning.'], ['While computers can only process numerical information and cannot directly understand human language, human language needs to be converted into numerical values.', 'While computers can only process numerical information and cannot directly understand human language, human language needs to be converted into numerical values.']]
```

**Combined Insertion**

You can also choose to randomly combine synonym, homophone, and local vocabulary for word insertion:

```python
custom_file_path = "custom.json"
aug = WordInsert(['custom','synonym'], custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language vocabulary is abstract information data symbols that contain rich semantic information. Humans can easily understand their meaning.', 'Human language is abstract information symbols that contain rich semantic information. Humans can easily understand their meaning.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols that contain rich semantic data information. Humans can easily understand their meaning.', 'Human language grammar is abstract information symbols that contain rich semantic data information. Humans can easily understand their meaning.'], ['While computer systems can only process numerical value information and cannot directly understand human language, human language needs to be converted into numerical values.', 'While computers can only process numerical result information and cannot directly understand human language, human language needs to be converted into numerical transformation.']]
```

**Random Word Insertion**

Insert random words into sentences:

```python
# [The original code implementation would be here]
```
``` python
aug = WordInsert('random', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols, containing rich semantic information that humans can easily comprehend.', 'seeddestiny human language is abstract information symbols, containing rich semantic information that humans can easily comprehend.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols, containing rich semantic information that humans can easily comprehend.', 'Human language is abstract information symbols, containing rich semantic information that humans can easily comprehend.'], ['Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical form.', 'Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical form.']]
```

**Contextual Insertion**

Contextual insertion randomly masks words in a sentence and utilizes the Chinese pre-trained model ERNIE 1.0 to predict masked words based on contextual information. Compared to vocabulary-based word insertion, contextually inserted words better match sentence content, though requiring more time for data augmentation.

Using the model for contextual word insertion:
``` python
import paddle
# Run on GPU
paddle.set_device("gpu")
# Run on CPU
# paddle.set_device("cpu")
aug = WordInsert('mlm', create_n=1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols, containing rich semantic information that humans can easily comprehend.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols, containing rich semantic information that humans can easily comprehend.'], ['Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical form.']]
```
Currently, only `aug_n`=1 is supported for inserted word count.

### Word Deletion

Word deletion data augmentation strategy randomly removes words from sentences. Here we demonstrate how to use `paddlenlp.dataaug.WordDelete` for word-level deletion.

```text
WordDelete Parameters:

    create_n (int):
        Number of augmented sentences. Default: 1.

    aug_n (int):
        Number of words to delete in augmented sentences. Default: None

    aug_percent (int):
        Percentage of words to delete relative to total words in sentence. If aug_n is specified, word count takes precedence. Default: 0.1.

    aug_min (int):
        Minimum number of words to delete. Default: 1.

    aug_max (int):
        Maximum number of words to delete. Default: 10.
```

We will illustrate word-level deletion using the following example:
``` python
from paddlenlp.dataaug import WordDelete
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

Randomly delete words in the sentence:
``` python
aug = WordDelete(create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的。', '人类语言是抽象的信息符号，其中蕴含着丰富的语义，人类可以松地其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是的信息符号，其中丰富的语义，人类可以很轻松地理解其中的含义。', '人类语言是的信息，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的。'], ['而计算机只能处理数值化的信息，无法直接理解语言，所以需要将人类语言进行转换。', '而计算机处理数值化的信息，无法直接人类语言，所以需要将人类语言进行数值化。']]
```

### Word Swap

The word swap data augmentation strategy involves randomly swapping the positions of words in a sentence. Here, we will introduce how to use `paddlenlp.dataaug.WordSwap` for word-level swap data augmentation.

```text
WordSwap Parameter Description:

    create_n (int):
        Number of augmented sentences. Default is 1.

    aug_n (int):
        Number of words to be swapped in each augmented sentence. If None, the number is determined by aug_percent. Default is None.

    aug_percent (int):
        Percentage of words to be swapped relative to the total words in the sentence. If aug_n is specified, this parameter is ignored. Default is 0.1.

    aug_min (int):
        Minimum number of words to be swapped. Default is 1.

    aug_max (int):
        Maximum number of words to be swapped. Default is 10.
```

We will use the following example to demonstrate word swap usage:

``` python
from paddlenlp.dataaug import WordSwap
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

Randomly swap words in the sentence:
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

## 2. Sentence-level Data Augmentation Strategies

<a name="同义句生成"></a>

### 2.1 Paraphrase Generation

The paraphrase generation data augmentation strategy generates similar sentences based on input sentences. The model first generates `generate_n` sentences, then selects the best `create_n` sentences. Here we demonstrate how to use `paddlenlp.dataaug.SentenceGenerate` for data augmentation through paraphrase generation.

```text
SentenceGenerate Parameters:

    model_name (str):
        The model name for generating paraphrased sentences, options include "roformer-chinese-sim-char-ft-base", "roformer-chinese-sim-char-base", "roformer-chinese-sim-char-ft-small", "roformer-chinese-sim-char-small". Default is "roformer-chinese-sim-char-base".

    create_n (int):
        Number of augmented sentences to select from generated candidates. Default is 1.

    generate_n (int):
        Number of candidate sentences to generate. Default is 5.

    max_length (int):
        Maximum length of generated sentences. Default is 128.

    top_p (float):
        Cumulative probability for top-p sampling, should satisfy math:`0<=top_p<1`. Default is 0.95.
```

The following example demonstrates the usage of paraphrase generation:

```python
from paddlenlp.dataaug import SentenceGenerate
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

```python
import paddle
# Recommended to run in GPU environment
paddle.set_device("gpu")
# To run in CPU environment
# paddle.set_device("cpu")
aug = SentenceGenerate(create_n=2, generate_n=5, max_length=128, top_p=0.95)
augmented = aug.augment(s[0])
print(augmented)
# ['人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义', '人类语言是一个抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']
augmented = aug.augment(s)
print(augmented)
# [['语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。', '人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义答。'], ['而计算机只能处理数值化的信息，无法直接理解人类语言，故需要将人类语言进行数值化转换。', '2、计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。']]
```
<a name="Sentence Back-Translation"></a>

### 2.2 Sentence Back-Translation

The sentence back-translation data augmentation strategy involves translating the input sentence into another language and then translating it back to generate sentences with identical semantics but different expressions. Here we introduce how to use the Baidu Translate API-based `paddlenlp.dataaug.SentenceBackTranslateAPI` and model-based `paddlenlp.dataaug.SentenceBackTranslate` for sentence back-translation data augmentation.

```text
SentenceBackTranslateAPI Parameters:

    src_lang (str):
        Language of input sentences. Default: "zh"

    tgt_lang (str):
        Target language for translation. The augmentation strategy will translate sentences into the target language and then back to the source language. Default: "en"

    appid (str):
        APPID for Baidu Translate API (if using your own Baidu Translate API service appid/secretKey). Default: None

    secretKey (str):
        Secret key for Baidu Translate API (if using your own Baidu Translate API service appid/secretKey). Default: None

    qps (int):
        Queries per second for Baidu Translate API (if using your own Baidu Translate API service appid/secretKey). Default: 1
```

We will demonstrate the usage of Baidu Translate API-based sentence back-translation with the following example:

Using SentenceBackTranslateAPI requires installing PaddleHub
```shell
pip install paddlehub==2.3.1
```

```python
from paddlenlp.dataaug import SentenceBackTranslateAPI
s = ["Human language is abstract information symbols containing rich semantic information, which humans can easily understand.", "Computers can only process numerical information and cannot directly understand human language, thus requiring the conversion of human language into numerical form."]
```

```python
aug = SentenceBackTranslateAPI(src_lang='zh', tgt_lang='en')
augmented = aug.augment(s[0])
print(augmented)
# ['Human language is abstract information symbols containing rich semantic information. Humans can easily understand its meaning.']
augmented = aug.augment(s)
print(augmented)
# ['Human language is abstract information symbols containing rich semantic information. Humans can easily understand its meaning.', 'However, computers can only process numerical information and cannot directly understand human language, thus requiring the conversion of human language into numerical information.']
```

**Notes**
1. By default, this uses the Baidu Translate API service provided by PaddleHub. You can also register your own Baidu Translate API service account to get corresponding AppID and secretKey. For account registration, please refer to [Baidu Translate API Documentation](https://fanyi-api.baidu.com/doc/21). Using your own AppID and secretKey does not require PaddleHub installation.
2. The `src_lang` and `tgt_lang` parameters must follow the language codes specified in the Baidu Translate API documentation.
Parameter Introduction for `SentenceBackTranslate`:

    src_lang (str):
        Language of input sentences. Default: "zh". Supported languages: 'ar', 'cs', 'de', 'en', 'es', 'et', 'fi', 'fr', 'gu', 'hi', 'it', 'ja', 'kk', 'ko', 'lt', 'lv', 'my', 'ne', 'nl', 'ro', 'ru', 'si', 'tr', 'vi', 'zh', 'af', 'az', 'bn', 'fa', 'he', 'hr', 'id', 'ka', 'km', 'mk', 'ml', 'mn', 'mr', 'pl', 'ps', 'pt', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'uk', 'ur', 'xh', 'gl', 'sl'.

    tgt_lang (str):
        Target language for back translation. The enhancement strategy will translate sentences into the target language and then back into the source language. Default: "en". Supported languages: Same as src_lang.

    max_length (int):
        Maximum length of generated similar sentences. Default: 128.

    batch_size (int):
        Batch size. Reduce this value if GPU memory is insufficient. Default: 1.

    num_beams (int):
        Beam number for "beam_search" strategy. Default: 4.

    use_faster (bool):
        Whether to use FasterGeneration for acceleration. Default: False.

    decode_strategy (str):
        Decoding strategy for generation. Currently supports three strategies: "greedy_search", "sampling", and "beam_search". Default: "beam_search".

We will demonstrate the usage of model-based sentence back translation with the following example:

```python
from paddlenlp.dataaug import SentenceBackTranslate
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```
```python
import paddle
# It is recommended to run in a GPU environment
paddle.set_device("gpu")
# Run in CPU environment
# paddle.set_device("cpu")
aug = SentenceBackTranslate(src_lang='zh', tgt_lang='en', batch_size=1, max_length=128)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols, containing rich semantic information that can be easily understood.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols, containing rich semantic information that can be easily understood.'], ['Computers can only process digitized information and cannot directly understand human language, thus digitization is necessary.']]
```

**Note**
1. When `use_faster` is set to True, the first execution of PaddleNLP will trigger Just-In-Time (JIT) Compile to automatically compile high-performance decoding operators. The compilation process typically takes a few minutes. This compilation only needs to be done once, and subsequent uses of high-performance decoding won't require recompilation. The program will continue running after compilation is completed.

<a name="Sentence Summarization"></a>

### 2.3 Sentence Summarization

Sentence summarization data augmentation strategy involves generating summary sentences from input sentences. Here we will introduce how to use `paddlenlp.dataaug.SentenceSummarize` for sentence summarization data augmentation.

```text
SentenceSummarize parameter description:

    create_n (int):
        Number of augmented sentences, selecting the best sentences from generated summaries. Default is 1.

    max_length (int):
        Maximum length of generated summary sentences. Default is 128.

    batch_size (int):
        Batch size. Reduce this value if GPU memory is insufficient. Default is 1.

    top_k (int):
        Number of top-k tokens in "sampling" strategy. 0 means no effect. Default is 5.

    top_p (float):
        Cumulative probability for top-p-filtering in "sampling" strategy. Should satisfy: math:`0<=top_p<1`. Default is 1.0, meaning no effect.

    temperature (float):
        Value for modeling next token probability in "sampling" strategy. Default is 1.0, meaning no effect.

    use_fp16_decoding (bool):
        Whether to use fp16 for acceleration. Default is False.
```

We will use the following example to demonstrate the usage of sentence summarization:

```python
from paddlenlp.dataaug import SentenceSummarize
s = ["Human language is abstract information symbols containing rich semantic information that humans can easily understand.","While computers can only process numerical information and cannot directly understand human language, thus requiring the conversion of human language into numerical form."]
```
``` python
import paddle
# It is recommended to run in a GPU environment
paddle.set_device("gpu")
# Run in CPU environment
# paddle.set_device("cpu")
aug = SentenceSummarize(create_n=2, batch_size=1, max_length=128)
augmented = aug.augment(s[0])
print(augmented)
# [['What is human language?', 'Why is human language considered abstract information symbols?']]
augmented = aug.augment(s)
print(augmented)
# [['What is human language?', 'Why is human language considered abstract information symbols?'], ['Computers can only process numerical information (graph)', 'Computers can only process numerical information']]
```

<a name="SentenceContinuation"></a>

### 2.4 Sentence Continuation

Sentence continuation data augmentation strategy involves generating continuations for input sentences. Here we introduce how to use `paddlenlp.dataaug.SentenceContinue` for sentence continuation data augmentation.

```text
SentenceContinue Parameters:

    model_name (str):
        Name of the sentence continuation model, options: "gpt-cpm-large-cn", "gpt-cpm-small-cn-distill". Default: "gpt-cpm-small-cn-distill".

    max_length (int):
        Maximum length of generated continuation. Default: 128.

    decode_strategy (str):
        Decoding strategy during generation. Currently supports three strategies: "greedy_search", "sampling", and "beam_search". Default: "beam_search".

    use_faster (bool):
        Whether to use FasterGeneration for acceleration. Default: False.

    create_n (int):
        Number of augmented sentences, selecting top sentences from generated continuations. Default: 1.

    top_k (int):
        Number of top-k tokens for "sampling" strategy, 0 means no effect. Default: 5.

    top_p (float):
        Cumulative probability for top-p-filtering in "sampling" strategy. The value should satisfy: math:`0<=top_p<1`. Default: 1.0, meaning no effect.

    temperature (float):
        Temperature value for "sampling" strategy when modeling next token probability. Default: 1.0, meaning no effect.

    batch_size (int):
        Batch size, adjust accordingly if GPU memory is insufficient. Default: 1.
```

We will use the following example to demonstrate the usage of sentence continuation:

``` python
from paddlenlp.dataaug import SentenceContinue
s = ["Human language is abstract information symbols containing rich semantic information, which humans can easily understand.","While computers can only process numerical information and cannot directly understand human language, thus requiring numerical conversion of human language."]
```
``` python
import paddle
# It is recommended to run in GPU environment
paddle.set_device("gpu")
# Run in CPU environment
# paddle.set_device("cpu")
aug = SentenceContinue(create_n=2, batch_size=1, max_length=64)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily understand. However, the abstraction of language itself is not simple. The complexity of language and its abstraction are another important way for humans to understand the world. Information itself is independent of human comprehension ability. The process of humans understanding the world is the continuous enrichment and expansion of information processes.', 'Human language is abstract information symbols containing rich semantic information, which humans can easily understand. But this is also quite challenging. Some things are impossible to achieve. For some people, unattainable goals are just distant dreams, which is why their thoughts often...']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily understand. So why does this phenomenon occur? First, we know humans possess the simplest language, but we cannot directly understand it through language, which necessitates building mathematical models to make the comprehension process more complex than language models.', 'Human language is abstract information symbols containing rich semantic information, which humans can easily understand. If humans could use language to solve linguistic problems, then this issue cannot be avoided. This is why computers exist as linguistic entities, because they can process logical relationships in language. This requires us to meticulously examine the fundamental facts of language and various types of information.']]

```

<a name="CharacterLevelDataAugmentationStrategies"></a>

## 3. Character-Level Data Augmentation Strategies

<a name="CharacterSubstitution"></a>

### 3.1 Character Substitution
Character substitution is a data augmentation strategy that randomly replaces characters in a sentence with other single characters. Here we will explain how to use `paddlenlp.dataaug.CharSubstitute` for character-level substitution data augmentation.
```text
CharSubstitute Parameter Introduction:

    aug_type(str or list(str)):
        Character substitution augmentation strategy type. Can choose "antonym", "homonym", "custom", "random", "mlm", or combinations of the first three character substitution strategies.

    custom_file_path (str, *optional*):
        Local data augmentation character table path. Must be specified when using "custom" substitution strategy. Default: None.

    create_n (int):
        Number of augmented sentences to generate. Default: 1.

    aug_n (int):
        Number of characters to be substituted in each augmented sentence. Default: None.

    aug_percent (int):
        Percentage of characters to be substituted relative to sentence length. If aug_n is specified, it takes precedence. Default: 0.1.

    aug_min (int):
        Minimum number of characters to substitute per sentence. Default: 1.

    aug_max (int):
        Maximum number of characters to substitute per sentence. Default: 10.

Next, we demonstrate character-level substitution with examples:

```python
from paddlenlp.dataaug import CharSubstitute
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。","而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

**Homonym Substitution**

Replace characters with homonyms based on homonym table. You can configure the substitution ratio `aug_percent` and number of augmented sentences `create_n` as needed.

```python
aug = CharSubstitute('homonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是筹象的信汐符号，其中蕴含着逢富的语义锌息，人类可以很轻诵地理解其中的含义。', '人类语嫣是抽象的信息符号，其中蕴含着丰富的语义信息，人垒可以很情松地理婕其种的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的辛息符豪，其中匀含着丰富的语义信息，人类可以很庆耸地理解其中的含义。', '人磊语晏是抽象的新息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理劫其种的含义。'], ['而叽算机只能处理数值化的信息，无法直接理解人蕾语堰，所以需要将人类语演进行数值化专换。', '而疾算机只能杵理数值华的信息，无法直接理捷人类语验，所以需要将人类语言进行数值化转换。']]
```

You can directly set the number of characters to substitute per sentence `aug_n` based on actual requirements:
```
``` python
aug = CharSubstitute('homonym', create_n=1, aug_n=3)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily understand.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily comprehend.'], ['While computers can only process numerical information and cannot directly understand human language, thus requiring numerical conversion of human language.']]
```

**Antonym Character Substitution**

Replace characters in sentences with their antonyms based on an antonym character table:

``` python
aug = CharSubstitute('antonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols containing poor semantic information, which humans can hardly understand.', 'Human language is concrete information symbols containing scarce semantic information, which humans can easily misunderstand.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is concrete information symbols containing abundant semantic information, which humans can easily comprehend.', 'Human language is abstract information symbols containing limited semantic information, which humans can barely understand.'], ['While computers can only process numerical data and cannot directly parse human language, thus necessitating numerical conversion of human language.', 'While computers can solely handle numerical information and fail to directly interpret human language, therefore requiring numerical transformation of human language.']]
```

**Local Character Table Substitution**

Simply pass the local character table file path `custom_file_path` to use a custom character table for substitution. The local character table file should be a fixed-format JSON file where the dictionary key is a character and the value is a list of substitute characters. For example, a custom local character table `custom.json` contains:
```
{"人":["任", "认","忍"], "抽":["丑","臭"], "轻":["亲","秦"],"数":["书","树"],"转":["赚","专"],"理":["里","例"]}
```

Using custom local character table for character substitution in sentences:
``` python
custom_file_path = "custom.json"
aug = CharSubstitute('custom', custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is an abstract information symbol containing rich semantic information, which humans can easily understand.', 'Human language is an odorous information symbol containing rich semantic information, which humans can easily understand.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is an abstract information symbol containing rich semantic information, which humans can easily comprehend.', 'Human language is an odorous information symbol containing rich semantic information, which humans can easily understand.'], ['While computers can only process numerical information and cannot directly comprehend human language, thus requiring conversion of human language into numerical representations.', 'While computers can only process numerical information and cannot directly comprehend human language, thus requiring transformation of human language into numerical representations.']]
```

**Combined Substitution**

We can also combine homophone substitution with local character tables for randomized character replacement. For example, combining homophone tables with local character tables:

``` python
custom_file_path = "custom.json"
aug = CharSubstitute(['custom','homonym'], custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is an abstract information symbol containing rich semantic information, which humans can effortlessly understand.', 'Human language is an abstract information symbol containing abundant semantic information, which humans can easily comprehend.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is an abstract information symbol containing rich semantic information, which humans can effortlessly comprehend.', 'Human language is an abstract information symbol containing abundant semantic information, which humans can easily understand.'], ['While computers can only process numerical information and cannot directly comprehend human language, thus necessitating the conversion of human language into numerical representations.', 'While computers can only process numerical information and cannot directly understand human language, thus requiring the transformation of human language into numerical representations.']]
```

**Random Character Substitution**

Perform random character replacement in sentences using randomized character substitution:
```python
aug = CharSubstitute('random', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人开自言是抽象的信息符号，其中蕴正着丰富的语义信息，人类可以很拜松地理解其中的含侯。', '人类语言是抽象的许息符号，其世蕴银着丰B的语义莘息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类吧言是抽象的信息符号，其中蕴含着丰富的萎义桅小，人类可以很轻松地理解其中的后义。', '人类语言是河象的信夹符号，其中蕴含着丰刘的语义信息，人类可以很轻李地理解其中的含阿。'], ['而庙算机只能处葛数弘化的信息，无法直接理解人类语拉，所以需要将人吴语言进行数值化转换。', '而ｎ算机只能处理数值化的信息，无法直接理解人红语言，所以需要将人类语言进行林值查转P。']]

```

**Contextual Substitution**

Contextual substitution randomly masks characters in a sentence and uses the Chinese pre-trained model ERNIE 3.0 to predict the masked characters based on the context. Compared to character substitution based on a predefined vocabulary, contextually substituted characters better match the sentence content, though the data augmentation process takes longer.

Use the model to predict characters based on context for substitution:
```python
import paddle
# Run in GPU environment
paddle.set_device("gpu")
# Run in CPU environment
# paddle.set_device("cpu")
aug = CharSubstitute('mlm', create_n=1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息符号，其中包含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
```
Currently, the number of substituted characters in a sentence only supports `aug_n` set to 1.

### Character Insertion
The character insertion data augmentation strategy randomly inserts other characters into the sentence. Here we introduce how to use `paddlenlp.dataaug.CharInsert` for character-level insertion data augmentation.

```text
CharInsert Parameter Description:

    aug_type(str or list(str)):
        Character insertion augmentation type. Options include "antonym", "homonym", "custom", "random", "mlm", or combinations of the first three strategies.

    custom_file_path (str, *optional*):
        Local data augmentation character table path. Required when selecting "custom" strategy. Default: None.

    create_n (int):
        Number of augmented sentences. Default: 1.

    aug_n (int):
        Number of characters to insert per sentence. Default: None.

    aug_percent (int):
        Percentage of characters to insert relative to sentence length. If aug_n is specified, this value is ignored. Default: 0.1.

    aug_min (int):
        Minimum number of characters to insert. Default: 1.

    aug_max (int):
        Maximum number of characters to insert. Default: 10.
```
We will use the following example to demonstrate character-level insertion:

```python
from paddlenlp.dataaug import CharInsert
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

**Homonym Insertion**
Insert homophones before/after characters in the sentence based on a homophone list. You can adjust the insertion ratio `aug_percent` (percentage of characters to augment) and the number of augmented sentences `create_n`.

```python
aug = CharInsert('homonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语寓言咽是抽象的信息符复号，其中蕴韵含着丰富夫的语义信息，人类可以很轻松地理解其中的含义。', '人镭类语岩言是抽想象的信息符号，其忠中蕴含着疯丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类勒语言是抽象想的信息符号，其中蕴含着丰富的语誉义以信息，人类可以很轻卿松地理解其中的含义。', '人泪类语言是抽象的芯信息符号，其中蕴含着枫丰富的语疑义锌信息，人类可以很轻松地理解其中的含义。'], ['而计算机只能处理数植值化的新信息，无法直接狸理解人类语言，所以需要将人类峪语言进行书数值化转换。', '而计算机只能处理梳数值化的新信息，无法直接笠理解人类语言，所以需要将人类语衍言进行数值化赚转换。']]
```

You can directly set the number of characters to replace per sentence using `aug_n`:

```python
aug = CharInsert('homonym', create_n=1, aug_n=3)
augmented = aug.augment(s[0])
print(augmented)
# [['人类勒语言是抽象的信息符号，其中蕴含着丰缝富的语义信息，人类可以很轻松颂地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义新信息，人类可以很轻松地荔理解其终中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人类语言，所以序需要将人類类语言进刑行数值化转换。']]
```

**Custom Character Insertion**

To use a custom character table, provide the local file path `custom_file_path`. The custom character table should be a JSON file with specific format, where each key is a character and its value is a list of candidate insertion characters. Example `custom.json`:

```json
{"人":["任", "认","忍"], "抽":["丑","臭"], "轻":["亲","秦"],"数":["书","树"],"转":["赚","专"],"理":["里","例"]}
```

Using custom character table for insertion:

```python
aug = CharInsert('custom', custom_file_path='custom.json', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽丑象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻亲松地理解其中的含义。', '人类语言是抽臭象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻秦松地理解其中的含义。']]
```
``` python
custom_file_path = "custom.json"
aug = CharInsert('custom', custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is an abstract information symbol containing rich semantic information, which humans can easily understand.', 'Human language is an abstract information symbol containing rich semantic information, which humans can easily understand.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is an abstract information symbol containing rich semantic information, which humans can easily understand.', 'Human language is an abstract information symbol containing rich semantic information, which humans can easily understand.'], ['Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations.', 'Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations.']]
```

**Antonym Insertion**

Insert antonyms before/after characters based on an antonym dictionary:

``` python
aug = CharInsert('antonym', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is an abstract (concrete) information symbol containing rich (poor) semantic information, which humans can easily (hard) understand.', 'Human language is an abstract information (disinformation) symbol containing rich semantic information, which humans can easily understand.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is an abstract (concrete) information symbol containing rich (poor) semantic information, which humans can easily (hard) understand.', 'Human language is an abstract information (disinformation) symbol containing rich semantic information, which humans can easily understand.'], ['Computers can only process numerical (categorical) information and cannot directly (indirectly) understand human language, thus requiring conversion of human language into numerical representations.', 'Computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical (non-numerical) representations.']]
```

**Combination Insertion**

We can also randomly combine homophones, homonyms and local dictionaries for character insertion. For example, combining homophone dictionary and local dictionary:

```
``` python
custom_file_path = "custom.json"
aug = CharInsert(['custom','homonym'], custom_file_path=custom_file_path, create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily understand.', 'Human language is abstract information symbols containing rich semantic information, which humans can easily understand.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily understand.', 'Human language is abstract information symbols containing rich semantic information, which humans can easily understand.'], ['While computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations.', 'While computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations.']]
```

**Random Character Insertion**

Insert randomly selected characters into sentences:
``` python
aug = CharInsert('random', create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['HumS language is abstract information symbols containing rich semantic information, which humans can easily understand.', 'HumS language is abstract information symbols containing rich semantic information, which humans can easily understand.']]
augmented = aug.augment(s)
print(augmented)
# [['Human language is abstract information symbols containing rich semantic information, which humans can easily understand.', 'Human language is abstract information symbols containing rich semantic information, which humans can easily understand.'], ['While computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations.', 'While computers can only process numerical information and cannot directly understand human language, thus requiring conversion of human language into numerical representations.']]
```

**Contextual Insertion**

Contextual insertion randomly masks individual characters in sentences and uses the Chinese pre-trained model ERNIE 3.0 to predict masked characters based on contextual information. Compared to character table-based insertion, contextually predicted characters better match sentence content, though requiring longer time for data augmentation.

Using the model to predict characters based on context for sentence insertion:
``` python
import paddle
# Run on GPU
paddle.set_device("gpu")
# Run on CPU
# paddle.set_device("cpu")
aug = CharInsert('mlm', create_n=1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的信息息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。'], ['而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值转化转换。']]
```
The number of characters inserted in a sentence currently only supports `aug_n` set to 1.

### Character Deletion

The character deletion data augmentation strategy randomly removes characters from sentences. Here we introduce how to use `paddlenlp.dataaug.CharDelete` for character-level deletion data augmentation.

```text
CharDelete Parameters:

    create_n (int):
        Number of augmented sentences. Defaults to 1.

    aug_n (int):
        Number of characters to delete in each augmented sentence. Defaults to None.

    aug_percent (int):
        Percentage of characters to delete relative to sentence length. If aug_n is not None, the number of deleted characters will be aug_n. Defaults to 0.1.

    aug_min (int):
        Minimum number of characters to delete. Defaults to 1.

    aug_max (int):
        Maximum number of characters to delete. Defaults to 10.
```

We will demonstrate the usage of character-level deletion with the following example:

``` python
from paddlenlp.dataaug import CharDelete
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

Randomly delete characters from sentences:
``` python
aug = CharDelete(create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的。', '人类语言是抽象的信息符号，其中蕴含着丰富的语义，人类可以松地其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是的信息符号，其中丰富的语义，人类可以很轻松地理解其中的含义。', '人类语言是的信息，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的。'], ['而计算机只能处理数值化的信息，无法直接理解语言，所以需要将人类语言进行转换。', '而计算机处理数值化的信息，无法直接人类语言，所以需要将人类语言进行数值化。']]
```

### Character Swapping

The character swapping data augmentation strategy randomly swaps character positions in sentences. Here we introduce how to use `paddlenlp.dataaug.CharSwap`
# Character-Level Swapping for Data Augmentation

```text
CharSwap Parameter Introduction:

    create_n (int):
        Number of augmented sentences. Default is 1.

    aug_n (int):
        Number of characters to be swapped in each augmented sentence. Default is None.

    aug_percent (int):
        Percentage of characters to be swapped relative to total sentence length. If aug_n is not None, the number of swapped characters will be aug_n. Default is 0.1.

    aug_min (int):
        Minimum number of characters to be swapped. Default is 1.

    aug_max (int):
        Maximum number of characters to be swapped. Default is 10.
```

We will demonstrate the usage of character-level swapping with the following example:

```python
from paddlenlp.dataaug import CharSwap
s = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
```

Randomly swap characters in sentences:
```python
aug = CharSwap(create_n=2, aug_percent=0.1)
augmented = aug.augment(s[0])
print(augmented)
# [['人类语言是抽象的符号信息，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。']]
augmented = aug.augment(s)
print(augmented)
# [['人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以松地很轻理解其中的含义。'], ['而计算机只能处理化数值的信息，无法直接理解人类语言，所以需要将人类语言进行数值转换化。']]
```


<a name="Document One-Click Augmentation"></a>

## 4. Document One-Click Augmentation

The data augmentation API also provides document one-click augmentation feature, which can perform data augmentation on specified format files.

```text
FileAugment Initialization Parameters:

    strategies (list):
        List of data augmentation strategies to apply.
```

We will demonstrate the usage of document one-click augmentation with the following example.

Just process a custom input file `data.txt` in fixed-format txt:

```text
25岁已经感觉脸部松弛了怎么办
小孩的眉毛剪了会长吗？
...
```

Apply word substitution and word insertion augmentation strategies to file `data.txt`:

```python
from paddlenlp.dataaug import WordSubstitute, WordInsert, FileAugment
aug1 = WordSubstitute('synonym', create_n=1, aug_percent=0.1)
aug2 = WordInsert('synonym', create_n=1, aug_percent=0.1)
aug = FileAugment([aug1, aug2])
aug.augment(input_file='data.txt', output_file="aug.txt")
```

The augmented results are saved in `aug.txt` as follows:
```text
What to do if you already feel facial sagging at 25?
Will a child's eyebrows grow back if trimmed?
What to do if you already feel facial sagging at 25?
Will a young child's eyebrows grow back if trimmed?
```

If the input file contains text labels, such as the following custom input file `data.txt`:

```text
What to do if you already feel facial sagging at 25?    Treatment Plan
Will a child's eyebrows grow back if trimmed?    Others
```

We can apply data augmentation strategies to only part of the text by defining `separator` and `separator_id`:
```python
aug.augment(input_file='data.txt', output_file="aug.txt", separator='\t', separator_id=0)
```

The data augmentation results are saved in `aug.txt` as:

```text
What to do if you already feel facial sagging at lunar age?    Treatment Plan
Will a young child's eyebrows grow back if trimmed?    Others
What to do if you already feel facial sagging at 25?    Treatment Plan
Will a child's eyebrows grow back if trimmed?    Others
```
