# Faster Tokenizer 性能测试

为了进一步对比Faster Tokenizer的性能，我们选取的业界对于Transformer类常用的Tokenizer分词工具进行对比。
我们以 bert-base-chinese 模型为例，对比的Tokenizer分词工具有以下选择：

* HuggingFace BertTokenizer: 以下简称 HFTokenizer。

```python
from transformers import AutoTokenizer

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=False)
```

* [HuggingFace BertTokenizerFast](https://github.com/huggingface/tokenizers): 以下简称 HFFastTokenizer。

```python
from transformers import AutoTokenizer

hf_tokenizer_fast = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)
```

* [TensorFlow-Text BertTokenizer](https://www.tensorflow.org/text/api_docs/python/text/BertTokenizer)：以下简称 TFTokenizer。

```python
import tensorflow_text as tf_text

# vocab 为bert-base-chinese的词汇表
tf_tokenizer = tf_text.BertTokenizer(vocab)
```

* PaddleNLP BertTokenizer：以下简称 PPNLPTokenizer

```python
from paddlenlp.transformers import BertTokenizer

py_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
```



## 环境依赖

* paddlenlp >= 2.2.0

* transformers == 4.11.3

* tensorflow_text == 2.5.0


```shell
pip install -r requirements.txt
```

## 运行

```shell
python perf.py
```

- 测试环境：

    * CPU： Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz，物理核数40
    * GPU： CUDA 10.2, CuDNN 7.6.5, 16G

- 测试结果：

    文本序列长度为128， 线程数为16，batch_size=32的性能对比结果：


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/9d46bfe903614444b4cf9e63206b28ee06f06c5d5cb04da58bb206431904af00"  ></center>
<br> <center> Tokenizer性能对比 </center></br>

从以上结果可以看出，FasterTokenizer性能远远超过了其他Tokenizer， 高达HFFastTokenizer性能20倍。
