# 飞桨FasterTokenizer性能测试

在PaddleNLP v2.2.0版本中PaddleNLP推出了高性能的Transformer类文本分词器，简称飞桨FasterTokenizer。为了验证飞桨FasterTokenizer的性能快的特点，PaddleNLP选取了业内常见的一些文本分词器进行了性能对比比较，主要进行性能参考的是HuggingFace BertTokenizer， Tensorflow-text BertTokenizer. 我们以 bert-base-chinese 模型为例进行了文本分词性能实验对比，在中文的数据下进行性能对比实验，下面是具体实验设置信息：
* [HuggingFace Tokenizers(Python)](https://github.com/huggingface/tokenizers):

```python
from transformers import AutoTokenizer

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=False)
```

* [HuggingFace Tokenizers(Rust)](https://github.com/huggingface/tokenizers):

```python
from transformers import AutoTokenizer

hf_tokenizer_fast = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)
```

* [TensorFlow-Text](https://www.tensorflow.org/text/api_docs/python/text/BertTokenizer):

```python
import tensorflow_text as tf_text

# vocab 为bert-base-chinese的词汇表
tf_tokenizer = tf_text.BertTokenizer(vocab)
```

* [飞桨FasterTokenizer](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/experimental):

```python
from paddlenlp.experimental import FasterTokenizer

faster_tokenizer = FasterTokenizer.from_pretrained("bert-base-chinese")

```


## 环境依赖

* paddlepaddle >= 2.2.1
* paddlenlp >= 2.2
* transformers == 4.11.3
* tokenizers == 0.10.3
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


<center><img width="1343" alt="图片" src="https://user-images.githubusercontent.com/16698950/145664356-0b766d5a-9ff1-455a-bb85-1ee51e2ad77d.png"></center>

飞桨FasterTokenizer与其他框架性能的对比，是在固定文本长度在不同batch size下的分词吞吐量。纵坐标是对数坐标，单位是1w tokens/秒。随着batch size的增大，飞桨FasterTokenizer速度会远远超过其他同类产品的实现，尤其是在大batch文本上飞桨框架能充分发挥多核机器的优势，取得领先的速度。
