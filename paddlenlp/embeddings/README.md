# paddlenlp.embeddings

## TokenEmbedding参数

|  参数 | 类型  | 属性  |
| ------------ | ------------ | ------------ |
| embedding_name | **string**  | 预训练embedding名称，可通过paddlenlp.embeddings.list_embedding_name()或[Embedding 模型汇总](../../docs/model_zoo/embeddings.md)查询。 |
| unknown_token | **string**  | unknown token。 |
| unknown_token_vector | **list** 或者 **np.array** | 用来初始化unknown token对应的vector。默认为None（以正态分布方式初始化vector）|
| extended_vocab_path | **string**  | 扩展词表的文件名路径。词表格式为一行一个词。 |
| trainable | **bool**  | 是否可训练。True表示Embedding可以更新参数，False为不可更新。 |

## 初始化
```python
import paddle
from paddlenlp.embeddings import TokenEmbedding, list_embedding_name
paddle.set_device("cpu")

# 查看预训练embedding名称：
print(list_embedding_name()) # ['w2v.baidu_encyclopedia.target.word-word.dim300']

# 初始化TokenEmbedding， 预训练embedding没下载时会自动下载并加载数据
token_embedding = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300")

# 查看token_embedding详情
print(token_embedding)

Object   type: <paddlenlp.embeddings.token_embedding.TokenEmbedding object at 0x7fda7eb5f290>
Unknown index: 635963
Unknown token: [UNK]
Padding index: 635964
Padding token: [PAD]
Parameter containing:
Tensor(shape=[635965, 300], dtype=float32, place=CPUPlace, stop_gradient=False,
       [[-0.24200200,  0.13931701,  0.07378800, ...,  0.14103900,  0.05592300, -0.08004800],
        [-0.08671700,  0.07770800,  0.09515300, ...,  0.11196400,  0.03082200, -0.12893000],
        [-0.11436500,  0.12201900,  0.02833000, ...,  0.11068700,  0.03607300, -0.13763499],
        ...,
        [ 0.02628800, -0.00008300, -0.00393500, ...,  0.00654000,  0.00024600, -0.00662600],
        [-0.00924490,  0.00652097,  0.01049327, ..., -0.01796000,  0.03498908, -0.02209341],
        [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]])

```

## 查询embedding结果

```python
test_token_embedding = token_embedding.search("中国")
print(test_token_embedding)
[[ 0.260801  0.1047    0.129453 -0.257317 -0.16152   0.19567  -0.074868
   0.361168  0.245882 -0.219141 -0.388083  0.235189  0.029316  0.154215
  -0.354343  0.017746  0.009028  0.01197  -0.121429  0.096542  0.009255
   ...,
  -0.260592 -0.019668 -0.063312 -0.094939  0.657352  0.247547 -0.161621
   0.289043 -0.284084  0.205076  0.059885  0.055871  0.159309  0.062181
   0.123634  0.282932  0.140399 -0.076253 -0.087103  0.07262 ]]
```

## 可视化embedding结果
使用深度学习可视化工具[VisualDL](https://github.com/PaddlePaddle/VisualDL)的High Dimensional组件可以对embedding结果进行可视化展示，便于对其直观分析，步骤如下：
```python
# 获取词表中前1000个单词
labels = token_embedding.vocab.to_tokens(list(range(0,1000)))
test_token_embedding = token_embedding.search(labels)

# 引入VisualDL的LogWriter记录日志
from visualdl import LogWriter

with LogWriter(logdir='./visualize') as writer:
    writer.add_embeddings(tag='test', mat=test_token_embedding, metadata=labels)
```
执行完毕后会在当前路径下生成一个visualize目录，并将日志存放在其中，我们在命令行启动VisualDL即可进行查看，启动命令为：
```shell
visualdl logdir ./visualize
```
启动后打开浏览器即可看到可视化结果

<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/103188111-1b32ac00-4902-11eb-914e-c2368bdb8373.gif" width="80%"/>
</p>

使用VisualDL除可视化embedding结果外，还可以对标量、图片、音频等进行可视化，有效提升训练调参效率。关于VisualDL更多功能和详细介绍，可参考[VisualDL使用文档](https://github.com/PaddlePaddle/VisualDL/tree/develop/docs)。

## 计算词向量cosine相似度

```python
score = token_embedding.cosine_sim("中国", "美国")
print(score) # 0.49586025
```

## 计算词向量内积

```python
score = token_embedding.dot("中国", "美国")
print(score) # 8.611071
```


## 训练

以下为`TokenEmbedding`简单的组网使用方法。有关更多`TokenEmbedding`训练流程相关的使用方法，请参考[Word Embedding with PaddleNLP](../../examples/word_embedding/README.md)。

```python
in_words = paddle.to_tensor([0, 2, 3])
input_embeddings = token_embedding(in_words)
linear = paddle.nn.Linear(token_embedding.embedding_dim, 20)
input_fc = linear(input_embeddings)
print(input_fc)
Tensor(shape=[3, 20], dtype=float32, place=CPUPlace, stop_gradient=False,
       [[ 0.        ,  0.        ,  0.        ,  ...,  0.        ,  0.        ,  0.        ],
        [-0.23473957,  0.17878169,  0.07215232,  ...,  0.03698236,  0.14291850,  0.05136518],
        [-0.42466098,  0.15017235, -0.04780108,  ..., -0.04995505,  0.15847842,  0.00025209]])
```

## 切词

```python
from paddlenlp.data import JiebaTokenizer
tokenizer = JiebaTokenizer(vocab=token_embedding.vocab)
words = tokenizer.cut("中国人民")
print(words) # ['中国人', '民']

tokens = tokenizer.encode("中国人民")
print(tokens) # [12530, 1334]
```
