# [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)

## 摘要
大型 Transformer 模型通常会在许多任务上获得最先进的结果，但训练这些模型的成本可能高得惊人，尤其是在长序列上。 我们介绍了两种技术来提高 Transformer 的效率。 一方面，我们将点积注意力替换为使用局部敏感哈希的注意力，将其复杂度从 O(L²) 降为 O(LlogL)，其中 L 是序列的长度。 此外，我们使用可逆残差层而不是标准残差，这允许在训练过程中仅存储一次激活而不是 N 次，其中 N 是层数。 生成的模型，Reformer，与 Transformer 模型的性能相当，同时在长序列上的内存效率更高，速度更快。

## 文本生成测试
```sh
python demo.py
```
模型生成使用到的参数释义如下：
- `decode_strategy` 解码策略，可选择`greedy_search`和`sampling`。
- `max_predict_len` 表示最大生成的句子长度。
- `repetition_penalty` 表示生成重复token的惩罚参数，详细信息可查看[这篇论文](https://arxiv.org/pdf/1909.05858.pdf)。

## 生成结果样例

```
In 1965, Brooks left IBM to found the Department of Defense. The Department was able to convince the Department to resign from the Department's constitutional amendments to the Department of Defense.\n\n
```
