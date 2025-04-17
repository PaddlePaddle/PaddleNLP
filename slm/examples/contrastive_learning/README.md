# Contrastive Learning (CL)

对比学习（Contrastive Learning）是一种自监督学习的方法，旨在通过比较样本之间的相似性和差异性来学习数据的有效表示。在对比学习中，模型通常被训练以最大化相似样本对（正样本对）的相似性，同时最小化非相似样本对（负样本对）的相似性。这种方法不需要明确的标签信息，因此能够利用大量未标注的数据进行训练。

对比学习的优势在于其能够有效利用未标注数据，减少对大规模标注数据的依赖，并且通常能在下游任务中获得强大的泛化能力。这种方法在文本、图像以及其他领域的数据表示学习中都表现出了优异的性能。

## 1.数据准备
以文本向量模型（Embedding Model）的对比学习为例，需要准备的数据样例如下：
```
{"query": "四季青的炮制方法是什么?",
"pos_passage": ["取原药材，除去残枝、枯叶及杂质，略润，切成丝，干燥，筛去灰屑。饮片性状：为大小、长短不一的丝状，革质。上表面光滑有光泽，灰绿色或暗褐色，下表面色较浅，主脉微隆。气微清香，味苦、微涩。贮干燥容器内，置阴凉干燥处。"],
"neg_passage": ['平时多注意锻炼。饮食方面多吃大叶的绿色蔬菜，肉类食用一些白肉，比如鸡肉和鱼肉，水果可以吃一些含果胶多的，比如苹果、桃子、橙子等。']}
```
**注释**:
- query : 查询文本
- pos_passage : 查询文本对应的正样本列表
- neg_passage : 查询文本对应的负样本列表

### 1.1 Query 清洗
Embedding Model 进行对比学习时，训练效果高度依赖于数据的质量。如果数据集中存在大量相似的 query，模型可能会陷入过度关注这些样本的误区，忽视其他关键特征，进而干扰训练过程，削弱最终效果。因此，在进行对比学习之前，对 query 进行清洗，去除相似的 query，是提升模型性能的关键步骤。

我们设计了以下步骤，以高效完成相似 query 的清洗任务：
- **构建向量表示**。利用 Embedding Model 将每个 query 转换为向量表示。
- **计算文本相似度**。利用余弦相似度计算 query 之间的相似度，并设置相似度阈值。如果 query 之间的相似度超过阈值，则认为它们是相似的，需进行去重处理。
- **多步骤加速**。用了多卡推理技术，充分利用多 GPU 的并行计算能力加速向量表示的构建过程。同时，结合 faiss 库构建高效的向量索引，并通过 GPU 进一步加速相似度计算与 query 召回过程。

Query 清洗的示例如下：
```
from clean_query import Clean_Query

model_path = 'BAAI/bge-m3'
tokenizer_path = 'BAAI/bge-m3'
input_data_path = './toy_data/toy_source.json'
output_data_path='./toy_data/test_clean.json'
test_clean = Clean_Query(model_path, tokenizer_path, input_data_path=input_data_path, output_data_path=output_data_path, similarity_threshold=0.70)
test_clean.clean()
```

可以通过多卡推理有效提高清洗效率，示例如下：
```
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" clean_query.py
```

### 1.2 负样本挖掘
在对比学习框架中，负样本的质量与数量对模型训练效率及收敛速度具有显著影响。高质量的负样本能够有效提升模型区分正负样本的能力，从而加速训练过程并增强模型的泛化性能。然而，从海量数据中高效挖掘出真正具有挑战性的负样本，是一项既复杂又关键的任务。

为此，我们设计了以下步骤，旨在快速精准地从数据集中筛选出有价值的负样本：

- **构建向量表示**。利用 Embedding Model 将 query 与 positive passage 转换为向量表示。
- **负样本识别**。在向量空间中，计算 query 与候选样本之间的余弦相似度，召回识别出那些虽与 query 不直接相关但具有一定相似性的样本作为负样本。这类样本能够促使模型在训练过程中学习到更细腻的特征区分能力。
- **多步骤加速**。为提升负样本挖掘的效率，我们采用了多卡推理技术，充分利用多 GPU 的并行计算能力加速向量表示的构建过程。同时，结合 faiss 库构建高效的向量索引，并通过 GPU 进一步加速负样本的召回与筛选过程。

负样本挖掘的示例如下：
```
from mining_negative_samples import MiningNegativeSamples

input_data_path='./toy_data/toy_source.json'
output_data_path='./toy_data/test_min_neg.json'
model_path = 'BAAI/bge-m3'
tokenizer_path = 'BAAI/bge-m3'
test_mining = MiningNegativeSamples(model_path, tokenizer_path, input_data_path=input_data_path, output_data_path=output_data_path)
test_mining.mining()
```

可以通过多卡推理有效提高挖掘效率，示例如下：
```
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" mining_negative_samples.py
```

## 2.训练
Embedding Model 训练代码位置详见：
- [run_embedding.py](../../../llm/run_embedding.py)

Embedding Model 训练示例如下：
```
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_embedding.py ./config/xlm_roberta/emb_argument.json
```

## 3.推理评估
训练完成后，可以对 Embedding Model 进行推理评估，评估指标包括：hit rate，MRR，NDCG 等。示例代码如下：
```
model_path = 'BAAI/bge-m3'
tokenizer_path = 'BAAI/bge-m3'
query_pos_passage_path = './toy_data/toy_dev.json'
neg_passage_path = './toy_data/toy_dev_neg.json'
eval = Embedding_Evaluation(model_path, tokenizer_path, query_pos_passage_path, neg_passage_path)
print(eval.evaluate())
```

多卡推理评估的示例如下：
```
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" embedding_evaluate.py
```
**注释**:
- 其中 query_pos_passage_path 为需要评估的查询文本（query）-正样本（positive passage）对，示例如下：
```
{"query": "四季青的炮制方法是什么?",
"pos_passage": ["取原药材，除去残枝、枯叶及杂质，略润，切成丝，干燥，筛去灰屑。饮片性状：为大小、长短不一的丝状，革质。上表面光滑有光泽，灰绿色或暗褐色，下表面色较浅，主脉微隆。气微清香，味苦、微涩。贮干燥容器内，置阴凉干燥处。"]}
```
- neg_passage_path 为需要加入评估的负样本（negative passage）数据，示例如下：
```
{"neg_passage": ['平时多注意锻炼。饮食方面多吃大叶的绿色蔬菜，肉类食用一些白肉，比如鸡肉和鱼肉，水果可以吃一些含果胶多的，比如苹果、桃子、橙子等。']}
```
- 推理评估后将打印并返回 Hit_rate、MRR、NDCG 等评价指标。

## Acknowledge

我们借鉴了[FlagEmbedding/Tutorials/7_Fine-tuning](https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/7_Fine-tuning/7.2.1_Hard_Negative_Mining.ipynb)中有关挖掘强负样本的代码设计，在此对其作者表示感谢。
