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
  
### 1.1 Query清洗
Embedding Model进行对比学习时，数据质量尤为重要。通过多卡推理与faiss库，快速高效地对数据中的query进行清洗，能够有效去除低质量数据，避免出现过多的相似query，对模型训练造成干扰，影响训练效果。

Query清洗的示例如下：
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
高质量的负样本数据，能够提高对比学习的效率，加快模型的收敛速度。通过多卡推理与faiss库，可以快速高效地挖掘数据中的负样本。

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
Embedding Model训练代码位置详见：
- [run_embedding.py](../../../../llm/../PaddleNLP_zhangjie/llm/run_embedding.py)

Embedding Model训练示例如下：
```
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_embedding.py ./config/xlm_roberta/emb_argument.json
```

## 3.推理评估
训练完成后，可以对Embedding Model进行推理评估，评估指标包括：hit rate，MRR，NDCG等。示例代码如下：
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
- 其中query_pos_passage_path为需要评估的查询文本（query）-正样本（positive passage）对，示例如下：
```
{"query": "四季青的炮制方法是什么?", 
"pos_passage": ["取原药材，除去残枝、枯叶及杂质，略润，切成丝，干燥，筛去灰屑。饮片性状：为大小、长短不一的丝状，革质。上表面光滑有光泽，灰绿色或暗褐色，下表面色较浅，主脉微隆。气微清香，味苦、微涩。贮干燥容器内，置阴凉干燥处。"]}
```
- neg_passage_path为需要加入评估的负样本（negative passage）数据，示例如下：
```
{"neg_passage": ['平时多注意锻炼。饮食方面多吃大叶的绿色蔬菜，肉类食用一些白肉，比如鸡肉和鱼肉，水果可以吃一些含果胶多的，比如苹果、桃子、橙子等。']}
```
- 推理评估后将打印并返回Hit_rate、MRR、NDCG等评价指标。
