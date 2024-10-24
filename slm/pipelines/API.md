# API 的预置模型介绍

以下是`Pipelines`的主要 API 的模型介绍，有其他定制化的需求的用户可提 issue。

## DensePassageRetriever

除了`DensePassageRetriever`的默认模型外，还可以选择下面的模型试试效果：

| 模型  | 语言 | 模型详细信息 |
| -------- | -------- | -------- |
| rocketqa-zh-base-query-encoder     | Chinese     | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on DuReader retrieval text.     |
| rocketqa-zh-medium-query-encoder     | Chinese     | 6-layer, 768-hidden, 12-heads, 75M parameters. Trained on DuReader retrieval text.     |
| rocketqa-zh-mini-query-encoder     | Chinese     | 6-layer, 384-hidden, 12-heads, 27M parameters. Trained on DuReader retrieval text.     |
| rocketqa-zh-micro-query-encoder    | Chinese     | 4-layer, 384-hidden, 12-heads, 23M parameters. Trained on DuReader retrieval text.     |
| rocketqa-zh-nano-query-encoder     | Chinese     | 4-layer, 312-hidden, 12-heads, 18M parameters. Trained on DuReader retrieval text.     |
| rocketqav2-en-marco-query-encoder    | English     | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on MSMARCO.     |
| ernie-search-base-dual-encoder-marco-en    | English     | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on MSMARCO.     |

## ErnieRanker

类似地`ErnieRanker`可以选择下面的模型试试效果：

| 模型  | 语言 | 模型详细信息 |
| -------- | -------- | -------- |
| rocketqa-base-cross-encoder     | Chinese     | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on DuReader retrieval text.     |
| rocketqa-medium-cross-encoder     | Chinese     | 6-layer, 768-hidden, 12-heads, 75M parameters. Trained on DuReader retrieval text.     |
| rocketqa-mini-cross-encoder    | Chinese     | 6-layer, 384-hidden, 12-heads, 27M parameters. Trained on DuReader retrieval text.     |
| rocketqa-micro-cross-encoder     | Chinese     | 4-layer, 384-hidden, 12-heads, 23M parameters. Trained on DuReader retrieval text.     |
| rocketqa-nano-cross-encoder    | Chinese     | 4-layer, 312-hidden, 12-heads, 18M parameters. Trained on DuReader retrieval text.    |
| rocketqav2-en-marco-cross-encoder    | English     | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on Trained on MSMARCO.    |
| ernie-search-large-cross-encoder-marco-en    | English     | 24-layer, 768-hidden, 12-heads, 118M parameters. Trained on Trained on MSMARCO.    |

## ErnieReader

`ErnieReader`目前提供了一个预置的模型：

| 模型  | 语言 | 模型详细信息 |
| -------- | -------- | -------- |
| ernie-gram-zh-finetuned-dureader-robust     | Chinese     | 12-layer, 768-hidden, 12-heads, 118M parameters. Trained on DuReader Robust Text.     |
