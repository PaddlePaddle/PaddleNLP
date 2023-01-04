# 基于PaddleNLP SimpleServing 的服务化部署

## 目录
- [环境准备](#环境准备)
- [Server启动服务](#Server服务启动)
- [其他参数设置](#其他参数设置)

## 环境准备

使用有SimpleServing功能的PaddleNLP版本

```shell
pip install paddlenlp >= 2.5.0
```

## Server服务启动

paddlenlp server server:app --workers 1 --host 0.0.0.0 --port 8189

## Client请求启动

```bash
python client.py
```

## 自定义参数设置

可在 client 端设置以下参数：
- `data`：输入文本，每条样本为一个字典，至少包括`text_a`关键字，其余关键字`text_b`，`choices`可选。
- `max_length`：最长文本长度，包括所有标签候选长度。
- `batch_size`：每次预测的样本数量。
- `prob_limit`：预测标签的阈值，默认为0.8。
- `choices`：标签候选列表。当`data`中未定义标签候选 `choices` 关键字时，用于定义标签候选。
