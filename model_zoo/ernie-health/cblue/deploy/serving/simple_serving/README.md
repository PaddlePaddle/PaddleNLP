# 基于PaddleNLP SimpleServing 的服务化部署

## 目录
- [环境准备](#环境准备)
- [Server启动服务](#Server服务启动)
- [其他参数设置](#其他参数设置)

## 环境准备
使用有SimpleServing功能的PaddleNLP版本
```shell
pip install paddlenlp >= 2.3.6
```
## Server服务启动
### 分类任务启动
#### 启动 分类 Server 服务
```bash
paddlenlp server server_classification:app --host 0.0.0.0 --port 8189
```

#### 分类任务发送服务
```bash
python client_classification.py --dataset  chip-cdn-2c
```

### NER 任务启动
#### 启动 NER Server 服务
```bash
paddlenlp server server_ner:app --host 0.0.0.0 --port 8189
```

#### NER Client发送服务
```bash
python client_ner.py
```

### SPO 任务启动
#### 启动 SPO Server 服务
```bash
paddlenlp server server_spo:app --host 0.0.0.0 --port 8189
```

#### SPO Client 发送服务
```bash
python client_spo.py
```

## 其他参数设置
可以在client端设置 `max_seq_len`, `batch_size` 参数
```python
    data = {
        'data': {
            'text': texts,
            'text_pair': text_pairs if len(text_pairs) > 0 else None
        },
        'parameters': {
            'max_seq_len': args.max_seq_len,
            'batch_size': args.batch_size
        }
    }
```
