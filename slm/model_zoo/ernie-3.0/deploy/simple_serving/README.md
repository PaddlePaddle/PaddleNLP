# 基于 PaddleNLP SimpleServing 的服务化部署

## 目录
- [环境准备](#环境准备)
- [Server 启动服务](#Server 服务启动)
- [其他参数设置](#其他参数设置)

## 环境准备
使用有 SimpleServing 功能的 PaddleNLP 版本

## Server 服务启动
### 文本分类任务启动
#### 启动文本分类 Server 服务
```bash
paddlenlp server server_seq_cls:app --host 0.0.0.0 --port 8189
```

#### 分类任务发送服务
```bash
python client_seq_cls.py --dataset afqmc
```

### 命名实体识别任务启动
#### 启动命名实体识别 Server 服务
```bash
paddlenlp server server_token_cls:app --host 0.0.0.0 --port 8189
```

#### 命名实体识别 Client 发送服务
```bash
python client_token_cls.py
```

###  问答任务启动
#### 启动问答 Server 服务
```bash
paddlenlp server server_qa:app --host 0.0.0.0 --port 8189
```

#### 问答 Client 发送服务
```bash
python client_qa.py
```

## 其他参数设置
可以在 client 端设置 `max_seq_len`, `batch_size` 参数
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
