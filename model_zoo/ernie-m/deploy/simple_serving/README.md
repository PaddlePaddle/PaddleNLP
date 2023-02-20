# 基于PaddleNLP SimpleServing 的服务化部署

## 目录
- [环境准备](#环境准备)
- [Server启动服务](#Server服务启动)
- [其他参数设置](#其他参数设置)

## 环境准备

paddlenlp >= 2.5.0

## Server服务启动
### 文本分类任务启动
#### 启动文本分类 Server 服务
```bash
paddlenlp server server_seq_cls:app --host 0.0.0.0 --port 8189
```

#### 分类任务发送服务
```bash
python client_seq_cls.py --language zh
```

## 其他参数设置
可以在client端设置 `max_seq_len`, `batch_size` 参数
```python
    data = {
        'data': {
            'text': texts,
            'text_pair': text_pairs
        },
        'parameters': {
            'max_seq_len': args.max_seq_len,
            'batch_size': args.batch_size
        }
    }
```
