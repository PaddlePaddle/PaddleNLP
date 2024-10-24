# 基于 PaddleNLP SimpleServing 的服务化部署

## 目录
- [环境准备](#环境准备)
- [Server 启动服务](#Server 服务启动)
- [其他参数设置](#其他参数设置)

## 环境准备
使用有 SimpleServing 功能的 PaddleNLP 版本
```shell
pip install paddlenlp --upgrade
```
## Server 服务启动
### 分类任务启动
#### 启动 分类 Server 服务
```bash
paddlenlp server server:app --host 0.0.0.0 --port 8189
```
如果是 ERNIE-M 模型则启动
```bash
paddlenlp server ernie_m_server:app --host 0.0.0.0 --port 8189
```
#### 分类任务发送服务
```bash
python client.py
```

## 其他参数设置
可以在 client 端设置 `max_seq_len`, `batch_size`, `prob_limit` 参数
```python
    data = {
        'data': {
            'text': texts,
        },
        'parameters': {
            'max_seq_len': args.max_seq_len,
            'batch_size': args.batch_size,
            'prob_limit': args.prob_limit
        }
    }
```
