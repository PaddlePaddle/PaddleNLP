# 基于PaddleNLP SimpleServing 的服务化部署

## 目录
- [环境准备](#环境准备)
- [Server启动服务](#模型转换)
- [Client发送请求](#部署模型)

## 环境准备
使用有SimpleServing功能的PaddleNLP版本
```shell
pip install paddlenlp >= 2.4.3
```
## Server服务启动

打开 `server.py`, 想自定义自己的在线服务化只要简单替换以下两个字段

### schema替换
```python
# Default schema
schema = ['出发地', '目的地', '费用', '时间']

# Defined task schema
schema = [xxx]
```

### 设置模型路径
```
# Default task_path
uie = Taskflow('information_extration', task_path='../../../checkpoint/best_model/', schema=schema)

# Defined task_path
uie = Taskflow('information_extration', task_path='./path/xxx', schema=schema)
```

### 多卡服务化预测
PaddleNLP SimpleServing 支持多卡负载均衡预测，主要在服务化注册的时候，注册两个Taskflow的task即可，下面是示例代码
```
uie1 = Taskflow('information_extration', task_path='../../../checkpoint/best_model/', schema=schema, device_id=0)
uie2 = Taskflow('information_extration', task_path='../../../checkpoint/best_model/', schema=schema, device_id=1)
service.register_taskflow('uie', [uie1, uie2])
```

修改完上面的字段之后即可使用 `ppnlp-server` 启动服务
```shell
paddlenlp server server:app --host 0.0.0.0 --port 8989
```


## Client发送请求

```shell
python client.py
```

### 可以自定义输入文本
```python
# Changed to input texts you wanted
texts = ['城市内交通费7月5日金额114广州至佛山', '5月9日交通费29元从北苑到望京搜后']
```
### 设置相关参数
可以自己的需要来设置来 `batch_size` 和 `max_seq_len`
```python
data = {
    'data': {
        'text': texts
    },
    'parameters': {
        'max_seq_len': 512,
        'batch_size': 1
    }
}
```
