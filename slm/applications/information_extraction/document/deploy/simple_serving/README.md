# 基于 PaddleNLP SimpleServing 的服务化部署

## 目录
- [环境准备](#环境准备)
- [Server 服务启动](#Server 服务启动)
- [Client 请求启动](#Client 请求启动)
- [服务化自定义参数](#服务化自定义参数)

## 环境准备
使用有 SimpleServing 功能的 PaddleNLP 版本(或者最新的 develop 版本)

```shell
pip install paddlenlp >= 2.4.4
```


## Server 服务启动

```bash
paddlenlp server server:app --workers 1 --host 0.0.0.0 --port 8189
```

## Client 请求启动

```bash
python client.py
```

## 服务化自定义参数

### Server 自定义参数
#### schema 替换
```python
# Default schema
schema = ['开票日期', '名称', '纳税人识别号', '开户行及账号', '金额', '价税合计', 'No', '税率', '地址、电话', '税额']
```

#### 设置模型路径
```
# Default task_path
uie = Taskflow('information_extraction', task_path='../../checkpoint/model_best/', schema=schema)
```

#### 多卡服务化预测
PaddleNLP SimpleServing 支持多卡负载均衡预测，主要在服务化注册的时候，注册两个 Taskflow 的 task 即可，下面是示例代码
```
uie1 = Taskflow('information_extraction', task_path='../../checkpoint/model_best/', schema=schema, device_id=0)
uie2 = Taskflow('information_extraction', task_path='../../checkpoint/model_best/', schema=schema, device_id=1)
service.register_taskflow('uie', [uie1, uie2])
```

### Client 自定义参数

```python
# Changed to image paths you wanted
image_paths = ['../../data/images/b1.jpg']
```
