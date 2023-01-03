# 基于PaddleNLP SimpleServing 的服务化部署

## 目录
- [环境准备](#环境准备)
- [Server服务启动](#Server服务启动)
- [Client请求启动](#Client请求启动)
- [服务化自定义参数](#服务化自定义参数)

## 环境准备
使用有SimpleServing功能的PaddleNLP版本(或者最新的develop版本)

```shell
pip install paddlenlp >= 2.4.4
```


## Server服务启动

```bash
paddlenlp server server:app --workers 1 --host 0.0.0.0 --port 8189
```

## Client请求启动

```bash
python client.py
```

## 服务化自定义参数

### Server 自定义参数
#### schema替换
```python
# Default schema
schema = {"武器名称": ["产国", "类型", "研发单位"]}
```

#### 设置模型路径
```
# Default task_path
uie = Taskflow('information_extration', model='uie-data-distill-gp', task_path='../../checkpoint/model_best/', schema=schema)
```

#### 多卡服务化预测
PaddleNLP SimpleServing 支持多卡负载均衡预测，主要在服务化注册的时候，注册两个Taskflow的task即可，下面是示例代码
```
uie1 = Taskflow('information_extration', model='uie-data-distill-gp', task_path='../../checkpoint/model_best/', schema=schema, device_id=0)
uie2 = Taskflow('information_extration', model='uie-data-distill-gp', task_path='../../checkpoint/model_best/', schema=schema, device_id=1)
service.register_taskflow('uie', [uie1, uie2])
```

### Client 自定义参数

```python
# Changed to input texts you wanted
texts = ['威尔哥（Virgo）减速炸弹是由瑞典FFV军械公司专门为瑞典皇家空军的攻击机实施低空高速轰炸而研制，1956年开始研制，1963年进入服役，装备于A32“矛盾”、A35“龙”、和AJ134“雷”攻击机，主要用于攻击登陆艇、停放的飞机、高炮、野战火炮、轻型防护装甲车辆以及有生力量。']

```
