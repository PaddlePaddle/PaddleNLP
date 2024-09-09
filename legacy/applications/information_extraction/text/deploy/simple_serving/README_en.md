# Service deployment based on PaddleNLP SimpleServing

- [Environment Preparation](#1)
- [Server](#2)
- [Client](#3)
- [Service Custom Parameters](#4)

<a name="1"></a>

## Environment Preparation
Use the PaddleNLP version with SimpleServing function (or the latest develop version)

```shell
pip install paddlenlp >= 2.4.4
```

<a name="2"></a>

## Server

```bash
paddlenlp server server:app --workers 1 --host 0.0.0.0 --port 8189
```

<a name="3"></a>

## Client

```bash
python client.py
```

<a name="4"></a>

## Service Custom Parameters

### Server Custom Parameters

#### schema replacement
```python
# Default schema
schema = {"Weapon Name": ["Country of Production", "Type", "R&D Unit"]}
```

#### Set model path
```
# Default task_path
uie = Taskflow('information_extraction', task_path='../../checkpoint/model_best/', schema=schema)
```

#### Doka Service Prediction
PaddleNLP SimpleServing supports multi-card load balancing prediction, mainly during service registration, just register two Taskflow tasks, the following is the sample code
```
uie1 = Taskflow('information_extraction', task_path='../../checkpoint/model_best/', schema=schema, device_id=0)
uie2 = Taskflow('information_extraction', task_path='../../checkpoint/model_best/', schema=schema, device_id=1)
service. register_taskflow('uie', [uie1, uie2])
```

### Client Custom Parameters

```python
# Changed to input texts you wanted
texts = ['威尔哥（Virgo）减速炸弹是由瑞典FFV军械公司专门为瑞典皇家空军的攻击机实施低空高速轰炸而研制，1956年开始研制，1963年进入服役，装备于A32“矛盾”、A35“龙”、和AJ134“雷”攻击机，主要用于攻击登陆艇、停放的飞机、高炮、野战火炮、轻型防护装甲车辆以及有生力量。']
```
