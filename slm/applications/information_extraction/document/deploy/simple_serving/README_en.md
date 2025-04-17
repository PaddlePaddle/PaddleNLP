# Service deployment based on PaddleNLP SimpleServing

## Table of contents
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

## Service custom parameters

### Server Custom Parameters

#### schema replacement
```python
# Default schema
schema = ['Billing Date', 'Name', 'Taxpayer Identification Number', 'Account Bank and Account Number', 'Amount', 'Total Price and Tax', 'No', 'Tax Rate', 'Address, Phone', 'tax']
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
# Changed to image paths you wanted
image_paths = ['../../data/images/b1.jpg']
```
