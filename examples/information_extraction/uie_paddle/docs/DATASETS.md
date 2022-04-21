# 数据说明

``` json
{
  "text": "MULTAN , Pakistan , April 27 ( AFP )",
  "tokens": ["MULTAN", ",", "Pakistan", ",", "April", "27", "(", "AFP", ")"],
  "record": "<extra_id_0> <extra_id_0> geographical social political <extra_id_5> MULTAN <extra_id_0> part whole <extra_id_5> Pakistan <extra_id_1> <extra_id_1> <extra_id_0> geographical social political <extra_id_5> Pakistan <extra_id_1> <extra_id_0> organization <extra_id_5> AFP <extra_id_1> <extra_id_1>",
  "entity": [
    {"type": "geographical social political", "offset": [0], "text": "MULTAN"},
    {"type": "geographical social political", "offset": [2], "text": "Pakistan"},
    {"type": "organization", "offset": [7], "text": "AFP"}
  ],
  "relation": [
    {
      "type": "part whole",
      "args": [
        {"type": "geographical social political", "offset": [0], "text": "MULTAN"},
        {"type": "geographical social political", "offset": [2], "text": "Pakistan"}
      ]
    }
  ],
  "event": [],
  "spot": ["geographical social political", "organization"],
  "asoc": ["part whole"],
  "spot_asoc": [
    {
      "span": "MULTAN",
      "label": "geographical social political",
      "asoc": [["part whole", "Pakistan"]]
    },
    {
      "span": "Pakistan",
      "label": "geographical social political", "asoc": []
    },
    {
      "span": "AFP", "label": "organization", "asoc": []
    }
  ],
  "task": 'record'
}
```

- task: `seq`, `record`, `t5mlm`
  - mlm 只要求有 Text
  - seq 只要求有 Record
  - record 要求有 Text-record 数据
  - 若无，默认为 Text-record 数据
- spot、asoc
  - 文本中的正例类别
- spot_asoc
  - record 结构表示
- entity relation event
  - Offset 标准答案，用于模型验证。
