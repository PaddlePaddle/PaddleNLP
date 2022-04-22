# Dataset Format

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
  ]
}
```

- text: raw input text
- tokens: tokens for evaluation
- record: the spot-asoc style structured extraction language instance
- spot、asoc
  - Positive type of single task fine-tuning
  - All types of specific task in the multi-task learning
- spot_asoc: target structure in the spot-asoc style
- entity/relation/event: Gold answer with offset for model evaluation
