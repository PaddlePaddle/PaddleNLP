# UIE数据蒸馏

## 1.数据蒸馏

- 合成封闭域训练及评估数据

```shell
python data_generate.py --data_dir ../law_data --output_dir law_distill --task_type relation_extraction --synthetic_ratio 1
```

## 2.封闭域训练

```shell
python train.py --task_type relation_extraction --train_path law_distill/train_data.json --dev_path law_distill/dev_data.json --label_maps_path law_distill/label_maps.json
```

## 3.Taskflow装载

```shell
from paddlenlp import Taskflow

ie = Taskflow("information_extraction", model="uie-data-distill-gp", task_path="checkpoint/model_best/")
```
