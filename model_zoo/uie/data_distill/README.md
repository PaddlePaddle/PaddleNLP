# UIE Slim 数据蒸馏

在UIE强大的抽取能力背后，是需要同样强大的算力才能支撑起如此大规模模型的训练和预测。很多工业应用场景对性能要求较高，若不能有效压缩则无法实际应用。因此，我们基于数据蒸馏技术构建了UIE Slim数据蒸馏系统。其原理是通过数据作为桥梁，将UIE模型的知识迁移到小模型，以达到精度损失较小的情况下却能达到大幅度预测速度提升的效果。

#### UIE数据蒸馏三步

- **Step 1**: 使用UIE模型对标注数据进行fine-tune，得到Teacher Model。

- **Step 2**: 用户提供大规模无标注数据，需与标注数据同源。使用Taskflow UIE对无监督数据进行预测。

- **Step 3**: 使用标注数据以及步骤2得到的合成数据训练出Student Model。


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

## 实验效果

5-shot

5-shot + UIE数据蒸馏

full-shot

# References

- **[GlobalPointer](https://kexue.fm/search/globalpointer/)**

- **[GPLinker](https://kexue.fm/archives/8888)**

- **[JunnYu/GPLinker_pytorch](https://github.com/JunnYu/GPLinker_pytorch)**
