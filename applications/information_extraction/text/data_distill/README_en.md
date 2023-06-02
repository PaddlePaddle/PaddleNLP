# UIE Slim data distillation

While UIE has powerful zero-shot extraction capabilities, its prompting structure requires significant compute to serve in real time. Some industrial application scenarios have high inference performance requirements and the model cannot go into production without being effectively compressed. We built the UIE Slim Data Distillation with knowledge distillation techniques. The principle is to use the data as a bridge to transfer the knowledge of the UIE model to the smaller closed-domain information extraction model in order to achieve speedup inference significantly with minimal loss to accuracy.

#### Three steps of UIE data distillation

- **Step 1**: Finetune the UIE model on the labeled data to get the Teacher Model.

- **Step 2**: Process the user-provided unlabeled data and run inference with Taskflow UIE.

- **Step 3**: Use the labeled data and the inference results obtained in step 2 to train a closed-domain Student Model.

## UIE Finetune

Refer to [UIE relationship extraction fine-tuning](../README.md) to complete the model fine-tuning and get ``../checkpoint/model_best``.

## Offline Distillation

#### Predict the label of unsupervised data through the trained UIE custom model

```shell
python data_distill.py \
     --data_path ../data \
     --save_dir student_data \
     --task_type relation_extraction \
     --synthetic_ratio 10 \
     --model_path ../checkpoint/model_best
```

**NOTE**: The schema needs to be configured in `data_distill.py` according to the label data, and the schema needs to contain all label types in the label data.

Description of configurable parameters:

- `data_path`: Path to labeled data (`doccano_ext.json`) and unsupervised text (`unlabeled_data.txt`).
- `model_path`: The path of the trained UIE custom model.
- `save_dir`: The path to save the training data of the student model.
- `synthetic_ratio`: Controls the ratio of synthetic data. The maximum number of synthetic data=synthetic_ratio*number of labeled data.
- `platform`: The labeling platform used to label data, optional are `doccano`, `label_studio`, the default is `label_studio`.
- `task_type`: Select the task type, optional are `entity_extraction`, `relation_extraction`, `event_extraction` and `opinion_extraction`. Because it is a closed-domain extraction, the post-processing logic of different tasks is different, so the task type needs to be specified.
- `seed`: random seed, default is 1000.

#### Teacher model evaluation

In the UIE fine-tuning stage, the model performance is evaluated on UIE training format data, which is not a standard end-to-end evaluation method for relation extraction or event extraction. The end-to-end evaluation can be performed through the following evaluation script.

```shell
python evaluate_teacher.py \
     --task_type relation_extraction \
     --test_path ./student_data/dev_data.json\
     --label_maps_path ./student_data/label_maps.json \
     --model_path ../checkpoint/model_best
```

Description of configurable parameters:

- `model_path`: The path of the trained UIE custom model.
- `test_path`: test dataset path.
- `label_maps_path`: dictionary of student model labels.
- `batch_size`: batch size, default is 8.
- `max_seq_len`: Maximum text length, default is 256.
- `task_type`: Select the task type, optional are `entity_extraction`, `relation_extraction`, `event_extraction` and `opinion_extraction`. Because it is an evaluation of closed-domain information extraction, the task type needs to be specified.


#### Student model training

```shell
python train.py\
     --task_type relation_extraction \
     --train_path student_data/train_data.json \
     --dev_path student_data/dev_data.json \
     --label_maps_path student_data/label_maps.json \
     --num_epochs 50 \
     --encoder ernie-3.0-mini-zh
```

Description of configurable parameters:

- `train_path`: training set file path.
- `dev_path`: Validation set file path.
- `batch_size`: batch size, default is 16.
- `learning_rate`: Learning rate, default is 3e-5.
- `save_dir`: model storage path, the default is `./checkpoint`.
- `max_seq_len`: Maximum text length, default is 256.
- `weight_decay`: Indicates the coefficient of weight_decay used in the AdamW optimizer.
- `warmup_proportion`: The proportion of the learning rate warmup strategy. If it is 0.1, the learning rate will slowly increase from 0 to learning_rate during the first 10% training step, and then slowly decay. The default is 0.0.
- `num_epochs`: The number of training epochs, the default is 100.
- `seed`: random seed, default is 1000.
- `encoder`: select the model base of the student model, the default is `ernie-3.0-mini-zh`.
- `task_type`: Select the task type, optional are `entity_extraction`, `relation_extraction`, `event_extraction` and `opinion_extraction`. Because it is closed-domain information extraction, the task type needs to be specified.
- `logging_steps`: The interval steps of log printing, the default is 10.
- `eval_steps`: The interval steps of evaluate, the default is 200.
- `device`: What device to choose for training, optional cpu or gpu.
- `init_from_ckpt`: optional, model parameter path, hot start model training; default is None.

#### Student model evaluation

```shell
python evaluate.py \
     --model_path ./checkpoint/model_best \
     --test_path student_data/dev_data.json \
     --task_type relation_extraction \
     --label_maps_path student_data/label_maps.json \
     --encoder ernie-3.0-mini-zh
```

Description of configurable parameters:

- `model_path`: The path of the trained UIE custom model.
- `test_path`: test dataset path.
- `label_maps_path`: dictionary of student model labels.
- `batch_size`: batch size, default is 8.
- `max_seq_len`: Maximum text length, default is 256.
- `encoder`: select the model base of the student model, the default is `ernie-3.0-mini-zh`.
- `task_type`: Select the task type, optional are `entity_extraction`, `relation_extraction`, `event_extraction` and `opinion_extraction`. Because it is an evaluation of closed-domain information extraction, the task type needs to be specified.

## Student model deployment

- Fast deployment of the closed-domain information extraction model through Taskflow, `task_path` is the path of the student model.

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> my_ie = Taskflow("information_extraction", model="uie-data-distill-gp", task_path="checkpoint/model_best/") # Schema is fixed in closed-domain information extraction
>>> pprint(my_ie("Virgo deceleration bomb was developed by the Swedish FFV Ordnance Company specially for the attack aircraft of the Swedish Royal Air Force to carry out low-altitude and high-speed bombing. It was developed in 1956 and entered service in 1963. It is equipped on the A32 "Contradiction", A35 "Dragon", and AJ134 "Thunder" attack aircraft are mainly used to attack landing craft, parked aircraft, anti-aircraft artillery, field artillery, light armored vehicles and active forces."))
[{'weapon name': [{'end': 14,
             'probability': 0.9976037,
             'relations': {'country of origin': [{'end': 18,
                                   'probability': 0.9988706,
                                   'relations': {},
                                   'start': 16,
                                   'text': 'Sweden'}],
                           'R&D unit': [{'end': 25,
                                     'probability': 0.9978277,
                                     'relations': {},
                                     'start': 18,
                                     'text': 'FFV Ordnance Company'}],
                           'type': [{'end': 14,
                                   'probability': 0.99837446,
                                   'relations': {},
                                   'start': 12,
                                   'text': 'bomb'}]},
             'start': 0,
             'text': 'Virgo slowing bomb'}]}]
```


# References

- **[GlobalPointer](https://kexue.fm/search/globalpointer/)**

- **[GPLinker](https://kexue.fm/archives/8888)**

- **[JunnYu/GPLinker_pytorch](https://github.com/JunnYu/GPLinker_pytorch**
