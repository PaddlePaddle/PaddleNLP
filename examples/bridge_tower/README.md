# BridgeTower

本项目是论文 ["BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning"](https://arxiv.org/abs/2206.08657)的 `PaddlePaddle`的实现.

## 模型介绍

具有双塔结构的视觉语言（VL）模型近年来主导了视觉语言表示学习。当前的VL模型要么使用轻量级的单模态编码器，并学习在深度交叉模态编码器中同时提取、对齐和融合两种模态，要么将最后一层的单模态表示从深度预训练的单模态编码器馈送到顶部交叉模态编码器。这两种方法都可能限制视觉语言表示学习并限制模型性能。在本文中，我们提出了BridgeTower，它引入了多个桥接层，在单模态编码器的顶层和跨模态编码器的每一层之间建立了连接。这实现了跨模态编码器中预训练的单模态编码器的不同语义级别的视觉和文本表示之间的有效自底向上的跨模态对齐和融合。BridgeTower通过仅400万张图像的预训练，在各种下游视觉语言任务上实现了最先进的性能。特别是，在VQAv2测试标准集上，BridgeTower实现了78.73%的准确率，在相同的预训练数据和几乎可以忽略的额外参数和计算成本的情况下，比之前的最先进模型METER高1.09%。值得注意的是，当进一步缩放模型时，BridgeTower实现了81.15%的准确率，超过了在数量级更大数据集上预先训练的模型。

![framework](https://user-images.githubusercontent.com/12107462/211287540-51c784b5-5399-49bf-9dd9-3c07fa2d5acf.jpeg)


## Checkpoints

- Pre-trained checkpoints on 4M data: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_pt_base.ckpt?sv=2020-10-02&st=2022-11-24T12%3A18%3A49Z&se=2027-11-25T12%3A18%3A00Z&sr=b&sp=r&sig=BJigddAMHfNUtQuTGH8bJUrzAO3LfaeSm48AXUqZngY%3D) and [LARGE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_pt_large.ckpt?sv=2020-10-02&st=2022-11-24T12%3A19%3A19Z&se=2027-11-25T12%3A19%3A00Z&sr=b&sp=r&sig=8yWqesQACrJSi0JMLIA0uAbNlMQKb653gOXjXjQuIW4%3D)
- Fine-tuned checkpoints for
  - Visual Question Answering on VQAv2: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_vqav2.ckpt?sv=2020-10-02&st=2022-11-24T12%3A16%3A38Z&se=2027-11-25T12%3A16%3A00Z&sr=b&sp=r&sig=t35v4kezDcSOm9Q9E767PhNGAQRsiYm%2FMSDgHIz%2Fvto%3D), [BASE(w/ VGQA)](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_vqav2_vgqa.ckpt?sv=2020-10-02&st=2022-11-24T12%3A17%3A18Z&se=2027-11-25T12%3A17%3A00Z&sr=b&sp=r&sig=BD%2BOsI%2F6R905vBJUlrWlgx3%2BmaBRsa2rQcHBChhW0eE%3D), [LARGE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_large_vqav2.ckpt?sv=2020-10-02&st=2022-11-24T12%3A17%3A47Z&se=2027-11-25T12%3A17%3A00Z&sr=b&sp=r&sig=RqL7Eeye4385oaO1nvVvRwC4d%2ByhpEVGM3xmS4GcKkQ%3D), [LARGE(w/ VGQA)](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_large_vqav2_vgqa.ckpt?sv=2020-10-02&st=2022-11-24T12%3A18%3A29Z&se=2027-11-25T12%3A18%3A00Z&sr=b&sp=r&sig=xtI8rmEqjMmN1b1bcE0KB9ePUax3SuRfOt%2Bp2ATH9ng%3D)
  - Image-Text Retrieval on Flickr30k: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_irtr_itm_itc_f30k.ckpt?sv=2020-10-02&st=2022-11-24T12%3A13%3A42Z&se=2027-11-25T12%3A13%3A00Z&sr=b&sp=r&sig=0BP3pOiE4AFkK4BTgQl5Dy6iJWxHuJffpjU4LFMTfWY%3D)
  - Visual Entailment on SNLI-VE: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_snlive.ckpt?sv=2020-10-02&st=2022-11-24T12%3A15%3A27Z&se=2027-11-25T12%3A15%3A00Z&sr=b&sp=r&sig=IccPmnxQYIpWO8m6kwtEFir9wmVq1SsLOqmw0FRc9hY%3D)
  - Visual Reasoning on NLVR$^2$: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_nlvr2.ckpt?sv=2020-10-02&st=2022-11-24T12%3A15%3A09Z&se=2027-11-25T12%3A15%3A00Z&sr=b&sp=r&sig=AL3q15eyhPBHaWY0FOop9goHVq8CbNluABDk%2FS94rkI%3D)
  - Image-Text Retrieval on MSCOCO: [BASE](https://chenfei.blob.core.windows.net/data/G/LCI/best_checkpoints/BridgeTower_ftfpt_base_irtr_itm_itc_coco.ckpt?sv=2020-10-02&st=2022-11-24T12%3A13%3A18Z&se=2027-11-25T12%3A13%3A00Z&sr=b&sp=r&sig=ahM%2FyI8fg9D4obCZsNKaxLzPVz2y8RX8ydZNToGavC4%3D)

## 代码导航

以下是本项目目录结构及说明：
```
├── configs
│   ├── pretrain_config.json # 预训练的配置
│   ├── snli_ve_config.json  # snlive的任务的配置
│   └── vqav2_config.json  # vqav2的任务配置
├── README.md              # 文档
├── requirements.txt       # 环境依赖
├── scripts
│   ├── ftfpt_base_snlive.sh # snlive任务的微调脚本
│   └── pre_train.sh         # 预训练脚本
├── src
│   ├── datasets
│   │   ├── base_dataset.py  # dataset基类
│   │   ├── coco_caption_karpathy_dataset.py # coco数据集
│   │   ├── conceptual_caption_dataset.py  # cc数据集
│   │   ├── __init__.py
│   │   ├── sbu_caption_dataset.py # sbu数据集
│   │   ├── snli_dataset.py # snli数据集
│   │   ├── vg_caption_dataset.py # vg数据集
│   │   └── vqav2_dataset.py # vqav2数据集
│   ├── gadgets
│   │   ├── __init__.py
│   │   ├── my_metrics.py # 评估函数
│   ├── modules
│   │   ├── bert_model.py # bert模型
│   │   ├── bt_module.py # BridgeTower模型的主函数
│   │   ├── clip_model.py # CLIP模型
│   │   ├── heads.py # 预训练的模型的head层
│   │   ├── __init__.py
│   │   ├── meter_utils.py # 评估指标，训练参数初始化
│   │   ├── objectives.py # 训练的loss计算
│   │   └── transformer.py # BertCrossLayer层
│   └── transforms
│       ├── __init__.py
│       ├── randaugment.py # 数据增强
│       ├── randaug.py # 数据增强
│       ├── transform.py # 数据增强
│       └── utils.py # 数据增强文件
├── torch2paddle.py # torch模型转换paddle的脚本
├── trainer_util.py # 预训练和微调的配置
├── train.py  # 训练文件
└── utils.py # 数据加载配置
```

# 开始运行

## 环境要求

* python 3.8
* 项目依赖 Paddle v2.4.1 版本，请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)安装升级。
* 项目依赖 opencv、einops 等包 ，可以使用如下命令完成安装

```
pip install -r requirements.txt
pip install paddlenlp --upgrade
```

## 数据准备

- We follow [ViLT](https://github.com/dandelin/ViLT) and use pyarrow to serialize the datasets. See [here](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.
- For SNLI-VE dataset, we follow [here](https://github.com/necla-ml/SNLI-VE).
- For VG-QA dataset, except the image-text pairs in [VG](https://visualgenome.org/api/v0/api_home.html) got from [here](https://github.com/dandelin/ViLT/blob/master/DATA.md), [image meta data](https://visualgenome.org/static/data/dataset/image_data_v1.json.zip), [question answers data](https://visualgenome.org/static/data/dataset/question_answers.json.zip) and [coco split information](https://github.com/peteanderson80/bottom-up-attention/tree/master/data/genome/coco_splits) also need to be downloaded.

数据预处理参考上述的链接即可，最终会生成下面所示的arraw文件：

```
dataset/
├── fine-tune
│   ├── snli_ve_dev.arrow
│   ├── snli_ve_test.arrow
│   └── snli_ve_train.arrow
└── pre-train
    ├── cc_0.arrow
    ├── cc_10.arrow
    ├── cc_11.arrow
    ├── cc_12.arrow
    ├── cc_13.arrow
    ├── cc_14.arrow
    ├── cc_15.arrow
    ├── cc_16.arrow
    ├── cc_17.arrow
    ├── cc_18.arrow
    ├── cc_19.arrow
    .....
    ├── sbu_6.arrow
    ├── sbu_7.arrow
    ├── sbu_8.arrow
    ├── vg_albef.arrow
    └── vg.arrow
```


## 预训练

预训练采用16卡的A100进行分布式训练，命令如下：

```
python -u -m paddle.distributed.launch --ips 10.78.122.11,10.78.115.13 \
                 --gpus "0,1,2,3,4,5,6,7" \
                python train.py --output_dir pretrain_output \
                --do_train \
                --do_eval \
                --evaluation_strategy epoch \
                --learning_rate 1.0 \
                --eval_steps 1000 \
                --warmup_ratio 0.1 \
                --save_steps 1000 \
                --dataloader_num_workers 8 \
                --logging_steps 50 \
                --num_nodes 1 \
                --num_gpus 8 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 128 \
                --batch_size 128 \
                --num_train_epochs 10 \
                --weight_decay 0.01 \
                --save_total_limit 50 \
                --adam_epsilon 1e-8 \
                --adam_beta1 0.9 \
                --adam_beta2 0.98 \
                --lr_scheduler_type polynomial \
                --lr_end 0 \
                --power 1 \
                --seed 1 \
                --max_grad_norm -1 \
                --max_steps 100000 \
                --label_names text_labels \
                --data_root /root/paddlejob/workspace/env_run/afs/laion400m_new/wugaosheng/dataset/pre-train \
                --config_name configs/pretrain_config.json
```
其中参数释义如下：
- `do_train` 训练控制变量。
- `do_eval` 评估控制变量。
- `evaluation_strategy` 模型评估的间隔策略。若为`epoch`，则每轮训练结束后评估模型。
- `learning_rate` 模型的学习率。
- `eval_steps` 评估模型的step间隔。
- `warmup_ratio` 模型的warmup的比率。
- `save_steps` 保存模型间隔。
- `dataloader_num_workers` dataloader的worker数目。
- `logging_steps` 打印日志的步数。
- `num_nodes` 分布式的机器数目。
- `num_gpus` 每台机器的gpu的卡数。
- `per_device_train_batch_size` 每次训练每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `per_device_eval_batch_size`: 每次评估每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `batch_size` 所有卡的训练的batch_size总数。
- `num_train_epochs` 表示训练轮数。
- `weight_decay` 表示优化器中使用的weight_decay的系数。
- ``save_total_limit`` 保存的ckpt数量的最大限制。
- `adam_epsilon` Adam优化器的参数，避免分母为零，默认为1e-8。
- `adam_beta1` Adam优化器的参数。
- `adam_beta2` Adam优化器的参数。
- `lr_scheduler_type` warmup的学习率变化类型。
- `lr_end` polynomial 终止学习率。
- `seed` 模型的随机种子。
- `max_grad_norm` 详见ClipGradByGlobalNorm参数，-1表示不使用ClipGradByGlobalNorm。
- `max_steps` 表示最大训练步数。若训练`num_train_epochs`轮包含的训练步数大于该值，则达到`max_steps`后就提前结束。
- `label_names` 训练数据标签label的名称，本项目设置为text_labels，默认值为None。
- `data_root` 预训练数据arrow文件的目录。
- `config_name`: 预训练配置文件。
- `power` polynomial 每次学习率衰减系数。

参数配置比较多，也可以直接执行下面的脚本：

```bash
# Pre-train BridgeTower Base Model
bash scripts/pre_train.sh

```

## 微调训练

snlive任务的微调，下面的配置默认使用的是8卡a100，命令如下：

```
log_dir=train_log
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
                --log_dir ${log_dir}  \
                train.py --output_dir output_pd \
                --do_eval \
                --do_train \
                --learning_rate 1.0 \
                --warmup_ratio 0.06 \
                --logging_steps 50 \
                --eval_steps 100 \
                --evaluation_strategy epoch \
                --num_nodes 1 \
                --num_gpus 8 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 64 \
                --batch_size 64 \
                --dataloader_num_workers 8 \
                --save_steps 8000 \
                --num_train_epochs 5 \
                --weight_decay 0.01 \
                --save_total_limit 50 \
                --adam_epsilon 1e-8 \
                --adam_beta1 0.9 \
                --adam_beta2 0.98 \
                --seed 1 \
                --data_root /root/paddlejob/workspace/env_run/afs/laion400m_new/wugaosheng/dataset/fine-tune \
                --lr_scheduler_type polynomial \
                --checkpoint_path ./checkpoints/checkpoint-40000/model_state.pdparams \
                --lr_end 0 \
                --power 1 \
                --max_grad_norm -1 \
                --config_name configs/snli_ve_config.json
```

这里面的参数解释请参考预训练部分，另外额外的参数释义如下：

- `checkpoint_path` 加载预训练模型的路径。

参数配置比较多，也可以直接执行下面的脚本：
```
# Base Model on SNLI-VE with VLP
bash scripts/ftfpt_base_snlive.sh
```


## Citation

```
@article{xu2022bridge,
  title={BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning},
  author={Xu, Xiao and Wu, Chenfei and Rosenman, Shachar and Lal, Vasudev and Che, Wanxiang and Duan, Nan},
  journal={arXiv preprint arXiv:2206.08657},
  year={2022}
}
```

## Acknowledgement

非常感谢论文的开源代码，我们的代码基于开源的代码进行了paddle转换:

- Main Code: [BridgeTower](https://github.com/microsoft/BridgeTower)
