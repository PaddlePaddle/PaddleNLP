# Diffusion 模型导出教程


[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers) 是一款支持跨模态（如图像与语音）训练和推理的扩散模型（Diffusion Model）工具箱，其借鉴了🤗 Huggingface 团队的 [Diffusers](https://github.com/huggingface/diffusers) 的优秀设计，并且依托 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) 框架和 [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) 自然语言处理库。下面将介绍如何将 PPDiffusers 提供的预训练模型进行模型导出。

### 模型导出

___注意：模型导出过程中，需要下载 StableDiffusion 模型。为了使用该模型与权重，你必须接受该模型所要求的 License，请访问 HuggingFace 的[model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), 仔细阅读里面的 License，然后签署该协议。___

___Tips: Stable Diffusion 是基于以下的 License: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___

可执行以下命令行完成模型导出。

```shell
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path stable-diffusion-v1-5 --height=512 --width=512
```
注: 上述指令导出固定尺寸的模型，固定尺寸的导出模型有利于优化模型推理性能，但会牺牲一定灵活性。若要导出支持多种推理尺寸的模型，可取消参数--height和--width的设置。

输出的模型目录结构如下：

```shell
stable-diffusion-v1-5/
├── model_index.json
├── scheduler
│   └── scheduler_config.json
├── tokenizer
│   ├── tokenizer_config.json
│   ├── merges.txt
│   ├── vocab.json
│   └── special_tokens_map.json
├── text_encoder
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── unet
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── vae_decoder
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
└── vae_encoder
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

#### Inpaint 任务模型导出

除了支持常规 StableDiffusion 文生图、图生图任务的模型导出以外，还支持Inpaint任务模型 (注意：这个不是 legacy 版本的 inpaint) 的导出、如果需要导出 inpaint 模型，可以执行以下命令：

```shell
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting --output_path stable-diffusion-v1-5-inpainting
```

#### 参数说明

`export_model.py` 各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
| <span style="display:inline-block;width: 230pt"> --pretrained_model_name_or_path </span> | ppdiffuers提供的diffusion预训练模型。默认为："CompVis/stable-diffusion-v1-4"。更多 StableDiffusion 预训练模型可参考 [ppdiffusers 模型列表](../README.md#ppdiffusers模型支持的权重)。|
| --output_path | 导出的模型目录。 |
| --sample | vae encoder 的输出是否调整为 sample 模式，注意：sample模式会引入随机因素，默认是 False。|
