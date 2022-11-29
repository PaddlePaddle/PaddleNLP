# Diffusion模型导出教程


[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)是一款支持跨模态（如图像与语音）训练和推理的扩散模型（Diffusion Model）工具箱，其借鉴了🤗 Huggingface团队的[Diffusers](https://github.com/huggingface/diffusers)的优秀设计，并且依托[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)框架和[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)自然语言处理库。下面将介绍如何将PPDiffusers提供的预训练模型进行模型导出。

### 模型导出

___注意：模型导出过程中，需要下载StableDiffusion模型。为了使用该模型与权重，你必须接受该模型所要求的License，请访问HuggingFace的[model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), 仔细阅读里面的License，然后签署该协议。___

___Tips: Stable Diffusion是基于以下的License: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___

可执行以下命令行完成模型导出。

```shell
python export_model.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --output_path stable-diffusion-v1-4
```

输出的模型目录结构如下：
```shell
stable-diffusion-v1-4/
├── text_encoder
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── unet
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
└── vae_decoder
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

#### 参数说明

`export_model.py` 各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|<div style="width: 230pt">--pretrained_model_name_or_path </div> | ppdiffuers提供的diffusion预训练模型。默认为："CompVis/stable-diffusion-v1-4    "。更多diffusion预训练模型可参考[ppdiffuser模型列表](../examples/textual_inversion/README.md)。|
|--output_path | 导出的模型目录。 |
