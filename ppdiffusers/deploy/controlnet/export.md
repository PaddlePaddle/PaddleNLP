# ControlNet 模型导出教程


[PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers) 是一款支持跨模态（如图像与语音）训练和推理的扩散模型（Diffusion Model）工具箱，其借鉴了🤗 Huggingface 团队的 [Diffusers](https://github.com/huggingface/diffusers) 的优秀设计，并且依托 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) 框架和 [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) 自然语言处理库。下面将介绍如何将 PPDiffusers 提供的预训练模型进行模型导出。

### 模型导出

可执行以下命令行完成模型导出。

```shell
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --controlnet_pretrained_model_name_or_path  lllyasviel/sd-controlnet-canny --output_path control_sd15_canny --height=512 --width=512
```
注: 上述指令导出固定尺寸的模型，固定尺寸的导出模型有利于优化模型推理性能，但会牺牲一定灵活性。若要导出支持多种推理尺寸的模型，可取消参数--height和--width的设置。

输出的模型目录结构如下：

```shell
control_sd15_canny/
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


`export_model.py` 各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
| <span style="display:inline-block;width: 230pt"> --pretrained_model_name_or_path </span> | ppdiffuers提供的diffusion预训练模型。默认为："runwayml/stable-diffusion-v1-5"。更多 StableDiffusion 预训练模型可参考 [ppdiffusers 模型列表](../README.md#ppdiffusers模型支持的权重)。|
| <span style="display:inline-block;width: 230pt"> --controlnet_pretrained_model_name_or_path </span> | ppdiffuers提供的controlnet预训练模型。默认为："lllyasviel/sd-controlnet-canny"。更多 ControlNET 预训练模型可参考 [lllyasviel的huggingface hub](https://huggingface.co/lllyasviel)。|
| --output_path | 导出的模型目录。 |
| --sample | vae encoder 的输出是否调整为 sample 模式，注意：sample模式会引入随机因素，默认是 False。|
| --height | 如果指定，则会固定导出模型的高度，即，在推理生成图片时只能生成该大小的图片，默认值为None。|
| --width | 如果指定，则会固定导出模型的宽度，即，在推理生成图片时只能生成该大小的图片，默认值为None。|
