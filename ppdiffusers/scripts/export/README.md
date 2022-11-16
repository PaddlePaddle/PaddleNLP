# Stable Diffusion模型导出教程

- [注意事项](#注意事项)
- [环境依赖](#环境依赖)
- [模型导出](#模型导出)
  - [参数说明](#参数说明)
- [推理部署](#推理部署)

## 注意事项

___注意：模型导出过程中，需要下载StableDiffusion模型。为了使用该模型与权重，你必须接受该模型所要求的License，请访问HuggingFace的[model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), 仔细阅读里面的License，然后签署该协议。___

___Tips: Stable Diffusion是基于以下的License: The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___

## 环境依赖

- paddlepaddle >= 2.4.0
- paddlenlp >= 2.4.1
- ppdiffusers >= 0.6.2

可执行以下命令行安装环境依赖包。

```shell
pip install --upgrade ppdiffusers paddlepaddle-gpu paddlenlp
```


## 模型导出

可执行以下命令行完成模型导出。

```shell
python export_model.py --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 --output_path stable-diffusion-v1-4
```

如需导出stable-diffusion-v1-5，可执行以下命令：

```shell
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path stable-diffusion-v1-5
```


输出的模型目录结构如下：
```shell
stable-diffusion-v1-4/
├── text_encoder
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── unet
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
└── vae_decoder
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

### 参数说明

`export_model.py` 各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|<div style="width: 230pt">--pretrained_model_name_or_path </div> | ppdiffuers提供的diffusion预训练模型名称以及用户自行训练的模型目录。默认为："CompVis/stable-diffusion-v1-4    "。更多diffusion预训练模型可参考[ppdiffuser模型列表](../examples/textual_inversion)。|
|--output_path | 导出的模型目录。 |


## 推理部署

完成模型导出后，可以加载导出后的模型，完成StableDiffusion的模型部署。我们提供在[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)上的StableDiffusion模型文生图任务的部署示例，可参考[FastDeploy Diffusion模型高性能部署](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/multimodal/stable_diffusion#%E5%BF%AB%E9%80%9F%E4%BD%93%E9%AA%8C)完成部署。
