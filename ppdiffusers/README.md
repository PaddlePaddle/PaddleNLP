# PPDiffusers: Diffusers toolbox implemented based on PaddlePaddle

**PPDiffusers**是一款支持**跨模态**（如图像与语音）训练和推理的**扩散模型**（Diffusion Model）工具箱，我们借鉴了🤗 Huggingface团队的 [**Diffusers**](https://github.com/huggingface/diffusers) 的优秀设计，并且依托 [**PaddlePaddle**](https://www.paddlepaddle.org.cn/) 框架和 [**PaddleNLP**](https://github.com/PaddlePaddle/PaddleNLP) 自然语言处理库，打造了一款国产化的工具箱。

## News 📢

* 🔥 **2022.12.06 发布 0.9.0 版本，支持 [StableDiffusion2.0](https://github.com/Stability-AI/stablediffusion) 的文生图、图生图、图像编辑及图像超分等功能；**

* 🔥 **2022.11.11 发布 0.6.2 版本，支持使用FastDeploy对 [StableDiffusion进行高性能部署](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/deploy/README.md)、支持 [Diffusers或原版模型->PPDiffusers权重转换](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/scripts/convert_diffusers_model/README.md)；**

* 🔥 **2022.11.04 支持 IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 和 IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1 中文权重；**

* 🔥 **2022.10.27 发布 PPDiffusers仓库**。


<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#模型部署> 模型部署</a>
</h4>

## 特性


#### <a href=#sota扩散模型pipelines集合> 📦 SOTA扩散模型Pipelines集合 </a>

#### <a href=#提供丰富的noise-scheduler> 🤗 提供丰富的Noise Scheduler </a>

#### <a href=#提供多种diffusion模型组件> 🎛️ 提供多种Diffusion模型组件 </a>

#### <a href=#提供丰富的训练和推理教程> 🚀 提供丰富的训练和推理教程 </a>


### SOTA扩散模型Pipelines集合
**最先进（State-of-the-art）** 的 扩散模型（Diffusion Model）管道（Pipelines）集合。
当前**PPDiffusers**已经集成了**33+Pipelines**，不仅支持 Stable Diffusion [文生图Pipeline](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)，还支持基于[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)的[🔥高性能文生图Pipeline](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion/pipeline_fastdeploy_stable_diffusion.py)。
如果想要了解当前所支持的所有 **Pipelines** 以及对应的论文信息，我们可以阅读[🔥这里](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/pipelines/README.md)的文档。
```python
from ppdiffusers import StableDiffusionPipeline, FastDeployStableDiffusionPipeline

orig_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
fd_pipe = FastDeployStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5@fastdeploy")
```

### 提供丰富的Noise Scheduler
我们提供了丰富的**噪声调度器（Noise Scheduler）**，我们可以权衡**速度**与**质量**，在推理过程中根据需求快速切换使用。
当前**PPDiffusers**已经集成了**14+Scheduler**，不仅支持 [DDPM](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_ddpm.py)、[DDIM](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_ddim.py) 和 [PNDM](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_pndm.py)，还支持最新的 [🔥DPMSolver](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_dpmsolver_multistep.py)！

```python
from ppdiffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler

ddpm_scheduler = DDPMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    steps_offset=1,
)
ddim_scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
dpmsolver_scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)
```

### 提供多种Diffusion模型组件
我们提供了**多种 Diffusion 模型**组件，如[UNet1d](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/models/unet_1d.py)、[UNet2d](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/models/unet_2d.py)、[UNet2d Conditional](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/models/unet_2d_conditional.py)。


### 提供丰富的训练和推理教程
我们提供了丰富的训练教程，不仅支持扩散模型的二次开发，如 [Textual Inversion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/textual_inversion) 和 [DreamBooth](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/dreamBooth)  使用 3-5 张图定制化训练自己的风格或物体。还支持使用 [Laion400M](https://laion.ai/blog/laion-400-open-dataset) 数据集 [🔥从零训练Latent Diffusion Model](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/text_to_image_laion400m) 模型！
此外，我们还提供了 [使用Paddle动态图推理的教程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference) 以及 [🔥高性能FastDeploy推理教程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/deploy)。


## 安装

### 环境依赖
- paddlepaddle-gpu>=2.4.0
- paddlenlp>=2.4.4
- ftfy
- regex
- Pillow

### pip安装

```shell
pip install --upgrade ppdiffusers
```

更多关于PaddlePaddle安装的详细教程请查看[Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)。

### 手动安装
```shell
git clone https://github.com/PaddlePaddle/PaddleNLP
# 注意：如果clone仓库非常慢的话，可以考虑使用镜像版本
# git clone https://gitee.com/paddlepaddle/PaddleNLP
cd PaddleNLP/ppdiffusers
python setup.py install
```

## 快速开始

为了快速上手使用该项目, 我们可以先阅读🤗 Huggingface团队提供的两个Notebook教程 [Getting started with Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) 和 [Training a diffusers model](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)。（Tips：国内用户可能无法正常打开）

**Stable Diffusion 1.x** 是一个**文本到图像（text-to-image）**的**潜在扩散模型(Latent Diffusion Model, LDM)**, 该模型是由来自 [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/), [LAION](https://laion.ai/) 的工程师以及 [RunwayML](https://runwayml.com/)一起开发而完成的。该模型使用了大小为 **512x512** 分辨率的 [LAION-5B](https://laion.ai/blog/laion-5b/) 数据集子集进行训练。该模型使用了 **Openai** 开源的 **CLIP ViT-L/14** 文本编码器（约**123M**参数）来编码提示（prompt）文本（注意该部分权重不进行训练）。该模型还使用了**UNet2d Conditional**模型（约**860M**参数）来建模扩散过程。

**Stable Diffusion 2.0** 由 [LAION](https://laion.ai/) 在 [Stability AI](https://stability.ai/) 的支持下开发完成的，它与早期的 **V1** 版本相比，大大改善了生成图像的质量。该版本中的文生图模型不仅可以生成默认分辨率为 **512x512** 像素还可以生成 **768x768** 分辨率的图像。该模型作为 **Stable Diffusion 1.x** 的升级版, 使用了全新的 [OpenCLIP-ViT/H](laion/CLIP-ViT-H-14-laion2B-s32B-b79K) 中的文本编码器（注意：该文本编码器一共**24层**，实际只使用**23层**）。LAION 团队首先使用 **V1 版的策略**在 **512x512** 像素的图片上进行训练得到了一个基础版模型 [stabilityai/stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base)，然后他们还使用了 [v-objective](https://arxiv.org/abs/2202.00512) 策略，在基础模型之上进一步使用 **768x768** 分辨率的图片进行训练，得到了一个最终版的模型 [stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)。

___Tips___:
___为了方便国内用户下载使用及快速体验Stable Diffusion模型，我们在百度云(BOS)上提供了paddle版本的镜像权重。注意：为了使用该模型与权重，你必须接受该模型所要求的**License**，请访问huggingface的[runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 和 [stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2), 仔细阅读里面的**License**，然后签署该协议。___
___Stable Diffusion是基于以下的License:
The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.___

下面将以最近较为火热的 **🔥Stable Diffusion** 模型为例，来说明如何快速使用 **ppdiffusers**，在开始之前我们可以点开下面的折叠按钮，查看当前 Stable Diffusion 模型所支持的权重！

### PPDiffusers模型支持的权重

<details><summary>&emsp; Stable Diffusion 模型支持的权重（英文） </summary>

**我们只需要将下面的"xxxx"，替换成所需的权重名，即可快速使用！**
```python
from ppdiffusers import *

pipe_text2img = StableDiffusionPipeline.from_pretrained("xxxx")
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained("xxxx")
pipe_inpaint_legacy = StableDiffusionInpaintPipelineLegacy.from_pretrained("xxxx")
pipe_mega = StableDiffusionMegaPipeline.from_pretrained("xxxx")

# pipe_mega.text2img() 等于 pipe_text2img()
# pipe_mega.img2img() 等于 pipe_img2img()
# pipe_mega.inpaint_legacy() 等于 pipe_inpaint_legacy()
```

| ppdiffusers支持的模型名称                     | 支持加载的Pipeline                                    | 备注 | huggingface.co地址 |
| :-------------------------------------------: | :--------------------------------------------------------------------: | --- | :-----------------------------------------: |
| CompVis/stable-diffusion-v1-4           | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | Stable-Diffusion-v1-4 使用 Stable-Diffusion-v1-2 的权重进行初始化。随后在"laion-aesthetics v2 5+"数据集上以 **512x512** 分辨率微调了 **225k** 步数，对文本使用了 **10%** 的dropout（即：训练过程中文图对中的文本有 10% 的概率会变成空文本）。模型使用了[CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)作为文本编码器。| [地址](https://huggingface.co/CompVis/stable-diffusion-v1-4) |
| CompVis/ldm-text2im-large-256               | LDMTextToImagePipeline | [LDM论文](https://arxiv.org/pdf/2112.10752.pdf) LDM-KL-8-G* 权重。| [地址](https://huggingface.co/CompVis/ldm-text2im-large-256) |
| CompVis/ldm-super-resolution-4x-openimages  | LDMSuperResolutionPipeline | [LDM论文](https://arxiv.org/pdf/2112.10752.pdf) LDM-VQ-4 权重，[原始权重链接](https://ommer-lab.com/files/latent-diffusion/sr_bsr.zip)。| [地址](https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages) |
| runwayml/stable-diffusion-v1-5              | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | Stable-Diffusion-v1-5 使用 Stable-Diffusion-v1-2 的权重进行初始化。随后在"laion-aesthetics v2 5+"数据集上以 **512x512** 分辨率微调了 **595k** 步数，对文本使用了 **10%** 的dropout（即：训练过程中文图对中的文本有 10% 的概率会变成空文本）。模型同样也使用了[CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)作为文本编码器。| [地址](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| runwayml/stable-diffusion-inpainting        | StableDiffusionInpaintPipeline | Stable-Diffusion-Inpainting 使用 Stable-Diffusion-v1-2 的权重进行初始化。首先进行了 **595k** 步的常规训练（实际也就是 Stable-Diffusion-v1-5 的权重），然后进行了 **440k** 步的 inpainting 修复训练。对于 inpainting 修复训练，给 UNet 额外增加了 **5** 输入通道（其中 **4** 个用于被 Mask 遮盖住的图片，**1** 个用于 Mask 本身）。在训练期间，会随机生成 Mask，并有 **25%** 概率会将原始图片全部 Mask 掉。| [地址](https://huggingface.co/runwayml/stable-diffusion-inpainting) |
| stabilityai/stable-diffusion-2-base         | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 该模型首先在 [LAION-5B 256x256 子集上](https://laion.ai/blog/laion-5b/) （过滤条件：[punsafe = 0.1 的 LAION-NSFW 分类器](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) 和 审美分数大于等于 4.5 ）从头开始训练 **550k** 步，然后又在分辨率 **>= 512x512** 的同一数据集上进一步训练 **850k** 步。| [地址](https://huggingface.co/stabilityai/stable-diffusion-2-base) |
| stabilityai/stable-diffusion-2              | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | stable-diffusion-2 使用 stable-diffusion-2-base 权重进行初始化，首先在同一数据集上（**512x512** 分辨率）使用 [v-objective](https://arxiv.org/abs/2202.00512) 训练了 **150k** 步。然后又在 **768x768** 分辨率上使用 [v-objective](https://arxiv.org/abs/2202.00512) 继续训练了 **140k** 步。| [地址](https://huggingface.co/stabilityai/stable-diffusion-2) |
| stabilityai/stable-diffusion-2-inpainting   | StableDiffusionInpaintPipeline |stable-diffusion-2-inpainting 使用 stable-diffusion-2-base 权重初始化，并且额外训练了 **200k** 步。训练过程使用了 [LAMA](https://github.com/saic-mdal/lama) 中提出的 Mask 生成策略，并且使用 Mask 图片的 Latent 表示（经过 VAE 编码）作为附加条件。| [地址](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) |
| stabilityai/stable-diffusion-x4-upscaler    | StableDiffusionUpscalePipeline | 该模型在**LAION 10M** 子集上（>2048x2048）训练了 1.25M 步。该模型还在分辨率为 **512x512** 的图像上使用 [Text-guided Latent Upscaling Diffusion Model](https://arxiv.org/abs/2112.10752) 进行了训练。除了**文本输入**之外，它还接收 **noise_level** 作为输入参数，因此我们可以使用 [预定义的 Scheduler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/configs/stable-diffusion/x4-upscaling.yaml) 向低分辨率的输入图片添加噪声。| [地址](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
</details>
<details><summary>&emsp; Stable Diffusion 模型支持的权重（中文和多语言） </summary>


| ppdiffusers支持的模型名称                     | 支持加载的Pipeline                                    | 备注 | huggingface.co地址 |
| :-------------------------------------------: | :--------------------------------------------------------------------: | --- | :-----------------------------------------: |
| BAAI/AltDiffusion                           | AltDiffusionPipeline、AltDiffusionImg2ImgPipeline | 该模型使用 [AltCLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/README.md) 作为文本编码器，在 Stable Diffusion 基础上训练了**双语Diffusion模型**，其中训练数据来自 [WuDao数据集](https://data.baai.ac.cn/details/WuDaoCorporaText) 和 [LAION](https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus) 。| [地址](https://huggingface.co/BAAI/AltDiffusion) |
| BAAI/AltDiffusion-m9                        | AltDiffusionPipeline、AltDiffusionImg2ImgPipeline |该模型使用9种语言的 [AltCLIP-m9](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/README.md) 作为文本编码器，其他同上。| [地址](https://huggingface.co/BAAI/AltDiffusion-m9) |
| IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 他们将 [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/) 数据集 (100M) 和 [Zero](https://zero.so.com/) 数据集 (23M) 用作预训练的数据集，先用 [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) 对这两个数据集的图文对相似性进行打分，取 CLIP Score 大于 0.2 的图文对作为训练集。 他们使用 [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) 作为初始化的text encoder，冻住 [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) ([论文](https://arxiv.org/abs/2112.10752)) 模型的其他部分，只训练 text encoder，以便保留原始模型的生成能力且实现中文概念的对齐。该模型目前在0.2亿图文对上训练了一个 epoch。 在 32 x A100 上训练了大约100小时，该版本只是一个初步的版本。| [地址](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1) |
| IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1 | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 他们将 [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/) 数据集 (100M) 和 [Zero](https://zero.so.com/) 数据集 (23M) 用作预训练的数据集，先用 [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) 对这两个数据集的图文对相似性进行打分，取 CLIP Score 大于 0.2 的图文对作为训练集。 他们使用 [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) ([论文](https://arxiv.org/abs/2112.10752)) 模型进行继续训练，其中训练分为**两个stage**。**第一个stage** 中冻住模型的其他部分，只训练 text encoder ，以便保留原始模型的生成能力且实现中文概念的对齐。**第二个stage** 中将全部模型解冻，一起训练 text encoder 和 diffusion model ，以便 diffusion model 更好的适配中文引导。第一个 stage 他们训练了 80 小时，第二个 stage 训练了 100 小时，两个stage都是用了8 x A100，该版本是一个初步的版本。| [地址](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1) |
</details>


### 文生图 （Text-to-Image Generation）

```python
import paddle
from ppdiffusers import StableDiffusionPipeline

# 可选模型权重
# CompVis/stable-diffusion-v1-4
# runwayml/stable-diffusion-v1-5
# stabilityai/stable-diffusion-2-base （原始策略 512x512）
# stabilityai/stable-diffusion-2 （v-objective 768x768）
# Linaqruf/anything-v3.0
# ......
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")

# 设置随机种子，我们可以复现下面的结果！
paddle.seed(5232132133)
prompt = "a portrait of shiba inu with a red cap growing on its head. intricate. lifelike. soft light. sony a 7 r iv 5 5 mm. cinematic post - processing "
image = pipe(prompt, guidance_scale=7.5, height=768, width=768).images[0]

image.save("shiba_dog_with_a_red_cap.png")
```
<div align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/50394665/204796701-d7911f76-8670-47d5-8d1b-8368b046c5e4.png">
</div>

### 基于文本引导的图生图（Image-to-Image Text-Guided Generation）

<details><summary>&emsp;Image-to-Image Text-Guided Generation Demo </summary>

```python
import paddle
from ppdiffusers import StableDiffusionImg2ImgPipeline
from ppdiffusers.utils import load_image

# 可选模型权重
# CompVis/stable-diffusion-v1-4
# runwayml/stable-diffusion-v1-5
# stabilityai/stable-diffusion-2-base （原始策略 512x512）
# stabilityai/stable-diffusion-2 （v-objective 768x768）
# Linaqruf/anything-v3.0
# ......
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("Linaqruf/anything-v3.0", safety_checker=None)

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/image_Kurisu.png"
image = load_image(url).resize((512, 768))

# 设置随机种子，我们可以复现下面的结果！
paddle.seed(42)
prompt = "Kurisu Makise, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, strength=0.75, guidance_scale=7.5).images[0]
image.save("image_Kurisu_img2img.png")
```
<div align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/50394665/204799529-cd89dcdb-eb1d-4247-91ac-b0f7bad777f8.png">
</div>
</details>

### 基于文本引导的图像编辑（Text-Guided Image Inpainting）

注意！当前有两种版本的图像编辑代码，一个是Legacy版本，一个是正式版本，下面将分别介绍两种代码如何使用！

<details><summary>&emsp;Legacy版本代码</summary>

```python
import paddle
from ppdiffusers import StableDiffusionInpaintPipelineLegacy
from ppdiffusers.utils import load_image

# 可选模型权重
# CompVis/stable-diffusion-v1-4
# runwayml/stable-diffusion-v1-5
# stabilityai/stable-diffusion-2-base （原始策略 512x512）
# stabilityai/stable-diffusion-2 （v-objective 768x768）
# Linaqruf/anything-v3.0
# ......
img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("stabilityai/stable-diffusion-2-base", safety_checker=None)

# 设置随机种子，我们可以复现下面的结果！
paddle.seed(10245)
prompt = "a red cat sitting on a bench"
image = pipe(prompt=prompt, image=image, mask_image=mask_image, strength=0.75).images[0]

image.save("a_red_cat_legacy.png")
```
<div align="center">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/50394665/204802186-5a6d302b-83aa-4247-a5bb-ebabfcc3abc4.png">
</div>

</details>

<details><summary>&emsp;正式版本代码</summary>

Tips: 下面的使用方法是新版本的代码，也是官方推荐的代码，注意必须配合 **runwayml/stable-diffusion-inpainting** 和 **stabilityai/stable-diffusion-2-inpainting** 才可正常使用。
```python
import paddle
from ppdiffusers import StableDiffusionInpaintPipeline
from ppdiffusers.utils import load_image

# 可选模型权重
# runwayml/stable-diffusion-inpainting
# stabilityai/stable-diffusion-2-inpainting
img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")

# 设置随机种子，我们可以复现下面的结果！
paddle.seed(1024)
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

image.save("a_yellow_cat.png")
```
<div align="center">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/50394665/204801946-6cd043bc-f3db-42cf-82cd-6a6171484523.png">
</div>
</details>

### 基于文本引导的图像放大 & 图像超分（Text-Guided Image Upscaling & Super Superresolution）

<details><summary>&emsp;Text-Guided Image Upscaling Demo</summary>

```python
import paddle
from ppdiffusers import StableDiffusionUpscalePipeline
from ppdiffusers.utils import load_image

pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png"
# 我们人工将原始图片缩小成 128x128 分辨率，最终保存的图片会放大4倍！
low_res_img = load_image(url).resize((128, 128))

prompt = "a white cat"
image = pipe(prompt=prompt, image=low_res_img).images[0]

image.save("upscaled_white_cat.png")
```
<div align="center">
<img width="200" alt="image" src="https://user-images.githubusercontent.com/50394665/204806180-b7f1b9cf-8a62-4577-b5c4-91adda08a13b.png">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/50394665/204806202-8c110be3-5f48-4946-95ea-21ad5a9a2340.png">
</div>
</details>

<details><summary>&emsp;Super Superresolution Demo</summary>

```python
import paddle
from ppdiffusers import LDMSuperResolutionPipeline
from ppdiffusers.utils import load_image

pipe = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"

# 我们人工将原始图片缩小成 128x128 分辨率，最终保存的图片会放大4倍！
low_res_img = load_image(url).resize((128, 128))

image = pipe(image=low_res_img, num_inference_steps=100).images[0]

image.save("ldm-super-resolution-image.png")
```
<div align="center">
<img width="200" alt="image" src="https://user-images.githubusercontent.com/50394665/204804426-5e28b571-aa41-4f56-ba26-68cca75fdaae.png">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/50394665/204804148-fe7c293b-6cd7-4942-ae9c-446369fe8410.png">
</div">
</details>

## 模型部署
StableDiffusion模型除了**支持Paddle动态图**运行，还支持将模型导出并使用推理引擎运行。我们提供在 [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) 上的 **StableDiffusion** 模型文生图、图生图、图像编辑等任务的部署示例，用户可以按照我们提供 [StableDiffusion模型导出教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/deploy/export.md) 将模型导出 或者使用 [一键导出脚本](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/scripts/convert_diffusers_model/convert_ppdiffusers_stable_diffusion_to_fastdeploy.py) 导出模型，然后使用我们提供的`FastDeployStableDiffusionMegaPipeline`进行高性能推理部署！

<details><summary>&emsp; 已预先导出的FastDeploy版Stable Diffusion权重 </summary>

**注意：当前导出的vae encoder带有随机因素！[随机因素代码地址](https://github.com/PaddlePaddle/PaddleNLP/blob/649b18a1834163007358e3a9dffd6462c0f9c7cf/ppdiffusers/ppdiffusers/models/vae.py#L365-L370)**

- CompVis/stable-diffusion-v1-4@fastdeploy
- runwayml/stable-diffusion-v1-5@fastdeploy
- runwayml/stable-diffusion-inpainting@fastdeploy
- stabilityai/stable-diffusion-2-base@fastdeploy
- stabilityai/stable-diffusion-2@fastdeploy
- stabilityai/stable-diffusion-2-inpainting@fastdeploy
- Linaqruf/anything-v3.0@fastdeploy
- hakurei/waifu-diffusion-v1-3@fastdeploy

</details>

```python
import fastdeploy as fd
from ppdiffusers import FastDeployStableDiffusionMegaPipeline
from ppdiffusers.utils import load_image

def create_runtime_option(device_id=-1, backend="paddle"):
    option = fd.RuntimeOption()
    if backend == "paddle":
        option.use_paddle_backend()
    else:
        option.use_ort_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    return option

runtime_options = {
    "text_encoder": create_runtime_option(-1, "onnx"),  # use cpu
    "vae_encoder": create_runtime_option(-1, "paddle"),  # use cpu
    "vae_decoder": create_runtime_option(-1, "paddle"),  # use cpu
    "unet": create_runtime_option(0, "paddle"),  # use gpu
}

fd_pipe = FastDeployStableDiffusionMegaPipeline.from_pretrained(
    "Linaqruf/anything-v3.0@fastdeploy", runtime_options=runtime_options
)

# text2img
prompt = "a portrait of shiba inu with a red cap growing on its head. intricate. lifelike. soft light. sony a 7 r iv 5 5 mm. cinematic post - processing "
image_text2img = fd_pipe.text2img(prompt=prompt, num_inference_steps=50).images[0]
image_text2img.save("image_text2img.png")

# img2img
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/image_Kurisu.png"
image = load_image(url).resize((512, 512))
prompt = "Kurisu Makise, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

image_img2img = fd_pipe.img2img(
    prompt=prompt, negative_prompt=negative_prompt, image=image, strength=0.75, guidance_scale=7.5
).images[0]
image_img2img.save("image_img2img.png")

# inpaint_legacy
img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))
prompt = "a red cat sitting on a bench"

image_inpaint_legacy = fd_pipe.inpaint_legacy(
    prompt=prompt, image=image, mask_image=mask_image, strength=0.75, num_inference_steps=50
).images[0]
image_inpaint_legacy.save("image_inpaint_legacy.png")
```
<div align="center">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/50394665/205297240-46b80992-34af-40cd-91a6-ae76589d0e21.png">
</div>

## Credits

This library concretizes previous work by many different authors and would not have been possible without their great research and implementations. We'd like to thank, in particular, the following implementations which have helped us in our development and without which the API could not have been as polished today:
- @huggingface' diffusers library, available [here](https://github.com/huggingface/diffusers)
- @CompVis' latent diffusion models library, available [here](https://github.com/CompVis/latent-diffusion)
- @hojonathanho original DDPM implementation, available [here](https://github.com/hojonathanho/diffusion) as well as the extremely useful translation into PyTorch by @pesser, available [here](https://github.com/pesser/pytorch_diffusion)
- @ermongroup's DDIM implementation, available [here](https://github.com/ermongroup/ddim).
- @yang-song's Score-VE and Score-VP implementations, available [here](https://github.com/yang-song/score_sde_pytorch)

We also want to thank @heejkoo for the very helpful overview of papers, code and resources on diffusion models, available [here](https://github.com/heejkoo/Awesome-Diffusion-Models) as well as @crowsonkb and @rromb for useful discussions and insights.

## Citation

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

## License

PPDiffusers遵循[Apache-2.0开源协议](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/LICENSE)。
