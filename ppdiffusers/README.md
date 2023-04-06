<div align="center">
  <img src="https://user-images.githubusercontent.com/11793384/215372703-4385f66a-abe4-44c7-9626-96b7b65270c8.png" width="40%" height="40%" />
</div>

<p align="center">
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/pyversions/paddlenlp"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/blob/develop/LICENSE"><img src="https://img.shields.io/github/license/paddlepaddle/paddlenlp"></a>
</p>

<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#模型部署> 模型部署</a>
</h4>

# PPDiffusers: Diffusers toolbox implemented based on PaddlePaddle

**PPDiffusers**是一款支持多种模态（如文本图像跨模态、图像、语音）扩散模型（Diffusion Model）训练和推理的国产化工具箱，依托于[**PaddlePaddle**](https://www.paddlepaddle.org.cn/)框架和[**PaddleNLP**](https://github.com/PaddlePaddle/PaddleNLP)自然语言处理开发库。

## News 📢

* 🔥 **2023.03.29 发布 0.14.0 版本，新增[LoRA](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/dreambooth)、[ControlNet](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/controlnet)，支持训练与推理；
模型加载升级，[可直接加载HF Diffusers的权重](#加载HF-Diffusers权重)（safetensors和pt）或 [SD等原库的Lightning权重进行推理](#加载原库的Lightning权重)，[支持加载Civitai社区的LoRA权重](#加载Civitai社区的LoRA权重)；
[支持xformers](#XFormers加速) 训练与推理；
新增用于超高分辨率生成的VAE tiling；
新增Instruct Pix2Pix、Semantic guidance、Depth2image等模型。**


* 🔥 **2023.01.18 发布 0.11.0 版本，新增Heun和Single step DPM-Solver噪声调度器，支持Karlo UnCLIP、Paint-by-example、Depth-Guided Stable Diffusion等图像生成扩散模型， 支持Audio Diffusion音频生成扩散模型。**


## 特性
#### 📦 SOTA扩散模型Pipelines集合
我们提供**SOTA（State-of-the-Art）** 的扩散模型Pipelines集合。
目前**PPDiffusers**已经集成了**50+Pipelines**，支持文图生成（Text-to-Image Generation）、文本引导的图像编辑（Text-Guided Image Inpainting）、文本引导的图像变换（Image-to-Image Text-Guided Generation）、超分（Super Superresolution）在内的10+任务，覆盖文本图像跨模态、图像、音频等多种模态。
如果想要了解当前支持的所有**Pipelines**以及对应的来源信息，可以阅读[🔥 PPDiffusers Pipelines](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/pipelines/README.md)文档。


#### 🔊 提供丰富的Noise Scheduler
我们提供了丰富的**噪声调度器（Noise Scheduler）**，可以对**速度**与**质量**进行权衡，用户可在推理时根据需求快速切换使用。
当前**PPDiffusers**已经集成了**14+Scheduler**，不仅支持 [DDPM](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_ddpm.py)、[DDIM](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_ddim.py) 和 [PNDM](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_pndm.py)，还支持最新的 [🔥 DPMSolver](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_dpmsolver_multistep.py)！

#### 🎛️ 提供多种扩散模型组件
我们提供了**多种扩散模型**组件，如[UNet1DModel](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/models/unet_1d.py)、[UNet2DModel](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/models/unet_2d.py)、[UNet2DConditionModel](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/models/unet_2d_condition.py)、[VQModel](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/models/vae.py)、[AutoencoderKL](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/models/vae.py)等。

#### 📖 提供丰富的训练和推理教程
我们提供了丰富的训练教程，不仅支持扩散模型的二次开发微调，如基于[Textual Inversion](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/textual_inversion)和[DreamBooth](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/dreambooth)使用3-5张图定制化训练生成图像的风格或物体，还支持使用[Laion400M](https://laion.ai/blog/laion-400-open-dataset)数据集[🔥 从零训练Latent Diffusion Model](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/text_to_image_laion400m) 模型！
此外，我们还提供了丰富的[🔥 Pipelines推理脚本](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/inference)。

#### 🚀 支持FastDeploy高性能部署
我们提供基于[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)的[🔥 高性能Stable Diffusion Pipeline](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion/pipeline_fastdeploy_stable_diffusion.py)，更多有关FastDeploy进行多推理引擎后端高性能部署的信息请参考[🔥 高性能FastDeploy推理教程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/deploy)。
```python
from ppdiffusers import StableDiffusionPipeline, FastDeployStableDiffusionPipeline

orig_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
fd_pipe = FastDeployStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5@fastdeploy")
```

## 安装

### 环境依赖
```
pip install -r requirements.txt
```
关于PaddlePaddle安装的详细教程请查看[Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)。

### pip安装

```shell
pip install --upgrade ppdiffusers -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### 手动安装
```shell
git clone https://github.com/PaddlePaddle/PaddleNLP
# 注意：如果clone仓库非常慢的话，可以考虑使用镜像版本
# git clone https://gitee.com/paddlepaddle/PaddleNLP
cd PaddleNLP/ppdiffusers
python setup.py install
```

## 快速开始
我们将以扩散模型的典型代表**Stable Diffusion**为例，带你快速了解PPDiffusers。

**Stable Diffusion**基于**潜在扩散模型（Latent Diffusion Models）**，专门用于**文图生成（Text-to-Image Generation）任务**。该模型是由来自 [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/), [LAION](https://laion.ai/)以及[RunwayML](https://runwayml.com/)的工程师共同开发完成，目前发布了v1和v2两个版本。v1版本采用了LAION-5B数据集子集（分辨率为 512x512）进行训练，并具有以下架构设置：自动编码器下采样因子为8，UNet大小为860M，文本编码器为CLIP ViT-L/14。v2版本相较于v1版本在生成图像的质量和分辨率等进行了改善。

### Stable Diffusion重点模型权重

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

| PPDiffusers支持的模型名称                     | 支持加载的Pipeline                                    | 备注 | huggingface.co地址 |
| :-------------------------------------------: | :--------------------------------------------------------------------: | --- | :-----------------------------------------: |
| CompVis/stable-diffusion-v1-4           | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | Stable-Diffusion-v1-4 使用 Stable-Diffusion-v1-2 的权重进行初始化。随后在"laion-aesthetics v2 5+"数据集上以 **512x512** 分辨率微调了 **225k** 步数，对文本使用了 **10%** 的dropout（即：训练过程中文图对中的文本有 10% 的概率会变成空文本）。模型使用了[CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)作为文本编码器。| [地址](https://huggingface.co/CompVis/stable-diffusion-v1-4) |
| CompVis/ldm-text2im-large-256               | LDMTextToImagePipeline | [LDM论文](https://arxiv.org/pdf/2112.10752.pdf) LDM-KL-8-G* 权重。| [地址](https://huggingface.co/CompVis/ldm-text2im-large-256) |
| CompVis/ldm-super-resolution-4x-openimages  | LDMSuperResolutionPipeline | [LDM论文](https://arxiv.org/pdf/2112.10752.pdf) LDM-VQ-4 权重，[原始权重链接](https://ommer-lab.com/files/latent-diffusion/sr_bsr.zip)。| [地址](https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages) |
| runwayml/stable-diffusion-v1-5              | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | Stable-Diffusion-v1-5 使用 Stable-Diffusion-v1-2 的权重进行初始化。随后在"laion-aesthetics v2 5+"数据集上以 **512x512** 分辨率微调了 **595k** 步数，对文本使用了 **10%** 的dropout（即：训练过程中文图对中的文本有 10% 的概率会变成空文本）。模型同样也使用了[CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)作为文本编码器。| [地址](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| runwayml/stable-diffusion-inpainting        | StableDiffusionInpaintPipeline | Stable-Diffusion-Inpainting 使用 Stable-Diffusion-v1-2 的权重进行初始化。首先进行了 **595k** 步的常规训练（实际也就是 Stable-Diffusion-v1-5 的权重），然后进行了 **440k** 步的 inpainting 修复训练。对于 inpainting 修复训练，给 UNet 额外增加了 **5** 输入通道（其中 **4** 个用于被 Mask 遮盖住的图片，**1** 个用于 Mask 本身）。在训练期间，会随机生成 Mask，并有 **25%** 概率会将原始图片全部 Mask 掉。| [地址](https://huggingface.co/runwayml/stable-diffusion-inpainting) |
| stabilityai/stable-diffusion-2-base         | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 该模型首先在 [LAION-5B 256x256 子集上](https://laion.ai/blog/laion-5b/) （过滤条件：[punsafe = 0.1 的 LAION-NSFW 分类器](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) 和 审美分数大于等于 4.5 ）从头开始训练 **550k** 步，然后又在分辨率 **>= 512x512** 的同一数据集上进一步训练 **850k** 步。| [地址](https://huggingface.co/stabilityai/stable-diffusion-2-base) |
| stabilityai/stable-diffusion-2              | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | stable-diffusion-2 使用 stable-diffusion-2-base 权重进行初始化，首先在同一数据集上（**512x512** 分辨率）使用 [v-objective](https://arxiv.org/abs/2202.00512) 训练了 **150k** 步。然后又在 **768x768** 分辨率上使用 [v-objective](https://arxiv.org/abs/2202.00512) 继续训练了 **140k** 步。| [地址](https://huggingface.co/stabilityai/stable-diffusion-2) |
| stabilityai/stable-diffusion-2-inpainting   | StableDiffusionInpaintPipeline |stable-diffusion-2-inpainting 使用 stable-diffusion-2-base 权重初始化，并且额外训练了 **200k** 步。训练过程使用了 [LAMA](https://github.com/saic-mdal/lama) 中提出的 Mask 生成策略，并且使用 Mask 图片的 Latent 表示（经过 VAE 编码）作为附加条件。| [地址](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) |
| stabilityai/stable-diffusion-x4-upscaler    | StableDiffusionUpscalePipeline | 该模型在**LAION 10M** 子集上（>2048x2048）训练了 1.25M 步。该模型还在分辨率为 **512x512** 的图像上使用 [Text-guided Latent Upscaling Diffusion Model](https://arxiv.org/abs/2112.10752) 进行了训练。除了**文本输入**之外，它还接收 **noise_level** 作为输入参数，因此我们可以使用 [预定义的 Scheduler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/low_res_scheduler/scheduler_config.json) 向低分辨率的输入图片添加噪声。| [地址](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| hakurei/waifu-diffusion    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | waifu-diffusion-v1-2 使用 stable-diffusion-v1-4 权重初始化，并且在**高质量动漫**图像数据集上进行微调后得到的模型。用于微调的数据是 **680k** 文本图像样本，这些样本是通过 **booru 网站** 下载的。| [地址](https://huggingface.co/hakurei/waifu-diffusion) |
| hakurei/waifu-diffusion-v1-3    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | waifu-diffusion-v1-3 是 waifu-diffusion-v1-2 基础上进一步训练得到的。他们对数据集进行了额外操作：（1）删除下划线；（2）删除括号；（3）用逗号分隔每个booru 标签；（4）随机化标签顺序。| [地址](https://huggingface.co/hakurei/waifu-diffusion) |
| naclbit/trinart_stable_diffusion_v2_60k    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | trinart_stable_diffusion 使用 stable-diffusion-v1-4 权重初始化，在 40k **高分辨率漫画/动漫风格**的图片数据集上微调了 8 个 epoch。V2 版模型使用 **dropouts**、**10k+ 图像**和**新的标记策略**训练了**更长时间**。| [地址](https://huggingface.co/naclbit/trinart_stable_diffusion_v2) |
| naclbit/trinart_stable_diffusion_v2_95k    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | **95k** 步数的的结果，其他同上。| [地址](https://huggingface.co/naclbit/trinart_stable_diffusion_v2) |
| naclbit/trinart_stable_diffusion_v2_115k    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | **115k** 步数的的结果，其他同上。| [地址](https://huggingface.co/naclbit/trinart_stable_diffusion_v2) |
| Deltaadams/Hentai-Diffusion    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | None| [地址](https://huggingface.co/Deltaadams/Hentai-Diffusion) |
| ringhyacinth/nail-set-diffuser    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 美甲领域的扩散模型，训练数据使用了 [Weekend](https://weibo.com/u/5982308498)| [地址](https://huggingface.co/ringhyacinth/nail-set-diffuser) |
| Linaqruf/anything-v3.0    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 该模型可通过输入几个文本提示词就能生成**高质量、高度详细的动漫风格图片**，该模型支持使用 **danbooru 标签文本** 生成图像。| [地址](https://huggingface.co/Linaqruf/anything-v3.0) |

</details>
<details><summary>&emsp; Stable Diffusion 模型支持的权重（中文和多语言） </summary>


| PPDiffusers支持的模型名称                     | 支持加载的Pipeline                                    | 备注 | huggingface.co地址 |
| :-------------------------------------------: | :--------------------------------------------------------------------: | --- | :-----------------------------------------: |
| BAAI/AltDiffusion                           | AltDiffusionPipeline、AltDiffusionImg2ImgPipeline | 该模型使用 [AltCLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/README.md) 作为文本编码器，在 Stable Diffusion 基础上训练了**双语Diffusion模型**，其中训练数据来自 [WuDao数据集](https://data.baai.ac.cn/details/WuDaoCorporaText) 和 [LAION](https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus) 。| [地址](https://huggingface.co/BAAI/AltDiffusion) |
| BAAI/AltDiffusion-m9                        | AltDiffusionPipeline、AltDiffusionImg2ImgPipeline |该模型使用9种语言的 [AltCLIP-m9](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/README.md) 作为文本编码器，其他同上。| [地址](https://huggingface.co/BAAI/AltDiffusion-m9) |
| IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 他们将 [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/) 数据集 (100M) 和 [Zero](https://zero.so.com/) 数据集 (23M) 用作预训练的数据集，先用 [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) 对这两个数据集的图文对相似性进行打分，取 CLIP Score 大于 0.2 的图文对作为训练集。 他们使用 [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) 作为初始化的text encoder，冻住 [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) ([论文](https://arxiv.org/abs/2112.10752)) 模型的其他部分，只训练 text encoder，以便保留原始模型的生成能力且实现中文概念的对齐。该模型目前在0.2亿图文对上训练了一个 epoch。 在 32 x A100 上训练了大约100小时，该版本只是一个初步的版本。| [地址](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1) |
| IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1 | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 他们将 [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/) 数据集 (100M) 和 [Zero](https://zero.so.com/) 数据集 (23M) 用作预训练的数据集，先用 [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) 对这两个数据集的图文对相似性进行打分，取 CLIP Score 大于 0.2 的图文对作为训练集。 他们使用 [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) ([论文](https://arxiv.org/abs/2112.10752)) 模型进行继续训练，其中训练分为**两个stage**。**第一个stage** 中冻住模型的其他部分，只训练 text encoder ，以便保留原始模型的生成能力且实现中文概念的对齐。**第二个stage** 中将全部模型解冻，一起训练 text encoder 和 diffusion model ，以便 diffusion model 更好的适配中文引导。第一个 stage 他们训练了 80 小时，第二个 stage 训练了 100 小时，两个stage都是用了8 x A100，该版本是一个初步的版本。| [地址](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1) |
</details>


### 加载HF Diffusers权重
```python
from ppdiffusers import StableDiffusionPipeline
# 设置from_hf_hub为True，表示从huggingface hub下载，from_diffusers为True表示加载的是diffusers版Pytorch权重
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", from_hf_hub=True, from_diffusers=True)
```

### 加载原库的Lightning权重
```python
from ppdiffusers import StableDiffusionPipeline
# 可输入网址 或 本地ckpt、safetensors文件
pipe = StableDiffusionPipeline.from_pretrained_original_ckpt("https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/ppdiffusers/chilloutmix_NiPrunedFp32Fix.safetensors")
```

### 加载Civitai社区的LoRA权重
```python
from ppdiffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("TASUKU2023/Chilloutmix")
# 加载lora权重
pipe.apply_lora("https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/ppdiffusers/Moxin_10.safetensors")
```

### XFormers加速
为了使用**XFormers加速**，我们需要安装`develop`版本的`paddle`，Linux系统的安装命令如下：
```sh
python -m pip install paddlepaddle-gpu==0.0.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

```python
import paddle
from ppdiffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("TASUKU2023/Chilloutmix", paddle_dtype=paddle.float16)
# 开启xformers加速 默认选择"cutlass"加速
pipe.enable_xformers_memory_efficient_attention()
# flash 需要使用 A100、A10、3060、3070、3080、3090 等以上显卡。
# pipe.enable_xformers_memory_efficient_attention("flash")
```
### 文图生成 （Text-to-Image Generation）

```python
import paddle
from ppdiffusers import StableDiffusionPipeline

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

### 文本引导的图像变换（Image-to-Image Text-Guided Generation）

<details><summary>&emsp;Image-to-Image Text-Guided Generation Demo </summary>

```python
import paddle
from ppdiffusers import StableDiffusionImg2ImgPipeline
from ppdiffusers.utils import load_image

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

### 文本引导的图像编辑（Text-Guided Image Inpainting）

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

### 文本引导的图像放大 & 超分（Text-Guided Image Upscaling & Super-Resolution）

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

<details><summary>&emsp;Super-Resolution Demo</summary>

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
</div>

</details>

## 模型推理部署
除了**Paddle动态图**运行之外，很多模型还支持将模型导出并使用推理引擎运行。我们提供基于[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)上的**StableDiffusion**模型部署示例，涵盖文生图、图生图、图像编辑等任务，用户可以按照我们提供[StableDiffusion模型导出教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/deploy/export.md)将模型导出，然后使用`FastDeployStableDiffusionMegaPipeline`进行高性能推理部署！

<details><summary>&emsp; 已预先导出的FastDeploy版Stable Diffusion权重 </summary>

**注意：当前导出的vae encoder带有随机因素！**

- CompVis/stable-diffusion-v1-4@fastdeploy
- runwayml/stable-diffusion-v1-5@fastdeploy
- runwayml/stable-diffusion-inpainting@fastdeploy
- stabilityai/stable-diffusion-2-base@fastdeploy
- stabilityai/stable-diffusion-2@fastdeploy
- stabilityai/stable-diffusion-2-inpainting@fastdeploy
- Linaqruf/anything-v3.0@fastdeploy
- hakurei/waifu-diffusion-v1-3@fastdeploy

</details>

<details><summary>&emsp; FastDeploy Demo </summary>

```python
import paddle
import fastdeploy as fd
from ppdiffusers import FastDeployStableDiffusionMegaPipeline
from ppdiffusers.utils import load_image

def create_runtime_option(device_id=0, backend="paddle", use_cuda_stream=True):
    option = fd.RuntimeOption()
    if backend == "paddle":
        option.use_paddle_backend()
    else:
        option.use_ort_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
        if use_cuda_stream:
            paddle_stream = paddle.device.cuda.current_stream(device_id).cuda_stream
            option.set_external_raw_stream(paddle_stream)
    return option

runtime_options = {
    "text_encoder": create_runtime_option(0, "paddle"),  # use gpu:0
    "vae_encoder": create_runtime_option(0, "paddle"),  # use gpu:0
    "vae_decoder": create_runtime_option(0, "paddle"),  # use gpu:0
    "unet": create_runtime_option(0, "paddle"),  # use gpu:0
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
</details>
<div align="center">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/50394665/205297240-46b80992-34af-40cd-91a6-ae76589d0e21.png">
</div>



## License
PPDiffusers 遵循 [Apache-2.0开源协议](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/ppdiffusers/LICENSE)。

Stable Diffusion 遵循 [The CreativeML OpenRAIL M 开源协议](https://huggingface.co/spaces/CompVis/stable-diffusion-license)。
> The CreativeML OpenRAIL M is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which this license is based.

## Acknowledge
我们借鉴了🤗 Hugging Face的[Diffusers](https://github.com/huggingface/diffusers)关于预训练扩散模型使用的优秀设计，在此对Hugging Face作者及其开源社区表示感谢。


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
