# Paddle版Diffusers

___Note___:
___这里是基于[paddlepaddle](https://www.paddlepaddle.org.cn/)和[paddlenlp](https://github.com/PaddlePaddle/PaddleNLP)实现的[Diffusers](https://github.com/huggingface/diffusers)仓库，感谢Huggingface团队开源如此优秀的项目。___


diffusers_paddle 提供跨多种模态（例如视觉和音频）的预训练扩散模型，并提供服务作为用于推理和训练扩散模型的模块化工具箱。详细有关介绍可以参考原版[Diffusers](https://github.com/huggingface/diffusers)仓库。

## 1. 安装
**使用 `pip` 安装** (并未支持，未来将会支持)

```bash
pip install --upgrade diffusers_paddle
```

**手动安装**
```bash
# 克隆paddlenlp仓库，注意：如果clone仓库非常慢的话，可以考虑使用镜像版本
git clone https://github.com/PaddlePaddle/PaddleNLP
# 切换目录，进入diffusers_paddle文件夹
cd PaddleNLP/examples/diffusers_paddle
# 安装diffusers_paddle
python setup.py install
```

## 2. 快速开始

为了快速上手使用该项目, 我们可以先阅读一下huggingface提供的两个notebooks (注意国内可能无法正常打开。):

- The [Getting started with Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) notebook, which showcases an end-to-end example of usage for diffusion models, schedulers and pipelines.
  Take a look at this notebook to learn how to use the pipeline abstraction, which takes care of everything (model, scheduler, noise handling) for you, and also to understand each independent building block in the library.
- The [Training a diffusers model](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) notebook summarizes diffusion models training methods. This notebook takes a step-by-step approach to training your
  diffusion models on an image dataset, with explanatory graphics.

## 3. 使用diffusers_paddle体验Stable Diffusion模型!

Stable Diffusion 是一个**文本到图像(text-to-image)**的**潜在扩散模型(latent diffusion model, ldm)**, 该模型是由来自[CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/), [LAION](https://laion.ai/) 的工程师以及 [RunwayML](https://runwayml.com/)一起开发而完成的。该模型使用了大小为**512x512**的[LAION-5B](https://laion.ai/blog/laion-5b/)数据集子集进行训练。该模型使用了Openai开源的**CLIP ViT-L/14** 文本编码器(text_encoder)来编码提示(prompt)文本，从而作为引导条件（注意该部分权重不进行训练）。该模型使用了Unet模型(860M参数)和text encoder（123M参数），并且可以在具有4GB显存（注：当前paddle版本需要进行优化，无法在4GB的显卡上运行）的GPU进行推理预测。

为了方便国内用户下载使用及快速体验Stable Diffusion模型，我们在百度云(BOS)上提供了paddle版本的镜像权重。注意：为了使用该模型与权重，你必须接收该模型所要求的**License**，请访问huggingface的[model card](https://huggingface.co/runwayml/stable-diffusion-v1-5), 仔细阅读里面的**License**，然后签署该协议。

Tips： Stable Diffusion是基于下面的License
> The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing. See also the article about the BLOOM Open RAIL license on which this license is based.

### 3.1 使用Stable Diffusion进行文本-图像的生成
```python
import paddle
from diffusers_paddle import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")
```
<center><image src="https://user-images.githubusercontent.com/50394665/197779466-04543823-8b83-41d6-94e8-146a7dac00d7.png" width="600"></center>

### 3.2 使用Stable Diffusion体验基于文本引导的图片-图片的生成

```python
import requests
import paddle
from PIL import Image
from io import BytesIO

from diffusers_paddle import StableDiffusionImg2ImgPipeline

# load the pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# let's download an initial image
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"
# 使用fp16加快生成速度
with paddle.amp.auto_cast(True):
    image = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images[0]

image.save("fantasy_landscape.png")
```

<center><image src="https://user-images.githubusercontent.com/50394665/197780044-34e6f8ca-6864-4c3d-bb99-28e0aadf867b.png" width="600"></center>

### 3.3 使用Stable Diffusion进行补全图片

Tips: 下面的使用方法是旧版本的代码。
```python
import paddle
from io import BytesIO

import requests
import PIL

from diffusers_paddle import StableDiffusionInpaintPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

prompt = "a cat sitting on a bench"
with paddle.amp.auto_cast(True):
    image = pipe(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75).images[0]

image.save("cat_on_bench.png")
```
<center><image src="https://user-images.githubusercontent.com/50394665/197783711-ab3caf2e-5a4d-4099-8d01-d6ca80ca8e78.png" width="600"></center>

Tips: 下面的使用方法是新版本的代码，也是官方推荐的代码，注意必须配合**runwayml/stable-diffusion-inpainting**才可正常使用。
```python
import PIL
import requests
from io import BytesIO

from diffusers_paddle import StableDiffusionInpaintPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

image.save("cat_on_bench_new.png")
```
<center><image src="https://user-images.githubusercontent.com/50394665/198016801-87cec13b-0d89-41c3-aedb-c89a43d76153.png" width="600"></center>

## 4. Credits

This library concretizes previous work by many different authors and would not have been possible without their great research and implementations. We'd like to thank, in particular, the following implementations which have helped us in our development and without which the API could not have been as polished today:
- @huggingface' diffusers library, available [here](https://github.com/huggingface/diffusers)
- @CompVis' latent diffusion models library, available [here](https://github.com/CompVis/latent-diffusion)
- @hojonathanho original DDPM implementation, available [here](https://github.com/hojonathanho/diffusion) as well as the extremely useful translation into PyTorch by @pesser, available [here](https://github.com/pesser/pytorch_diffusion)
- @ermongroup's DDIM implementation, available [here](https://github.com/ermongroup/ddim).
- @yang-song's Score-VE and Score-VP implementations, available [here](https://github.com/yang-song/score_sde_pytorch)

We also want to thank @heejkoo for the very helpful overview of papers, code and resources on diffusion models, available [here](https://github.com/heejkoo/Awesome-Diffusion-Models) as well as @crowsonkb and @rromb for useful discussions and insights.

## 5. Citation

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
